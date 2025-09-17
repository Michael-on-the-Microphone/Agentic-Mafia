#!/usr/bin/env python3
"""
LLM Loop Experiment Runner (Ollama - llama3.2:3b)

Usage examples:
  # Baseline, 10 samples (no feedback), fixed seed:
  python llm_loop.py --mode baseline --scenario "You are an astronaut..." --samples 10 --seed 42

  # Looped, 15 iterations, perturb at turn 12:
  python llm_loop.py --mode loop --scenario-file scenario.txt --iterations 15 \
                     --perturb-at 12 --perturb-text "Oxygen levels drop by 20%."

  # Thoughts, 25 steps, show last 5 previous thoughts each turn:
  python llm_loop.py --mode thoughts --scenario "A Mars habitat..." --thoughts 25 --history-window 5

Outputs:
  - Writes a JSONL log (default: run_log.jsonl)
  - Prints compact summaries to stdout
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = "llama3.2:3b"

SYSTEM_PROMPT = (
    "ROLE: Structured JSON generator.\n"
    "OUTPUT CONTRACT:\n"
    "- Return ONE (1) JSON object and NOTHING ELSE. No prose, no backticks, no prefix/suffix.\n"
    "- Use only double-quoted keys/strings. No trailing commas. UTF-8 safe.\n"
    "- If a field has no content, output an empty list [], not null.\n"
    "- Never include explanations, markdown, or comments.\n"
    "QUALITY GATE (perform silently before responding):\n"
    "1) Does the output parse as strict JSON? 2) Keys exactly match the schema? 3) No extra keys?\n"
    "If any check fails, fix it BEFORE responding.\n"
)


# ======= Base schemas for baseline/loop =======
FORMAT_INSTRUCTIONS = """
You MUST output JSON that matches this schema EXACTLY:

{
  "notes": {
    "premises": ["<≤15 words>", "<≤15 words>", "<≤15 words>"],
    "hypotheses": ["<≤15 words>", "<≤15 words>"],
    "uncertainties": ["<≤15 words>", "<≤15 words>"],
    "plan_next": ["<≤15 words>", "<≤15 words>"]
  },
  "answer": "<2-3 sentences (≤45 words total)>"
}

CONSTRAINTS:
- Single JSON object only; no prose or markdown outside the JSON.
- Use empty lists [] when a section has no items. No nulls. No extra keys.
- Total words in notes ≤ 120.
"""


BASE_INSTRUCTIONS = (
    "INSTRUCTIONS:\n"
    "1) From the SCENARIO, infer key premises, uncertainties, and 1–2 next actions (plan_next).\n"
    "2) Keep items concise (≤15 words each). Keep notes ≤120 words total.\n"
    "3) Output MUST conform to the schema and be valid JSON.\n"
)


LOOP_INSTRUCTIONS = (
    "INSTRUCTIONS:\n"
    "- Reconcile PRIOR NOTES with the SCENARIO. Identify conflicts silently and update only if justified.\n"
    "- Adjust premises/hypotheses when evidence requires; otherwise keep them stable.\n"
    "- Keep items concise. Notes ≤120 words total. JSON only, per schema.\n"
)


# ======= Stepwise "thoughts" mode schema =======
THOUGHTS_FORMAT = """
You MUST output JSON that matches this schema EXACTLY:

{
  "next_thought": "<one sentence (5-18 words) that advances the reasoning>",
  "notes": {
    "premises": ["<≤15 words>", "<≤15 words>", "<≤15 words>"],
    "hypotheses": ["<≤15 words>", "<≤15 words>"],
    "uncertainties": ["<≤15 words>", "<≤15 words>"],
    "plan_next": ["<≤15 words>", "<≤15 words>"]
  }
}

RULES:
- Emit EXACTLY ONE sentence in next_thought; no lists or multiple sentences.
- Each list in notes may contain 0-3 items (premises up to 3; others up to 2).
- Keep total words in all notes ≤ 80. Prefer concrete, operational phrases.
- If you would repeat a prior thought, rephrase it to be more specific or produce the next incremental action instead.
"""


THOUGHTS_INSTRUCTIONS = (
    "TASK:\n"
    "- Read SCENARIO and the recent THOUGHT HISTORY.\n"
    "- Produce exactly one new thought that pushes the plan forward.\n"
    "- Update notes ONLY if justified by either the SCENARIO or a prior thought.\n"
    "\n"
    "CONSTRAINTS:\n"
    "- Output MUST be a single JSON object that matches the schema.\n"
    "- No extra keys. No commentary. No markdown. No code fences.\n"
    "- Use empty lists [] if you have no update for a notes field.\n"
)


# ======= Core helpers =======
def ollama_chat(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    seed: Optional[int] = 42,
    top_p: float = 0.95,
    repeat_penalty: float = 1.1,
    timeout: int = 120,
    json_only: bool = True,
) -> str:
    """Call Ollama /api/chat. Returns assistant text (not streamed)."""
    url = f"{OLLAMA_HOST}/api/chat"
    options = {
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "repeat_penalty": repeat_penalty,
    }
    if json_only:
        # Ollama supports forcing JSON
        options["format"] = "json"

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Ollama request failed: {e}", file=sys.stderr)
        sys.exit(1)


import re

def _extract_json_object(text: str) -> Optional[str]:
    """Try to grab the first valid top-level JSON object via bracket matching."""
    s = text.strip()
    # Remove fences like ```json ... ```
    if s.startswith("```"):
        s = s.strip("` \n")
        if s.lower().startswith("json"):
            s = s[4:].strip()
    # Bracket match the first {...}
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(s[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None

def parse_model_json(text: str) -> Dict[str, Any]:
    """Parse model output as JSON with multiple fallbacks."""
    # 1) direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) extract bracketed JSON
    candidate = _extract_json_object(text)
    if candidate:
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # 3) regex rescue for next_thought-only minimal case
    m = re.search(r'"next_thought"\s*:\s*"([^"]+)"', text)
    if m:
        return {"next_thought": m.group(1), "notes": {}}
    # Fail
    raise ValueError(f"Model did not return valid JSON.\nRaw:\n{text[:500]}...")


def compact_notes(notes: Dict[str, List[str]], max_items: int = 3) -> Dict[str, List[str]]:
    """Keep only top-N items per field to control growth."""
    out = {}
    for k in ["premises", "hypotheses", "uncertainties", "plan_next"]:
        arr = notes.get(k, [])
        if not isinstance(arr, list):
            arr = [str(arr)]
        out[k] = arr[:max_items]
    return out

def render_prior(notes: Dict[str, List[str]]) -> str:
    """Render prior notes section for the loop prompt."""
    def join(lst): return ", ".join(lst) if lst else ""
    return (
        "PRIOR NOTES (compressed):\n"
        f"Premises*: [{join(notes.get('premises', []))}]\n"
        f"Hypotheses*: [{join(notes.get('hypotheses', []))}]\n"
        f"Uncertainties*: [{join(notes.get('uncertainties', []))}]\n"
        f"Plan_next*: [{join(notes.get('plan_next', []))}]\n"
    )

def render_thought_history(thoughts: List[str], k: int) -> str:
    """Render the last k thoughts (most recent first)."""
    recent = thoughts[-k:] if k > 0 else []
    lines = "\n".join(f"- {t}" for t in reversed(recent))
    return "THOUGHT HISTORY (most recent first):\n" + (lines if lines else "- (none)")

def log_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ======= Baseline =======
def baseline_run(
    scenario: str,
    samples: int,
    temperature: float,
    seed: Optional[int],
    out_path: str,
    model: str = MODEL,
) -> None:
    print(f"== Baseline Run: {samples} samples, temp={temperature}, seed={seed}, model={model}")
    for i in range(samples):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"SCENARIO:\n{scenario}\n\n{BASE_INSTRUCTIONS}\n{FORMAT_INSTRUCTIONS}",
            },
        ]
        t0 = time.time()
        text = ollama_chat(model, messages, temperature=temperature, seed=seed, json_only=True)

        dt = time.time() - t0

        try:
            parsed = parse_model_json(text)
        except ValueError as e:
            print(f"[{i}] JSON parse error. See log. Proceeding.", file=sys.stderr)
            parsed = {"raw_text": text, "parse_error": str(e)}

        record = {
            "mode": "baseline",
            "iteration": i,
            "timestamp": datetime.utcnow().isoformat(),
            "model": model,
            "params": {"temperature": temperature, "seed": seed},
            "scenario": scenario,
            "prompt_kind": "baseline",
            "response": parsed,
            "latency_sec": round(dt, 3),
        }
        log_jsonl(out_path, record)

        # Console summary (avoid backslash in f-string expr)
        ans = parsed.get("answer", "") if isinstance(parsed, dict) else ""
        clean = (ans or "")[:160].replace("\n", " ")
        print(f"[{i}] {clean}")

# ======= Loop =======
def loop_run(
    scenario: str,
    iterations: int,
    temperature: float,
    seed: Optional[int],
    out_path: str,
    model: str = MODEL,
    perturb_at: Optional[int] = None,
    perturb_text: str = "",
) -> None:
    print(f"== Looped Run: {iterations} iterations, temp={temperature}, seed={seed}, model={model}")
    prior_notes: Dict[str, List[str]] = {"premises": [], "hypotheses": [], "uncertainties": [], "plan_next": []}
    current_scenario = scenario

    for k in range(iterations):
        change_log = ""
        if perturb_at is not None and k == perturb_at:
            current_scenario = scenario + f"\n\nCHANGE LOG (Iteration {k}): {perturb_text}"
            change_log = f"CHANGE LOG applied at iteration {k}: {perturb_text}"

        prior_section = render_prior(prior_notes)
        user_content = (
            f"SCENARIO:\n{current_scenario}\n\n"
            f"{prior_section}\n"
            f"{LOOP_INSTRUCTIONS}\n{FORMAT_INSTRUCTIONS}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        t0 = time.time()
        text = ollama_chat(model, messages, temperature=temperature, seed=seed, json_only=True)

        dt = time.time() - t0

        try:
            parsed = parse_model_json(text)
        except ValueError as e:
            print(f"[{k}] JSON parse error. See log. Proceeding.", file=sys.stderr)
            parsed = {"raw_text": text, "parse_error": str(e)}

        # Update prior notes (compressed)
        new_notes = parsed.get("notes", {}) if isinstance(parsed, dict) else {}
        if isinstance(new_notes, dict):
            prior_notes = compact_notes(new_notes, max_items=3)



        record = {
            "mode": "loop",
            "iteration": k,
            "timestamp": datetime.utcnow().isoformat(),
            "model": model,
            "params": {"temperature": temperature, "seed": seed},
            "scenario": current_scenario,
            "prompt_kind": "loop",
            "prior_notes": prior_notes,
            "response": parsed,
            "change_log": change_log,
            "latency_sec": round(dt, 3),
        }
        log_jsonl(out_path, record)

        # Console summary
        ans = parsed.get("answer", "") if isinstance(parsed, dict) else ""
        clean = (ans or "")[:160].replace("\n", " ")
        print(f"[{k}] {clean}")

# ======= Thoughts (stepwise, one action per turn) =======
def thoughts_run(
    scenario: str,
    total_thoughts: int,
    temperature: float,
    seed: Optional[int],
    out_path: str,
    model: str = MODEL,
    history_window: int = 5,
    perturb_at: Optional[int] = None,
    perturb_text: str = "",
) -> None:
    print(f"== Thoughts Run: {total_thoughts} steps, temp={temperature}, seed={seed}, model={model}")
    prior_notes: Dict[str, List[str]] = {"premises": [], "hypotheses": [], "uncertainties": [], "plan_next": []}
    thoughts: List[str] = []
    current_scenario = scenario

    for step in range(total_thoughts):
        change_log = ""
        if perturb_at is not None and step == perturb_at:
            current_scenario = scenario + f"\n\nCHANGE LOG (step {step}): {perturb_text}"
            change_log = f"CHANGE LOG applied at step {step}: {perturb_text}"

        history = render_thought_history(thoughts, history_window)
        user_content = (
            f"SCENARIO:\n{current_scenario}\n\n"
            f"{history}\n\n"
            "RESPONSE REQUIREMENTS:\n"
            "- Return ONE JSON object ONLY; no markdown, no text outside the JSON.\n"
            "- Follow the schema exactly; use empty lists [] when unknown.\n\n"
            f"{THOUGHTS_INSTRUCTIONS}\n{THOUGHTS_FORMAT}\n"
            "EXAMPLE (illustrative only—replace with your own content):\n"
            '{"next_thought":"Measure current habitat load and isolate non-critical circuits.",'
            '"notes":{"premises":["Partial power","Crew: 4","Supplies: 30 days"],'
            '"hypotheses":["Life support must be prioritized"],'
            '"uncertainties":["Storm duration"],'
            '"plan_next":["Read power telemetry","Switch to low-power mode"]}}\n"'
        )



        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        t0 = time.time()
        text = ollama_chat(model, messages, temperature=temperature, seed=seed, json_only=True)

        dt = time.time() - t0

        try:
            parsed = parse_model_json(text)
        except ValueError as e:
            print(f"[{step}] JSON parse error. See log. Proceeding.", file=sys.stderr)
            parsed = {"raw_text": text, "parse_error": str(e)}

        next_thought = ""
        if isinstance(parsed, dict):
            next_thought = str(parsed.get("next_thought", "")).strip()
            new_notes = parsed.get("notes", {}) if isinstance(parsed, dict) else {}
            if isinstance(new_notes, dict):
                prior_notes = compact_notes(new_notes, max_items=3)

        if next_thought:
            thoughts.append(next_thought)

        record = {
            "mode": "thoughts",
            "step": step,
            "timestamp": datetime.utcnow().isoformat(),
            "model": model,
            "params": {"temperature": temperature, "seed": seed},
            "scenario": current_scenario,
            "prompt_kind": "thoughts",
            "prior_notes": prior_notes,
            "next_thought": next_thought,
            "thoughts_so_far": len(thoughts),
            "history_window": history_window,
            "change_log": change_log,
            "latency_sec": round(dt, 3),
        }
        log_jsonl(out_path, record)

        # Console summary (one-line)
        clean = (next_thought or "")[:200].replace("\n", " ")
        print(f"[{step}] {clean}")

# ======= CLI =======
def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def main():
    ap = argparse.ArgumentParser(description="Run an LLM loop experiment using Ollama (llama3.2:3b).")
    ap.add_argument("--mode", choices=["baseline", "loop", "thoughts"], required=True, help="Experiment mode.")
    ap.add_argument("--scenario", type=str, help="Scenario text.")
    ap.add_argument("--scenario-file", type=str, help="Path to a file with the scenario.")

    # Baseline params
    ap.add_argument("--samples", type=int, default=10, help="Baseline: number of samples.")

    # Loop params
    ap.add_argument("--iterations", type=int, default=15, help="Loop: number of iterations.")
    ap.add_argument("--perturb-at", type=int, default=None, help="Loop/Thoughts: step/iteration index to apply perturbation.")
    ap.add_argument("--perturb-text", type=str, default="", help="Loop/Thoughts: text to append at perturbation.")

    # Thoughts params
    ap.add_argument("--thoughts", type=int, default=20, help="Thoughts mode: total number of thoughts.")
    ap.add_argument("--history-window", type=int, default=5, help="Thoughts mode: how many previous thoughts to show each step.")

    # Common sampling/model params
    ap.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (Ollama supports seed).")
    ap.add_argument("--model", type=str, default=MODEL, help="Ollama model name.")
    ap.add_argument("--out", type=str, default="run_log.jsonl", help="Output JSONL log path.")

    args = ap.parse_args()

    if not args.scenario and not args.scenario_file:
        print("Error: Provide --scenario or --scenario-file", file=sys.stderr)
        sys.exit(2)

    scenario = args.scenario or read_text_file(args.scenario_file)

    # Quick connectivity check
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Could not reach Ollama at {OLLAMA_HOST}. Is it running?\n{e}", file=sys.stderr)
        sys.exit(1)

    if args.mode == "baseline":
        baseline_run(
            scenario=scenario,
            samples=args.samples,
            temperature=args.temperature,
            seed=args.seed,
            out_path=args.out,
            model=args.model,
        )
    elif args.mode == "loop":
        loop_run(
            scenario=scenario,
            iterations=args.iterations,
            temperature=args.temperature,
            seed=args.seed,
            out_path=args.out,
            model=args.model,
            perturb_at=args.perturb_at,
            perturb_text=args.perturb_text,
        )
    else:  # thoughts
        thoughts_run(
            scenario=scenario,
            total_thoughts=args.thoughts,
            temperature=args.temperature,
            seed=args.seed,
            out_path=args.out,
            model=args.model,
            history_window=args.history_window,
            perturb_at=args.perturb_at,
            perturb_text=args.perturb_text,
        )

    print(f"\nDone. Log saved to: {args.out}")

if __name__ == "__main__":
    main()
