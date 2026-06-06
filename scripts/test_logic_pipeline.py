"""5-case parser test for logic puzzle answer parsing.

Run with:  poetry run python scripts/test_logic_pipeline.py
"""
from __future__ import annotations

import json
import sys

from src.domains.logic_puzzles.engine import parse_llm_answer_tagged, score_answer

SOLUTION = {
    "color": ["red", "green", "blue"],
    "pet":   ["cat", "dog", "fish"],
    "drink": ["tea", "coffee", "juice"],
}

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

failures = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global failures
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures += 1


# --- Case 1: Schema A, fenced JSON ---
print("Case 1: schema A in markdown fence")
raw1 = "```json\n" + json.dumps(SOLUTION) + "\n```"
result1, tag1 = parse_llm_answer_tagged(raw1)
check("returns dict", result1 is not None)
check("schema tag is attribute_dict", tag1 == "attribute_dict", f"got {tag1!r}")
check("scores perfectly", score_answer(SOLUTION, result1)[0] is True)

# --- Case 2: Schema A, bare JSON (no fence) ---
print("\nCase 2: schema A bare JSON")
raw2 = "Here is the answer:\n" + json.dumps(SOLUTION)
result2, tag2 = parse_llm_answer_tagged(raw2)
check("returns dict", result2 is not None)
check("schema tag is attribute_dict", tag2 == "attribute_dict", f"got {tag2!r}")
check("scores perfectly", score_answer(SOLUTION, result2)[0] is True)

# --- Case 3: Schema B position-record form (wrapped object) ---
print("\nCase 3: schema B position-record (wrapped object)")
rows = [
    {"position": 1, "color": "red",   "pet": "cat",  "drink": "tea"},
    {"position": 2, "color": "green", "pet": "dog",  "drink": "coffee"},
    {"position": 3, "color": "blue",  "pet": "fish", "drink": "juice"},
]
raw3 = json.dumps({"solution": rows})
result3, tag3 = parse_llm_answer_tagged(raw3)
check("returns dict", result3 is not None)
check("schema tag is position_record", tag3 == "position_record", f"got {tag3!r}")
check(
    "normalised correctly matches solution",
    result3 is not None and result3.get("color") == ["red", "green", "blue"],
    f"got color={result3.get('color') if result3 else None}",
)

# --- Case 4: Garbage input → None ---
print("\nCase 4: garbage input → None")
raw4 = "I could not determine the solution due to insufficient clues."
result4, tag4 = parse_llm_answer_tagged(raw4)
check("returns None", result4 is None)
check("tag is None", tag4 is None)

# --- Case 5: Schema B position-record scores non-zero (regression guard) ---
print("\nCase 5: schema B bare array scores correctly (not silent 0)")
raw5 = json.dumps(rows)  # bare [{position:1,...},...]
result5, tag5 = parse_llm_answer_tagged(raw5)
_, score5, _ = score_answer(SOLUTION, result5)
check("schema tag is position_record", tag5 == "position_record", f"got {tag5!r}")
check("score is 1.0 (not silent 0)", score5 == 1.0, f"got score={score5:.3f}")

# --- Summary ---
print(f"\n{'='*40}")
if failures == 0:
    print(f"All tests passed.")
else:
    print(f"{failures} test(s) FAILED.")
    sys.exit(1)
