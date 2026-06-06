from __future__ import annotations

import ast
import functools
import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LogicPuzzle:
	puzzle_id: str
	clues: str
	solution: dict[str, list[str]]
	num_positions: int
	num_attributes: int
	difficulty: str
	metadata: dict[str, Any] = field(default_factory=dict)


@functools.lru_cache(maxsize=1)
def _load_raw_records() -> list[dict[str, str]]:
	"""Load MysteryZebra once per process and return plain dict records."""
	from datasets import load_dataset

	dataset = load_dataset("arg-tech/MysteryZebra", split="train")
	return [
		{
			"ID": record["ID"],
			"Clues": record["Clues"],
			"SolutionGrid": record["SolutionGrid"],
		}
		for record in dataset
	]


def _parse_solution(solution_str: str) -> dict[str, list[str]]:
	"""Parse the Python-literal dict stored in SolutionGrid safely."""
	parsed = ast.literal_eval(solution_str)
	if not isinstance(parsed, dict):
		raise ValueError("SolutionGrid is not a dict")
	return parsed


def _is_easy(record: dict[str, str]) -> bool:
	"""Pt2 3×3 puzzles — fast reference floor."""
	return record["ID"].startswith("Pt2_") and "3x3" in record["ID"]


def _is_hard(record: dict[str, str]) -> bool:
	"""Pt2 5×5 puzzles — main discrimination sweep."""
	return record["ID"].startswith("Pt2_") and "5x5" in record["ID"]


def _filter_by_difficulty(difficulty: str) -> list[dict[str, str]]:
	if difficulty not in {"easy", "hard"}:
		raise ValueError(f"Unknown difficulty: {difficulty!r}")
	records = _load_raw_records()
	predicate = _is_easy if difficulty == "easy" else _is_hard
	filtered = [record for record in records if predicate(record)]
	if not filtered:
		raise ValueError(
			f"No records matched difficulty={difficulty!r}. "
			"Check filter predicates."
		)
	return filtered


def get_puzzle(difficulty: str, task_id: int) -> LogicPuzzle:
	pool = _filter_by_difficulty(difficulty)
	index = task_id % len(pool)
	record = pool[index]

	solution = _parse_solution(record["SolutionGrid"])
	num_positions = len(next(iter(solution.values())))
	num_attributes = len(solution)

	return LogicPuzzle(
		puzzle_id=record["ID"],
		clues=record["Clues"],
		solution=solution,
		num_positions=num_positions,
		num_attributes=num_attributes,
		difficulty=difficulty,
		metadata={
			"pool_size": len(pool),
			"pool_index": index,
		},
	)


def pool_size(difficulty: str) -> int:
	return len(_filter_by_difficulty(difficulty))


def _extract_json_str(raw: str) -> str | None:
	"""Find the first complete JSON object or array in raw text.

	Tries a fenced block first, then scans for the first balanced {…} or […].
	Handles nested structures correctly via bracket counting.
	"""
	fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
	if fence:
		return fence.group(1).strip()

	obj_pos = raw.find("{")
	arr_pos = raw.find("[")

	if obj_pos == -1 and arr_pos == -1:
		return None

	if obj_pos == -1 or (arr_pos != -1 and arr_pos < obj_pos):
		opener, closer, start = "[", "]", arr_pos
	else:
		opener, closer, start = "{", "}", obj_pos

	depth, in_str, esc = 0, False, False
	for i, ch in enumerate(raw[start:], start):
		if esc:
			esc = False
			continue
		if ch == "\\" and in_str:
			esc = True
			continue
		if ch == '"':
			in_str = not in_str
			continue
		if in_str:
			continue
		if ch == opener:
			depth += 1
		elif ch == closer:
			depth -= 1
			if depth == 0:
				return raw[start : i + 1]
	return None


def _normalize_position_records(rows: list[dict]) -> dict[str, list[str]] | None:
	"""Convert [{position: N, attr: val, ...}] → {attr: [val_pos1, ...]}."""
	if not rows or not isinstance(rows[0], dict):
		return None
	try:
		if "position" in rows[0]:
			rows = sorted(rows, key=lambda r: int(r.get("position", 0)))
		keys = [k for k in rows[0] if k != "position"]
		if not keys:
			return None
		result: dict[str, list[str]] = {k: [] for k in keys}
		for row in rows:
			for k in keys:
				result[k].append(str(row.get(k, "")))
		return result
	except Exception:
		return None


def parse_llm_answer_tagged(
	raw: str,
) -> tuple[dict[str, list[str]] | None, str | None]:
	"""Parse raw model output, returning (result, schema_tag).

	schema_tag:
	  "attribute_dict"   — model used the expected {attr: [val, ...]} schema
	  "position_record"  — model used {solution: [{position:N, attr:v}]} or
	                       [{position:N, attr:v}]; normalised to attribute_dict
	  None               — parse failure
	"""
	candidate_str = _extract_json_str(raw)
	if candidate_str is None:
		return None, None

	try:
		parsed = json.loads(candidate_str)
	except json.JSONDecodeError:
		try:
			parsed = ast.literal_eval(candidate_str)
		except Exception:
			return None, None

	# Schema A: dict whose values are lists of non-dict items
	if isinstance(parsed, dict) and parsed:
		has_row_lists = any(
			isinstance(v, list) and v and isinstance(v[0], dict)
			for v in parsed.values()
		)
		if not has_row_lists and all(isinstance(v, list) for v in parsed.values()):
			return parsed, "attribute_dict"

		# Schema B wrapped: {"solution": [{position:N,...},...]}
		if has_row_lists:
			rows = next(
				(
					v
					for v in parsed.values()
					if isinstance(v, list) and v and isinstance(v[0], dict)
				),
				None,
			)
			result = _normalize_position_records(rows)
			if result is not None:
				return result, "position_record"

	# Schema B bare: [{position:N,...},...]
	if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
		result = _normalize_position_records(parsed)
		if result is not None:
			return result, "position_record"

	return None, None


def parse_llm_answer(raw: str) -> dict[str, list[str]] | None:
	"""Backward-compatible wrapper; accepts both JSON schemas."""
	result, _ = parse_llm_answer_tagged(raw)
	return result


def score_answer(
	solution: dict[str, list[str]],
	prediction: dict[str, list[str]] | None,
) -> tuple[bool, float, dict[str, Any]]:
	if prediction is None:
		return False, 0.0, {"error": "parse_failure"}

	total_cells = 0
	correct_cells = 0
	attribute_scores: dict[str, float] = {}

	for attribute, true_vals in solution.items():
		pred_vals = prediction.get(attribute, [])
		if not isinstance(pred_vals, list):
			pred_vals = []
		attr_correct = 0
		for position, true_val in enumerate(true_vals):
			total_cells += 1
			if position < len(pred_vals):
				if _normalize(true_val) == _normalize(pred_vals[position]):
					correct_cells += 1
					attr_correct += 1
		if true_vals:
			attribute_scores[attribute] = attr_correct / len(true_vals)
		else:
			attribute_scores[attribute] = 0.0

	score = correct_cells / total_cells if total_cells > 0 else 0.0
	success = correct_cells == total_cells and total_cells > 0

	return success, score, {
		"correct_cells": correct_cells,
		"total_cells": total_cells,
		"attribute_scores": attribute_scores,
		"missing_attributes": [attr for attr in solution if attr not in prediction],
	}


def _normalize(value: str) -> str:
	if not isinstance(value, str):
		value = str(value)
	normalized = value.lower().strip().replace("-", " ").replace("_", " ")
	return " ".join(normalized.split())


if __name__ == "__main__":
	print(f"Easy pool size: {pool_size('easy')}")
	print(f"Hard pool size: {pool_size('hard')}")

	for difficulty in ("easy", "hard"):
		puzzle = get_puzzle(difficulty, task_id=0)
		print(f"\n--- {difficulty.upper()} example ---")
		print(f"ID: {puzzle.puzzle_id}")
		print(
			f"Grid: {puzzle.num_positions} positions x {puzzle.num_attributes} attributes"
		)
		print(f"Clues (first 300 chars): {puzzle.clues[:300]}")
		print(f"Solution keys: {list(puzzle.solution.keys())}")
		first_pos = {key: vals[0] for key, vals in puzzle.solution.items()}
		print(f"Solution[0]: {first_pos}")

	example = get_puzzle("easy", 0)
	success, score, _ = score_answer(example.solution, example.solution)
	print(f"\nPerfect answer: success={success}, score={score:.2f}")

	rotated = {key: vals[1:] + [vals[0]] for key, vals in example.solution.items()}
	success, score, details = score_answer(example.solution, rotated)
	print(
		f"Rotated answer: success={success}, score={score:.2f}, details={details}"
	)

	fake_output = "```json\n" + json.dumps(example.solution) + "\n```"
	parsed = parse_llm_answer(fake_output)
	print(f"\nFenced JSON parsed correctly: {parsed is not None}")
