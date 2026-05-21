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
	puzzle_id = record["ID"]
	if not puzzle_id.startswith("Pt2_") or "level1" not in puzzle_id:
		return False
	try:
		solution = _parse_solution(record["SolutionGrid"])
		num_positions = len(next(iter(solution.values())))
		return num_positions <= 4
	except Exception:
		return False


def _is_hard(record: dict[str, str]) -> bool:
	puzzle_id = record["ID"]
	return puzzle_id.startswith("Pt1_") and "lexical_replacements" in puzzle_id


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


def parse_llm_answer(raw: str) -> dict[str, list[str]] | None:
	"""Extract a JSON-like dict from raw model output."""
	fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
	if fence_match:
		candidate = fence_match.group(1)
	else:
		brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
		if not brace_match:
			return None
		candidate = brace_match.group(0)

	try:
		parsed = json.loads(candidate)
	except json.JSONDecodeError:
		try:
			parsed = ast.literal_eval(candidate)
		except Exception:
			return None

	if not isinstance(parsed, dict):
		return None
	if not all(isinstance(value, list) for value in parsed.values()):
		return None

	return parsed


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
