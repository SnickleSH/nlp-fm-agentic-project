from __future__ import annotations

from src.domains.base import BaseDomain, EvaluationResult, Task
from src.domains.logic_puzzles.engine import (
	get_puzzle,
	parse_llm_answer_tagged,
	score_answer,
)

_FEW_SHOT = (
	"Mini-example (2 positions, 2 attributes):\n"
	"Clue: The cat lives at position 1.\n"
	'Output: {"pet": ["cat", "dog"], "color": ["red", "green"]}\n\n'
)


class LogicPuzzlesDomain(BaseDomain):
	def generate_task(self, difficulty: str, task_id: int) -> Task:
		puzzle = get_puzzle(difficulty, task_id)

		description = (
			"Solve a logic grid puzzle using the clues below. "
			f"There are {puzzle.num_positions} positions and "
			f"{puzzle.num_attributes} attribute categories."
		)

		rules = [
			"Positions are numbered from 1 to N, left to right.",
			"Each attribute category has exactly one value per position.",
			"Use all clues to deduce the full assignment.",
			"Return a JSON object mapping each attribute to a list of values in position order.",
			"Return only JSON, with no extra text or markdown.",
		]

		return Task(
			task_id=task_id,
			description=description,
			rules=rules,
			ground_truth=puzzle.solution,
			difficulty=difficulty,
			metadata={
				"puzzle_id": puzzle.puzzle_id,
				"num_positions": puzzle.num_positions,
				"num_attributes": puzzle.num_attributes,
				"clues": puzzle.clues,
			},
		)

	def format_system_prompt(self, task: Task) -> str:
		return (
			"You are an expert at solving logic grid puzzles. "
			"Use only the given clues to infer the full grid assignment. "
			"Return the final solution in strict JSON format."
		)

	def format_task_prompt(self, task: Task) -> str:
		clues = task.metadata.get("clues")
		if not clues:
			raise RuntimeError("Task metadata missing clues.")

		attribute_keys = list(task.ground_truth.keys())
		keys_text = ", ".join(f'"{key}"' for key in attribute_keys)
		rules_text = "\n".join(f"- {rule}" for rule in task.rules)
		return (
			f"{task.description}\n\n"
			f"Rules:\n{rules_text}\n\n"
			f"Attribute keys (case-sensitive): {keys_text}\n"
			"Use these keys exactly in the JSON output.\n\n"
			f"{_FEW_SHOT}"
			f"Clues:\n{clues}\n\n"
			"Output format:\n"
			'{"attribute": ["value_at_position_1", "value_at_position_2", ...]}'
		)

	def format_critic_prompt(self, task: Task, solution: str) -> str:
		clues = task.metadata.get("clues", "")
		num_positions = task.metadata.get("num_positions", "?")
		num_attributes = task.metadata.get("num_attributes", "?")
		return (
			f"Logic grid puzzle: {num_positions} positions, {num_attributes} attributes.\n\n"
			f"Clues:\n{clues}\n\n"
			f"Proposed solution:\n{solution}\n\n"
			"Verify the proposed solution against EVERY clue above.\n"
			"For each clue, state whether it is SATISFIED or VIOLATED.\n"
			"If violated, explain which values conflict with that clue and what the "
			"correct assignment should be.\n\n"
			"End your response with exactly one of:\n"
			"VERDICT: ACCEPT  — all clues are satisfied\n"
			"VERDICT: REJECT  — one or more clues are violated; list each violation "
			"and the required correction."
		)

	def evaluate(self, task: Task, answer: str) -> EvaluationResult:
		prediction, schema = parse_llm_answer_tagged(answer)
		success, score, details = score_answer(task.ground_truth, prediction)
		details["puzzle_id"] = task.metadata.get("puzzle_id")
		details["answer_schema"] = schema  # "attribute_dict" | "position_record" | None
		return EvaluationResult(success=success, score=score, details=details)
