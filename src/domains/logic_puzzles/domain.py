from __future__ import annotations

from src.domains.base import BaseDomain, EvaluationResult, Task
from src.domains.logic_puzzles.engine import get_puzzle, parse_llm_answer, score_answer


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
		keys_text = ", ".join(f"\"{key}\"" for key in attribute_keys)
		rules_text = "\n".join(f"- {rule}" for rule in task.rules)
		return (
			f"{task.description}\n\n"
			f"Rules:\n{rules_text}\n\n"
			f"Attribute keys (case-sensitive): {keys_text}\n"
			"Use these keys exactly in the JSON output.\n\n"
			f"Clues:\n{clues}\n\n"
			"Output format:\n"
			"{\"attribute\": [\"value_at_position_1\", \"value_at_position_2\", ...]}"
		)

	def evaluate(self, task: Task, answer: str) -> EvaluationResult:
		prediction = parse_llm_answer(answer)
		success, score, details = score_answer(task.ground_truth, prediction)
		details["puzzle_id"] = task.metadata.get("puzzle_id")
		return EvaluationResult(success=success, score=score, details=details)
