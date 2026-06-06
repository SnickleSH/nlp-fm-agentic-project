from __future__ import annotations

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from pydantic import BaseModel


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # completion_tokens includes reasoning tokens on Qwen3
    reasoning_tokens: int = 0
    total_tokens: int = 0
    llm_call_count: int = 0
    # Per-call arrays — index N is call N (0-based). Used for H2/H3 analysis
    # (L2B revision costs, budget saturation per call).
    per_call_completion_tokens: list[int] = []
    per_call_finish_reasons: list[str] = []

    @property
    def max_per_call_completion_tokens(self) -> int:
        return max(self.per_call_completion_tokens, default=0)


class MetricsCallback(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.llm_call_count = 0
        self._per_call_completion: list[int] = []
        self._per_call_finish_reasons: list[str] = []

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.llm_call_count += 1
        call_completion = 0
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            call_completion = usage.get("completion_tokens", 0)
            self.completion_tokens += call_completion

        call_finish_reason = "unknown"
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                # Reasoning token breakdown (o1/o3 style; may populate on future
                # Qwen3 endpoint upgrades).
                details = (getattr(msg, "usage_metadata", None) or {}).get(
                    "output_token_details", {}
                )
                self.reasoning_tokens += details.get("reasoning_tokens", 0)
                # finish_reason per call — "stop" is also what budget exhaustion
                # reports, so detection must use completion_tokens ≈ budget.
                gen_info = getattr(gen, "generation_info", None) or {}
                fr = gen_info.get("finish_reason", "unknown")
                if fr and fr != "unknown":
                    call_finish_reason = fr

        self._per_call_completion.append(call_completion)
        self._per_call_finish_reasons.append(call_finish_reason)

    def get_usage(self) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            reasoning_tokens=self.reasoning_tokens,
            total_tokens=self.prompt_tokens + self.completion_tokens,
            llm_call_count=self.llm_call_count,
            per_call_completion_tokens=list(self._per_call_completion),
            per_call_finish_reasons=list(self._per_call_finish_reasons),
        )
