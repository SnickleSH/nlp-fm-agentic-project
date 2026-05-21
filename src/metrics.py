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


class MetricsCallback(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.llm_call_count = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.llm_call_count += 1
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)

        # Standard OpenAI path for reasoning token breakdown (o1/o3 and future
        # Qwen3 endpoint upgrades).  LangChain surfaces this via
        # generation.message.usage_metadata["output_token_details"]["reasoning_tokens"].
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                details = (getattr(msg, "usage_metadata", None) or {}).get(
                    "output_token_details", {}
                )
                self.reasoning_tokens += details.get("reasoning_tokens", 0)

    def get_usage(self) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            reasoning_tokens=self.reasoning_tokens,
            total_tokens=self.prompt_tokens + self.completion_tokens,
            llm_call_count=self.llm_call_count,
        )
