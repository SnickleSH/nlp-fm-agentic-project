import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

UNLIMITED_MAX_TOKENS = 65536
REASONING_EFFORT = "medium"


def create_llm(
    temperature: float = 0.7,
    max_tokens: int = UNLIMITED_MAX_TOKENS,
    thinking_token_budget: int | None = None,
    request_timeout: float = 1800.0,
) -> ChatOpenAI:
    extra_body: dict = {"reasoning_effort": REASONING_EFFORT}
    if thinking_token_budget is not None:
        extra_body["thinking_token_budget"] = thinking_token_budget
        effective_max_tokens = thinking_token_budget + 2000
    else:
        effective_max_tokens = max_tokens

    return ChatOpenAI(
        base_url=os.getenv("ELTE_API_BASE"),
        api_key=os.getenv("ELTE_API_KEY"),
        model=os.getenv("ELTE_MODEL_NAME"),
        temperature=temperature,
        max_tokens=effective_max_tokens,
        extra_body=extra_body,
        request_timeout=request_timeout,
    )
