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
    request_timeout: float = 900.0,
    max_retries: int = 0,
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
        max_retries=max_retries,
        # Stream so tokens flow continuously. The ELTE gateway enforces a
        # ~1700s idle timeout; in non-streaming mode the whole generation
        # produces no bytes until the end, so any reasoning tail >~1700s gets
        # the connection killed (surfaces as APITimeoutError below the client's
        # own request_timeout). Streaming resets the idle timer per token, so
        # request_timeout becomes an inter-token gap, not a total cap.
        streaming=True,
        # Emit a final usage chunk so usage_metadata is populated under
        # streaming (otherwise token accounting goes to 0) — see metrics.py.
        stream_usage=True,
    )
