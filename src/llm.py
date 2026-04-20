import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def create_llm(temperature: float = 0.7, max_tokens: int = 4096) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=os.getenv("ELTE_API_BASE"),
        api_key=os.getenv("ELTE_API_KEY"),
        model=os.getenv("ELTE_MODEL_NAME"),
        temperature=temperature,
        max_tokens=max_tokens,
    )
