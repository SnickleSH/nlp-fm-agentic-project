import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("ELTE_API_BASE"),
    api_key=os.getenv("ELTE_API_KEY")
)

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the purpose of life"},
]

print("=== Without reasoning ===")
response_no_reason = client.chat.completions.create(
    model=os.getenv("ELTE_MODEL_NAME"),
    messages=MESSAGES,
    stream=False,
    extra_body={
        "reasoning_effort": "none",
        "include_reasoning": False,
    },
)
msg_no_reasoning = response_no_reason.choices[0].message

if hasattr(msg_no_reasoning, "reasoning") and msg_no_reasoning.reasoning:
    print("[reasoning]")
    print(msg_no_reasoning.reasoning)
    print()
print("[answer]")
print(msg_no_reasoning.content)


print("\n=== With reasoning ===")
response_with_reason = client.chat.completions.create(
    model=os.getenv("ELTE_MODEL_NAME"),
    messages=MESSAGES,
    stream=False,
    extra_body={
        "reasoning_effort": "medium",
        "thinking_token_budget": 2048,
        "include_reasoning": True,
    },
)
msg = response_with_reason.choices[0].message

if hasattr(msg, "reasoning") and msg.reasoning:
    print("[reasoning]")
    print(msg.reasoning)
    print()
print("[answer]")
print(msg.content)

print("\n=== Token usage ===")
usage = response_with_reason.usage
print(f"prompt_tokens    : {usage.prompt_tokens}")
print(f"completion_tokens: {usage.completion_tokens}")
print(f"total_tokens     : {usage.total_tokens}")
details = getattr(usage, "completion_tokens_details", None)
if details:
    print(f"  reasoning_tokens: {getattr(details, 'reasoning_tokens', 'n/a')}")
    print(f"  output_tokens   : {getattr(details, 'accepted_prediction_tokens', 'n/a')}")
