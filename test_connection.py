import os
from dotenv import load_dotenv
from openai import OpenAI

# Load variables from the .env file
load_dotenv()

# Initialize the client using the environment variables
client = OpenAI(
    base_url=os.getenv("ELTE_API_BASE"),
    api_key=os.getenv("ELTE_API_KEY")
)

# Call the model
chat_completion = client.chat.completions.create(
    model=os.getenv("ELTE_MODEL_NAME"),
    messages=[
        {"role": "system", "content": "You are a helpful assistant." },
        {"role": "user", "content": "What is the purpose of life"}
    ],
    stream=False
)

print(chat_completion.choices[0].message.content)