import os
from openai import OpenAI

OLLAMA_URL = os.getenv("OPENAI_BASE_URL", "http://ollama:11434/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2")
TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.2"))

client = OpenAI(base_url=OLLAMA_URL, api_key=API_KEY)

def chat_once(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
    )
    return resp.choices[0].message.content.strip()
