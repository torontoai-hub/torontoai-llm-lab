#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepSeek v32 — Simple Validation App (vLLM/OpenAI-compatible)

Usage:
  python3 deepseek_v32_validation.py
Optional env vars:
  BASE_URL (default: http://127.0.0.1:8000/v1)
  MODEL_NAME (default: deepseek-v32)
"""

from openai import OpenAI
import os
import sys
import time
from typing import List, Tuple

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "deepseek-ai/DeepSeek-V3.2-Exp")
API_KEY = os.environ.get("API_KEY", "dummy")  # vLLM doesn't require a real key

# ---------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------
def get_client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def print_banner(title: str):
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")

def safe_chat_completion(client: OpenAI, **kwargs):
    """
    Small wrapper to catch and print helpful errors without crashing the whole run.
    """
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        print_banner("ERROR DURING COMPLETION")
        print(f"{type(e).__name__}: {e}")
        return None

# ---------------------------------------------------------------------
# 0) Connectivity + Hello World
# ---------------------------------------------------------------------
def sanity_check(client: OpenAI):
    print(f"Connected to vLLM server at {BASE_URL}")
    print(f"Using model: {MODEL_NAME}")

    resp = safe_chat_completion(
        client,
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Hello! Can you tell me what 2+2 equals?"}
        ],
        temperature=0.7,
        max_tokens=300
    )

    if resp is not None:
        print_banner("MODEL RESPONSE")
        try:
            print(resp.choices[0].message.content)
        except Exception:
            print(resp)
    else:
        print("Sanity check failed (no response).")

# ---------------------------------------------------------------------
# 1) Long Context Test (Streaming)
# ---------------------------------------------------------------------
def test_long_context(client: OpenAI, context_length_words: int = 8000):
    """
    Test with long input contexts.
    DeepSeek Sparse Attention should handle this efficiently.
    """

    # Create a long context document
    long_document = " ".join([
        (
            f"This is sentence number {i} in a very long technical document. "
            f"It discusses advanced topics in artificial intelligence, specifically topic {i % 10}. "
            f"The research findings indicate significant improvements in performance metrics. "
        )
        for i in range(max(1, context_length_words // 30))  # Approximate word count
    ])

    # Add a question that requires understanding the full context
    question = "Based on the entire document above, how many different topics are discussed?"
    full_prompt = f"{long_document}\n\n{question}"

    word_count = len(full_prompt.split())
    # Avoid float division TypeError in older Pythons; also keep it human-friendly
    estimated_tokens = int(word_count / 0.75)  # Rough estimate: 1 token ≈ 0.75 words

    print(f"📄 Testing with ~{word_count:,} words (~{estimated_tokens:,} tokens)...")

    # Get streaming response
    chunks: List[str] = []
    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.7,
            max_tokens=150,
            stream=True
        )
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content
                if delta:
                    chunks.append(delta)
            except Exception:
                # ignore malformed chunk pieces
                pass
    except Exception as e:
        print_banner("STREAM ERROR")
        print(f"{type(e).__name__}: {e}")
        return

    response_text = "".join(chunks)

    print_banner("LONG CONTEXT TEST")
    print(f"Context Size: ~{word_count:,} words (~{estimated_tokens:,} tokens)")
    print(f"\n💬 Model Response:\n{response_text}")

# ---------------------------------------------------------------------
# 2) Math Reasoning
# ---------------------------------------------------------------------
def application_math_reasoning(client: OpenAI):
    math_prompt = """
Solve this problem step by step:

A sequence is defined by a₁ = 1 and aₙ₊₁ = 2aₙ + 3 for n ≥ 1.
Find a₁₀.

Show your work clearly and verify your answer.
""".strip()

    resp = safe_chat_completion(
        client,
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are an expert mathematician. Solve problems step-by-step with clear reasoning."
            },
            {"role": "user", "content": math_prompt}
        ],
        temperature=0.3,
        max_tokens=1024
    )
    print_banner("APPLICATION 1: MATHEMATICAL REASONING")
    if resp is not None:
        try:
            print(resp.choices[0].message.content)
        except Exception:
            print(resp)

# ---------------------------------------------------------------------
# 3) Code Generation
# ---------------------------------------------------------------------
def application_code_generation(client: OpenAI):
    code_prompt = "Write a Python function to reverse a string. Include type hints and a docstring."

    resp = safe_chat_completion(
        client,
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an expert Python developer. Write clean, efficient, well-documented code."},
            {"role": "user", "content": code_prompt}
        ],
        temperature=0.4,
        max_tokens=512
    )

    print_banner("APPLICATION 2: CODE GENERATION")
    if resp is not None:
        try:
            print(resp.choices[0].message.content)
        except Exception:
            print(resp)

# ---------------------------------------------------------------------
# 4) Multilingual
# ---------------------------------------------------------------------
def application_multilingual(client: OpenAI):
    multilingual_prompts: List[Tuple[str, str]] = [
        ("English", "Explain how neural networks learn in 3 sentences."),
        ("中文 (Chinese)", "用三句话解释神经网络如何学习。"),
        ("Español (Spanish)", "Explica cómo aprenden las redes neuronales en 3 frases."),
        ("Français (French)", "Expliquez en 3 phrases comment les réseaux neuronaux apprennent.")
    ]

    print_banner("APPLICATION 3: MULTILINGUAL CAPABILITIES")
    for language, prompt in multilingual_prompts:
        resp = safe_chat_completion(
            client,
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=200
        )
        print(f"\n🌐 {language}:")
        print(f"Prompt: {prompt}")
        if resp is not None:
            try:
                print(f"Response: {resp.choices[0].message.content}")
            except Exception:
                print(f"Response object:\n{resp}")
        print("-" * 70)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    client = get_client()

    # 0) Sanity
    sanity_check(client)

    # 1) Long context sweeps
    for ctx_len in [2000, 8000, 16000]:
        test_long_context(client, ctx_len)
        time.sleep(2)

    # 2) Math
    application_math_reasoning(client)

    # 3) Code gen
    application_code_generation(client)

    # 4) Multilingual
    application_multilingual(client)

    print("\nAll tests complete.\n")

if __name__ == "__main__":
    # Quick dependency hint if the import fails in some environments
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
