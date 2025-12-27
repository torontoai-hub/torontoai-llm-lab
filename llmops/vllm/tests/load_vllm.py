#!/usr/bin/env python3
"""
Continuous vLLM load generator (OpenAI-compatible API).
- Runs forever until you Ctrl+C.
- Sends /v1/chat/completions requests to generate monitoring traffic.
"""

import os
import json
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3-0.6B")

# Concurrency and pacing
WORKERS = int(os.getenv("WORKERS", "4"))          # number of concurrent request loops
SLEEP_BETWEEN = float(os.getenv("SLEEP", "0.2"))  # seconds between requests per worker

# Generation params
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

PROMPTS = [
    "Summarize the benefits of containerization in 3 bullet points.",
    "Explain p95 vs p99 latency in simple terms with an example.",
    "Write a short haiku about GPUs and inference.",
    "Given a list of numbers, explain how to compute the median.",
    "Create a checklist for deploying Prometheus + Grafana for an LLM service.",
    "Explain what KV cache is in LLM inference and why it matters.",
    "Generate a quick troubleshooting plan when a Grafana panel shows no data.",
    "Write a 5-step guide to optimize throughput on vLLM.",
    "Explain the difference between throughput and latency for LLM APIs.",
    "Provide 3 ways to reduce tail latency in serving systems.",
]


def post_json(url: str, payload: dict, timeout: int = 60) -> tuple[int, float, str]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, time.time() - t0, body
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        return e.code, time.time() - t0, body
    except URLError as e:
        return 0, time.time() - t0, f"URLError: {e}"
    except Exception as e:
        return 0, time.time() - t0, f"Exception: {e}"


def one_request() -> tuple[bool, float, int]:
    prompt = random.choice(PROMPTS)
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Keep answers concise."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False,
    }

    status, dt, body = post_json(f"{VLLM_BASE_URL}/v1/chat/completions", payload)
    ok = (status == 200)

    # Parse token usage if present
    tokens = 0
    if ok:
        try:
            j = json.loads(body)
            tokens = int(j.get("usage", {}).get("total_tokens", 0) or 0)
        except Exception:
            tokens = 0

    return ok, dt, tokens


def worker_loop(worker_id: int):
    sent = 0
    okc = 0
    failc = 0
    tok_total = 0
    t_report = time.time()

    while True:
        ok, dt, toks = one_request()
        sent += 1
        if ok:
            okc += 1
            tok_total += toks
        else:
            failc += 1

        # print a short line occasionally
        if sent % 10 == 0:
            print(f"[worker {worker_id}] sent={sent} ok={okc} fail={failc} last_latency={dt:.3f}s last_tokens={toks}")

        # periodic summary
        now = time.time()
        if now - t_report >= 30:
            rps = sent / (now - t_report)  # for this period only it’s rough; just a sanity indicator
            print(f"[worker {worker_id}] 30s summary: ok={okc} fail={failc} tokens={tok_total} (avg {tok_total/max(okc,1):.1f} tok/req)")
            okc = failc = tok_total = 0
            sent = 0
            t_report = now

        time.sleep(SLEEP_BETWEEN)


def main():
    print("Starting continuous vLLM load...")
    print(f"Base URL : {VLLM_BASE_URL}")
    print(f"Model    : {MODEL}")
    print(f"Workers  : {WORKERS}")
    print(f"Sleep    : {SLEEP_BETWEEN}s between requests/worker")
    print("Stop with Ctrl+C\n")

    threads = []
    for i in range(WORKERS):
        t = threading.Thread(target=worker_loop, args=(i,), daemon=True)
        t.start()
        threads.append(t)

    # Keep main alive forever
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C). Bye.")


if __name__ == "__main__":
    main()
