import asyncio
import os
import random
import time
import aiohttp

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3-0.6B")

# Ramp plan: (concurrency, duration_seconds)
RAMP = [
    (1, 60),
    (2, 60),
    (4, 90),
    (8, 120),
    (12, 120),
    (16, 180),
    (24, 180),
    (32, 240),
]

SHORT_PROMPTS = [
    "Give a 1-sentence definition of KV cache in LLM inference.",
    "Explain p95 latency in one sentence.",
    "What is TTFT? Answer in one sentence.",
]
LONG_PROMPTS = [
    "Write a detailed explanation of how vLLM schedules requests, with bullet points and an example.",
    "Explain the difference between prefill and decode in transformer inference, with a small analogy.",
    "Explain why tail latency (p99) increases under load, include at least 5 reasons.",
]

def make_payload():
    # 70% short, 30% long
    if random.random() < 0.7:
        prompt = random.choice(SHORT_PROMPTS)
        max_tokens = random.choice([64, 96, 128])
    else:
        prompt = random.choice(LONG_PROMPTS)
        max_tokens = random.choice([256, 384, 512])

    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

async def one_request(session):
    payload = make_payload()
    t0 = time.time()
    try:
        async with session.post(VLLM_URL, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            txt = await resp.text()
            latency = time.time() - t0
            ok = 1 if resp.status == 200 else 0
            return ok, latency, resp.status, len(txt)
    except Exception:
        latency = time.time() - t0
        return 0, latency, 0, 0

async def worker(session, stop_event, stats):
    while not stop_event.is_set():
        ok, lat, status, _ = await one_request(session)
        stats["count"] += 1
        stats["ok"] += ok
        stats["fail"] += (1 - ok)
        stats["lat_sum"] += lat
        # simple latency buckets for a quick feel
        if lat < 1: stats["lt1"] += 1
        elif lat < 3: stats["lt3"] += 1
        elif lat < 10: stats["lt10"] += 1
        else: stats["gte10"] += 1

async def run_stage(concurrency, duration):
    stop_event = asyncio.Event()
    stats = {"count":0,"ok":0,"fail":0,"lat_sum":0.0,"lt1":0,"lt3":0,"lt10":0,"gte10":0}
    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(worker(session, stop_event, stats)) for _ in range(concurrency)]

        t_start = time.time()
        last_print = 0
        try:
            while time.time() - t_start < duration:
                await asyncio.sleep(1)
                if time.time() - last_print >= 5:
                    last_print = time.time()
                    c = max(stats["count"], 1)
                    rps = stats["count"] / (time.time() - t_start)
                    avg = stats["lat_sum"] / c
                    print(
                        f"[concurrency={concurrency:>2}] "
                        f"rps={rps:6.2f} ok={stats['ok']}/{stats['count']} "
                        f"avg={avg:5.2f}s buckets(<1s:{stats['lt1']} <3s:{stats['lt3']} <10s:{stats['lt10']} >=10s:{stats['gte10']})"
                    )
        finally:
            stop_event.set()
            await asyncio.gather(*tasks, return_exceptions=True)

async def main():
    print(f"Target: {VLLM_URL} | Model: {MODEL}")
    print("Running ramp forever. Ctrl+C to stop.\n")
    while True:
        for conc, dur in RAMP:
            print(f"\n=== Stage: concurrency={conc}, duration={dur}s ===")
            await run_stage(conc, dur)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
