import os
import time
import json
import argparse
import asyncio
import statistics
from datetime import datetime, timezone
import zoneinfo
from vlm_inference_lab import get_timezone
from typing import Any, Dict, List, Optional
import yaml
from dataclasses import asdict

from vlm_inference_lab.engines import VllmEngineAdapter, SglangEngineAdapter, ChatMessage, CompletionResult

async def run_request(adapter, messages, semaphore, **kwargs):
    """Sends a single request to the adapter within the given semaphore's concurrency limit."""
    async with semaphore:
        # Use a thread to run the synchronous chat_completion call
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: adapter.chat_completion(messages, **kwargs)
        )
        return result

async def benchmark_concurrency(
    adapter, 
    num_requests: int, 
    concurrency: int, 
    prompt: str,
    max_tokens: int = 128
):
    """Executes a set of requests with a specified concurrency level."""
    # Create a semaphore to control concurrent requests
    semaphore = asyncio.Semaphore(concurrency)
    messages = [ChatMessage(role="user", content=prompt)]
    
    # Record the start time of the benchmark
    start_time = time.perf_counter()
    # Create tasks for all requests
    tasks = [
        run_request(adapter, messages, semaphore, max_tokens=max_tokens) 
        for _ in range(num_requests)
    ]
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)
    # Calculate total elapsed time
    total_time = time.perf_counter() - start_time
    
    return results, total_time

def summarize_results(results, total_time, num_requests, concurrency):
    """Aggregates individual completion results into a summary report."""
    # Filter successful latencies and collect errors
    latencies = [r.latency_ms for r in results if not r.error]
    errors = [r.error for r in results if r.error]
    # Calculate total generated tokens
    total_tokens = sum(r.completion_tokens for r in results if not r.error)
    
    # Build the summary dictionary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_requests": num_requests,
        "concurrency": concurrency,
        "total_time_s": total_time,
        "throughput_rps": num_requests / total_time if total_time > 0 else 0,
        "throughput_tps": total_tokens / total_time if total_time > 0 else 0,
        "error_count": len(errors),
        "latency_avg_ms": statistics.mean(latencies) if latencies else 0,
        "latency_p50_ms": statistics.median(latencies) if latencies else 0,
        "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 20 else (max(latencies) if latencies else 0),
        "latency_p99_ms": sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) >= 100 else (max(latencies) if latencies else 0),
    }
    return summary

def main():
    """Parses command-line arguments and orchestrates the serving benchmark."""
    parser = argparse.ArgumentParser(description="VLM Serving Benchmark")
    parser.add_argument("--backend", choices=["vllm", "sglang"], default="vllm")
    parser.add_argument("--url", default="http://localhost:8000/v1")
    parser.add_argument("--model", help="Model name to use")
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--prompt", default="Explain the difference between KV cache and continuous batching.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--config", help="Path to experiment config YAML")
    parser.add_argument("--fail-on-errors", action="store_true", help="Exit with non-zero code if any request fails")

    args = parser.parse_args()
    
    # Check if an experiment config YAML is provided
    if args.config:
        with open(args.config, 'r') as f:
            # Load and parse the YAML configuration
            config = yaml.safe_load(f)
            # Override command-line arguments with configuration values
            for k, v in config.items():
                if hasattr(args, k):
                    setattr(args, k, v)

    # Instantiate the appropriate engine adapter
    if args.backend == "vllm":
        adapter = VllmEngineAdapter(base_url=args.url, model=args.model)
    else:
        adapter = SglangEngineAdapter(base_url=args.url, model=args.model)

    print(f"Starting benchmark on {args.backend} at {args.url}")
    print(f"Model: {adapter.model_name()}")
    print(f"Requests: {args.num_requests}, Concurrency: {args.concurrency}")

    # Verify the backend is reachable before starting
    if not adapter.healthcheck():
        print(f"Error: Backend {args.backend} is not healthy at {args.url}")
        exit(1)

    # Run the concurrent benchmark using asyncio
    results, total_time = asyncio.run(
        benchmark_concurrency(
            adapter, args.num_requests, args.concurrency, args.prompt, args.max_tokens
        )
    )

    # Summarize individual request results
    summary = summarize_results(results, total_time, args.num_requests, args.concurrency)
    summary["backend"] = args.backend
    summary["model"] = adapter.model_name()
    
    print("\nResults:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Use the configured timezone for local timestamp
    tz = zoneinfo.ZoneInfo(get_timezone())
    timestamp = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_{args.backend}_{timestamp}.json"
    filepath = os.path.join(args.output_dir, filename)
    
    # Combine summary and raw results into a single output object
    full_output = {
        "summary": summary,
        "raw_results": [asdict(r) for r in results]
    }
    
    # Write the results to a JSON file
    with open(filepath, 'w') as f:
        json.dump(full_output, f, indent=2)
    
    print(f"\nSaved detailed results to {filepath}")

    # Exit with error if requested and errors occurred
    if args.fail_on_errors and summary["error_count"] > 0:
        print(f"Error: Benchmark finished with {summary['error_count']} errors.")
        exit(1)

if __name__ == "__main__":
    main()
