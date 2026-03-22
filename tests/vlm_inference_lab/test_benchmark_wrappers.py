import unittest
import json
import os
import subprocess
from pathlib import Path

def get_ps_script_path(script_name: str) -> str:
    return str(Path(__file__).parent.parent.parent / "scripts" / script_name)

class TestBenchmarkWrappers(unittest.TestCase):
    """
    Since we cannot easily run PowerShell/Bash scripts that depend on the 'vllm' CLI
    in this environment without 'vllm' being present, we test the logic of command
    construction where possible or use dry-runs if we can.
    
    For now, we will verify that the scripts exist and we'll add a Python helper
    to simulate the argument logic if we were to refactor them into Python.
    """

    def test_scripts_exist(self):
        scripts = [
            "benchmark_vllm_bench.ps1",
            "benchmark_vllm_bench.sh",
            "benchmark_vllm_sweep.ps1",
            "benchmark_vllm_sweep.sh"
        ]
        for script in scripts:
            path = Path(__file__).parent.parent.parent / "scripts" / script
            self.assertTrue(path.exists(), f"Script {script} not found at {path}")

    def test_vllm_bench_arg_logic(self):
        # Simulation of the logic inside benchmark_vllm_bench.ps1/sh
        def get_vllm_bench_args(model_id, base_url, num_prompts, max_concurrency):
            endpoint = "/v1/completions"
            backend = "openai"
            if any(x in model_id.lower() for x in ["chat", "instruct", "llama-3"]):
                endpoint = "/v1/chat/completions"
                backend = "openai-chat"
            
            return [
                "vllm", "bench", "serve",
                "--base-url", base_url,
                "--model", model_id,
                "--num-prompts", str(num_prompts),
                "--max-concurrency", str(max_concurrency),
                "--endpoint", endpoint,
                "--backend", backend
            ]

        # Case 1: Base model
        args = get_vllm_bench_args("facebook/opt-125m", "http://localhost:8000", 10, 1)
        self.assertIn("--endpoint", args)
        self.assertEqual(args[args.index("--endpoint") + 1], "/v1/completions")
        self.assertEqual(args[args.index("--backend") + 1], "openai")

        # Case 2: Chat model
        args = get_vllm_bench_args("meta-llama/Llama-3-8b-chat", "http://localhost:8000", 10, 1)
        self.assertEqual(args[args.index("--endpoint") + 1], "/v1/chat/completions")
        self.assertEqual(args[args.index("--backend") + 1], "openai-chat")

    def test_vllm_sweep_json_logic(self):
        # Simulation of the JSON parameter file logic in benchmark_vllm_sweep.ps1/sh
        serve_params = {
            "gpu_memory_utilization": [0.85, 0.90, 0.95],
            "max_num_seqs": [16, 32, 64],
            "max_num_batched_tokens": [1024, 2048, 4096]
        }
        bench_params = {
            "num_prompts": [100],
            "max_concurrency": [1, 2, 4, 8]
        }
        
        # Verify JSON structure
        serve_json = json.dumps(serve_params)
        bench_json = json.dumps(bench_params)
        
        self.assertIn("gpu_memory_utilization", serve_json)
        self.assertIn("max_num_seqs", serve_json)
        self.assertIn("num_prompts", bench_json)
        
        # Verify lists are correct
        self.assertEqual(len(serve_params["gpu_memory_utilization"]), 3)
        self.assertEqual(len(bench_params["max_concurrency"]), 4)

if __name__ == "__main__":
    unittest.main()
