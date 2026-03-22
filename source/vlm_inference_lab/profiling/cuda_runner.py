import subprocess
import os
import json
import logging
import argparse
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

@dataclass
class CudaBenchmarkResult:
    """A structured result from a CUDA benchmark run."""
    name: str
    success: bool
    stdout: str
    stderr: str
    compile_command: Optional[str] = None
    run_command: Optional[str] = None
    metrics: Dict[str, Any] = None

class CudaRunner:
    """A utility to compile and run CUDA experiments and capture their performance metrics."""
    def __init__(self, resource_dir: str = "resources/vlm_inference_lab/cuda"):
        """Initializes the CUDA runner with the resource directory path."""
        self.resource_dir = os.path.normpath(resource_dir)
        self.logger = logging.getLogger(__name__)

    def compile(self, file_path: str, output_name: str, extra_args: Optional[List[str]] = None) -> (bool, str):
        """Compiles a CUDA file using nvcc and returns success status and command."""
        source_path = os.path.normpath(file_path)
        output_path = os.path.normpath(output_name)

        if not os.path.exists(source_path):
            # Log error if the source file is missing
            self.logger.error(f"Source file not found: {source_path}")
            return False, ""

        # Construct the nvcc compilation command
        cmd = ["nvcc", source_path, "-o", output_path]
        if extra_args:
            # Append additional arguments to the command
            cmd.extend(extra_args)

        cmd_str = ' '.join(cmd)
        try:
            # Execute the compilation process
            self.logger.info(f"Compiling: {cmd_str}")
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True, cmd_str
        except subprocess.CalledProcessError as e:
            # Log failure if the compiler returns a non-zero exit code
            self.logger.error(f"Compilation failed:\n{e.stderr}")
            return False, cmd_str
        except FileNotFoundError:
            # Handle cases where nvcc is not found in the system path
            self.logger.error("nvcc not found. Ensure CUDA toolkit is installed and in PATH.")
            return False, cmd_str

    def run_benchmark(self, executable_name: str, args: Optional[List[str]] = None, compile_cmd: Optional[str] = None) -> CudaBenchmarkResult:
        """Runs the compiled executable and parses the output for structured data."""
        executable_path = os.path.normpath(f"./{executable_name}")
        if os.name == 'nt' and not (executable_path.endswith('.exe') or '.' in os.path.basename(executable_path)):
            # Check for .exe extension on Windows systems
            if os.path.exists(executable_path + ".exe"):
                executable_path += ".exe"
            elif not os.path.exists(executable_path):
                 # Append .exe anyway if it's Windows
                 executable_path += ".exe"

        # Build the command to run the executable
        cmd = [executable_path] + (args or [])
        cmd_str = ' '.join(cmd)
        
        try:
            # Execute the benchmark and capture output
            self.logger.info(f"Running: {cmd_str}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Parse metrics from the output
            metrics = self._parse_output(result.stdout)
            return CudaBenchmarkResult(
                name=executable_name,
                success=True,
                stdout=result.stdout,
                stderr=result.stderr,
                compile_command=compile_cmd,
                run_command=cmd_str,
                metrics=metrics
            )
        except subprocess.CalledProcessError as e:
            # Handle benchmark execution failures
            self.logger.error(f"Execution failed:\n{e.stderr}")
            return CudaBenchmarkResult(
                name=executable_name,
                success=False,
                stdout=e.stdout,
                stderr=e.stderr,
                compile_command=compile_cmd,
                run_command=cmd_str,
                metrics={}
            )
        except Exception as e:
            # Catch unexpected errors during execution
            self.logger.error(f"An unexpected error occurred: {str(e)}")
            return CudaBenchmarkResult(
                name=executable_name,
                success=False,
                stdout="",
                stderr=str(e),
                compile_command=compile_cmd,
                run_command=cmd_str,
                metrics={}
            )

    def _parse_output(self, stdout: str) -> Dict[str, Any]:
        """Parses the stdout for structured metrics using multiple parsing strategies."""
        results = {}
        in_metrics_block = False
        
        for line in stdout.splitlines():
            # 1. Parse JSON tag (Legacy/Compatibility)
            if "RESULTS_JSON:" in line:
                try:
                    # Extract and parse JSON data from the line
                    json_str = line.split("RESULTS_JSON:")[1].strip()
                    results.update(json.loads(json_str))
                except (IndexError, json.JSONDecodeError) as e:
                    # Log warning for invalid JSON lines
                    self.logger.warning(f"Failed to parse JSON from line: {line}. Error: {e}")
            
            # 2. Parse METRICS block (New/Preferred)
            if "METRICS_START" in line:
                # Set flag when the metrics block begins
                in_metrics_block = True
                continue
            if "METRICS_END" in line:
                # Unset flag when the metrics block ends
                in_metrics_block = False
                continue
            
            if in_metrics_block and "=" in line:
                # Parse key-value pairs separated by '='
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                # Try to convert values to numeric types if possible
                try:
                    if "." in val:
                        results[key] = float(val)
                    else:
                        results[key] = int(val)
                except ValueError:
                    # Handle boolean and string values
                    if val.lower() == "true":
                        results[key] = True
                    elif val.lower() == "false":
                        results[key] = False
                    else:
                        results[key] = val
                        
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile and run CUDA benchmarks.")
    parser.add_argument("--source", type=str, help="Path to the CUDA source file.")
    parser.add_argument("--compile", action="store_true", help="Whether to compile the source.")
    parser.add_argument("--executable", type=str, default="cuda_bench", help="Name of the executable.")
    parser.add_argument("--args", nargs="*", help="Arguments for the executable.")
    parser.add_argument("--json-out", type=str, help="Path to save the results as JSON.")
    parser.add_argument("--keep", action="store_true", help="Keep the executable after running.")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    runner = CudaRunner()
    compile_cmd = None
    
    if args.source and args.compile:
        success, compile_cmd = runner.compile(args.source, args.executable)
        if not success:
            exit(1)
    
    res = runner.run_benchmark(args.executable, args.args, compile_cmd)
    
    if res.success:
        print("\n--- Benchmark Metrics ---")
        print(json.dumps(res.metrics, indent=2))
        
        if args.json_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
            with open(args.json_out, "w") as f:
                json.dump(asdict(res), f, indent=2)
            print(f"Results saved to {args.json_out}")
    else:
        print("\n--- Benchmark Failed ---")
        print(res.stderr)
        exit(1)
        
    if not args.keep:
        for ext in ['', '.exe', '.exp', '.lib', '.pdb', '.obj']:
            f = args.executable + ext
            if os.path.exists(f):
                os.remove(f)
