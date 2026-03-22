# Run vLLM parameter sweep remotely
# Usage: ./benchmark_vllm_sweep.ps1 <base_url> <model_id> [<tier>] [<extra_args>]

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Normalize-RunpodBaseUrl {
    param([string]$Url)
    $Url = $Url.TrimEnd('/')
    if ($Url.EndsWith('/v1')) {
        $Url = $Url.Substring(0, $Url.Length - 3)
    }
    return $Url
}

$ORIGINAL_URL = $args[0]
if (-not $ORIGINAL_URL) { Write-Error "Required base_url"; exit 1 }

$BASE_URL = Normalize-RunpodBaseUrl $ORIGINAL_URL

$MODEL_ID = $args[1]
if (-not $MODEL_ID) { Write-Error "Required model_id"; exit 1 }

$TIER = if ($args[2]) { $args[2] } else { "sweep-small" }

# Remove default sweep parameters as we generate JSON files
$OUTPUT_DIR = "results/benchmarks/sweeps"
if (-not (Test-Path $OUTPUT_DIR)) {
    New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null
}

$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$OUTPUT_FILE = "$OUTPUT_DIR/vllm_sweep_$TIMESTAMP.json"

Write-Host "--------------------------------------------------------"
Write-Host "Starting vLLM Professional Parameter Sweep"
Write-Host "Configured URL : $ORIGINAL_URL"
Write-Host "Normalized URL : $BASE_URL"
Write-Host "Model          : $MODEL_ID"
Write-Host "Output      : $OUTPUT_FILE"
Write-Host "--------------------------------------------------------"

# Check for vLLM sweep compatibility
$HELP_TEXT = vllm bench sweep serve --help 2>&1
if ($HELP_TEXT -match "--base-url" -and $HELP_TEXT -match "--serve-params" -and $HELP_TEXT -match "--bench-params") {
    Write-Host "vLLM sweep CLI compatibility check passed."
} else {
    Write-Error "Incompatible vLLM version detected for remote sweep."
    Write-Host "The current 'vllm bench sweep serve' command does not appear to support --base-url or JSON parameter files in this version."
    Write-Host "The repo's remote benchmarking path is production-ready for 'bench serve', while remote sweep behavior is version-sensitive."
    Write-Host "Please use a compatible vLLM version (0.4.0+) or stick to 'bench serve' for stable remote benchmarking."
    exit 1
}

# Generate temporary JSON parameter files
$SERVE_PARAMS_FILE = "$OUTPUT_DIR/serve_params_$TIMESTAMP.json"
$BENCH_PARAMS_FILE = "$OUTPUT_DIR/bench_params_$TIMESTAMP.json"

$SERVE_PARAMS = @{
    "gpu_memory_utilization" = @(0.85, 0.90, 0.95);
    "max_num_seqs" = @(16, 32, 64);
    "max_num_batched_tokens" = @(1024, 2048, 4096)
}
$SERVE_PARAMS | ConvertTo-Json | Out-File -FilePath $SERVE_PARAMS_FILE -Encoding utf8

$BENCH_PARAMS = @{
    "num_prompts" = @(100);
    "max_concurrency" = @(1, 2, 4, 8)
}
$BENCH_PARAMS | ConvertTo-Json | Out-File -FilePath $BENCH_PARAMS_FILE -Encoding utf8

# vllm bench sweep serve --base-url <url> --model <model> --serve-params <file> --bench-params <file>
$PARAMS = @(
    "bench", "sweep", "serve",
    "--base-url", $BASE_URL,
    "--model", $MODEL_ID,
    "--serve-params", $SERVE_PARAMS_FILE,
    "--bench-params", $BENCH_PARAMS_FILE,
    "--output-dir", $OUTPUT_DIR
)

Write-Host "Executing: vllm $($PARAMS -join ' ')"
vllm @PARAMS

if ($LASTEXITCODE -eq 0) {
    Write-Host "--------------------------------------------------------"
    Write-Host "Sweep Complete. Generating Pareto Summary..."
    $env:PYTHONPATH = "source"
    python -m vlm_inference_lab.experiments.pareto_analysis --input-dir $OUTPUT_DIR --output "$OUTPUT_DIR/pareto_summary_$TIMESTAMP.md"
} else {
    Write-Error "vLLM sweep failed."
    exit $LASTEXITCODE
}
