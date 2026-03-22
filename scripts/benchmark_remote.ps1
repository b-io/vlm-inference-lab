# Remote Benchmark Helper Script
# Usage: ./benchmark_remote.ps1 <base_url> <model_id> [<num_requests>] [<concurrency>] [<extra_args>]

# Support UTF-8 emojis in PowerShell console
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$BASE_URL = $args[0]
if (-not $BASE_URL) { Write-Error "Required base_url"; exit 1 }

$MODEL_ID = $args[1]
if (-not $MODEL_ID) { Write-Error "Required model_id"; exit 1 }

$NUM_REQUESTS = if ($args[2]) { $args[2] } else { 10 }
$CONCURRENCY = if ($args[3]) { $args[3] } else { 2 }
$EXTRA_ARGS = $args[4..($args.Count - 1)]

$OUTPUT_DIR = "results/remote"
if (-not (Test-Path $OUTPUT_DIR)) {
    New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null
}

Write-Host "--------------------------------------------------------"
Write-Host "Starting Remote Benchmark"
Write-Host "URL         : $BASE_URL"
Write-Host "Model       : $MODEL_ID"
Write-Host "Requests    : $NUM_REQUESTS"
Write-Host "Concurrency : $CONCURRENCY"
Write-Host "--------------------------------------------------------"

# Ensure dependencies are installed
$env:PYTHONPATH = "source;$env:PYTHONPATH"

# Run the benchmark
python -m vlm_inference_lab.experiments.benchmark_serving `
    --url "$BASE_URL" `
    --model "$MODEL_ID" `
    --num-requests "$NUM_REQUESTS" `
    --concurrency "$CONCURRENCY" `
    --output-dir "$OUTPUT_DIR" `
    $EXTRA_ARGS

if ($LASTEXITCODE -ne 0) {
    Write-Host "--------------------------------------------------------"
    Write-Error "Benchmark failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "--------------------------------------------------------"
Write-Host "Benchmark Complete"
Write-Host "Results saved in: $OUTPUT_DIR"
Write-Host "--------------------------------------------------------"
