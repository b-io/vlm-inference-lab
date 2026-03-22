# Remote Benchmark Helper Script
# Usage: ./benchmark_remote.ps1 <base_url> <model_id> [<num_requests>] [<concurrency>] [<extra_args>]
#        ./benchmark_remote.ps1 <base_url> <model_id> --tier <smoke|latency|throughput> [<extra_args>]

# Support UTF-8 emojis in PowerShell console
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$BASE_URL = $args[0]
if (-not $BASE_URL) { Write-Error "Required base_url"; exit 1 }

$MODEL_ID = $args[1]
if (-not $MODEL_ID) { Write-Error "Required model_id"; exit 1 }

$NUM_REQUESTS = 10
$CONCURRENCY = 2
$TIER = $null
$REMAINING_ARGS = @()

# Simple argument parsing
for ($i = 2; $i -lt $args.Count; $i++) {
    if ($args[$i] -eq "--tier") {
        $TIER = $args[++$i]
    } elseif ($args[$i] -match "^\d+$") {
        # Positional numeric args: first is requests, second is concurrency
        if ($i -eq 2) { $NUM_REQUESTS = $args[$i] }
        elseif ($i -eq 3) { $CONCURRENCY = $args[$i] }
        else { $REMAINING_ARGS += $args[$i] }
    } else {
        $REMAINING_ARGS += $args[$i]
    }
}

$OUTPUT_DIR = "results/remote"
if (-not (Test-Path $OUTPUT_DIR)) {
    New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null
}

Write-Host "--------------------------------------------------------"
Write-Host "Starting Remote Benchmark"
Write-Host "URL         : $BASE_URL"
Write-Host "Model       : $MODEL_ID"
if ($TIER) { Write-Host "Tier        : $TIER" }
Write-Host "Requests    : $NUM_REQUESTS (default if not tier/arg)"
Write-Host "Concurrency : $CONCURRENCY (default if not tier/arg)"
Write-Host "--------------------------------------------------------"

# Ensure dependencies are installed
$env:PYTHONPATH = "source;$env:PYTHONPATH"

$PARAMS = @("--url", "$BASE_URL", "--model", "$MODEL_ID", "--num-requests", "$NUM_REQUESTS", "--concurrency", "$CONCURRENCY", "--output-dir", "$OUTPUT_DIR")
if ($TIER) { $PARAMS += @("--tier", $TIER) }
$PARAMS += $REMAINING_ARGS

# Run the benchmark
python -m vlm_inference_lab.experiments.benchmark_serving @PARAMS

if ($LASTEXITCODE -ne 0) {
    Write-Host "--------------------------------------------------------"
    Write-Error "Benchmark failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "--------------------------------------------------------"
Write-Host "Benchmark Complete"
Write-Host "Results saved in: $OUTPUT_DIR"
Write-Host "--------------------------------------------------------"
