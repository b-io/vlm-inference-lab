# Run vLLM's internal benchmark tools remotely
# Usage: ./benchmark_vllm_bench.ps1 <base_url> <model_id> [<tier>] [<extra_args>]

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Normalize-RunpodBaseUrl
{
    param([string]$Url)
    $Url = $Url.TrimEnd('/')
    if ( $Url.EndsWith('/v1'))
    {
        $Url = $Url.Substring(0, $Url.Length - 3)
    }
    return $Url
}

$ORIGINAL_URL = $args[0]
if (-not $ORIGINAL_URL)
{
    Write-Error "Required base_url"; exit 1
}

$BASE_URL = Normalize-RunpodBaseUrl $ORIGINAL_URL

$MODEL_ID = $args[1]
if (-not $MODEL_ID)
{
    Write-Error "Required model_id"; exit 1
}

$TIER = if ($args[2])
{
    $args[2]
}
else
{
    "smoke"
}

$REMAINING_ARGS = @()
if ($args.Count -gt 3)
{
    $REMAINING_ARGS = $args[3..($args.Count - 1)]
}

# Map tiers to vllm bench serve arguments
$NUM_REQUESTS = 10
$CONCURRENCY = 1

if ($TIER -eq "smoke")
{
    $NUM_REQUESTS = 10
    $CONCURRENCY = 1
}
elseif ($TIER -eq "latency")
{
    $NUM_REQUESTS = 100
    $CONCURRENCY = 1
}
elseif ($TIER -eq "throughput")
{
    $NUM_REQUESTS = 200
    $CONCURRENCY = 8
}

$OUTPUT_DIR = "results/benchmarks/$TIER"
if (-not (Test-Path $OUTPUT_DIR))
{
    New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null
}

$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$OUTPUT_FILE = "$OUTPUT_DIR/vllm_bench_$TIMESTAMP.json"

Write-Host "--------------------------------------------------------"
Write-Host "Starting vLLM Professional Benchmark"
Write-Host "Configured URL : $ORIGINAL_URL"
Write-Host "Normalized URL : $BASE_URL"
Write-Host "Model          : $MODEL_ID"
Write-Host "Tier        : $TIER"
Write-Host "Requests    : $NUM_REQUESTS"
Write-Host "Concurrency : $CONCURRENCY"
Write-Host "Output      : $OUTPUT_FILE"
Write-Host "--------------------------------------------------------"

# Note: This assumes vllm is installed on the machine running this script.
# If running against a remote Runpod, we usually run the benchmark from local machine
# targeting the remote URL.

# vllm bench serve --base-url <url> --model <model> --num-prompts <n> --max-concurrency <c>
$ENDPOINT = "/v1/completions"
$BACKEND = "openai"

# Heuristic for chat models
if ($MODEL_ID -like "*chat*" -or $MODEL_ID -like "*instruct*" -or $MODEL_ID -like "*llama-3*")
{
    $ENDPOINT = "/v1/chat/completions"
    $BACKEND = "openai-chat"
}

$PARAMS = @(
    "bench", "serve",
    "--base-url", $BASE_URL,
    "--model", $MODEL_ID,
    "--num-prompts", "$NUM_REQUESTS",
    "--max-concurrency", "$CONCURRENCY",
    "--endpoint", $ENDPOINT,
    "--backend", $BACKEND,
    "--output-json", $OUTPUT_FILE
)
$PARAMS += $REMAINING_ARGS

Write-Host "Executing: vllm $( $PARAMS -join ' ' )"
vllm @PARAMS

if ($LASTEXITCODE -ne 0)
{
    Write-Error "vLLM benchmark failed."
    exit $LASTEXITCODE
}

Write-Host "--------------------------------------------------------"
Write-Host "Benchmark Complete"
Write-Host "--------------------------------------------------------"
