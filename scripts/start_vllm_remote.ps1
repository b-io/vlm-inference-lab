# Default values

# Support UTF-8 emojis in PowerShell console
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$MODEL = if ($args[0]) { $args[0] } else { "facebook/opt-125m" }
$PORT = 8000

Write-Host "--------------------------------------------------------"
Write-Host "Starting vLLM with model: $MODEL on port: $PORT"
Write-Host "--------------------------------------------------------"

$home_cache = Join-Path $HOME ".cache\huggingface"
if (-not (Test-Path $home_cache)) {
    New-Item -ItemType Directory -Force -Path $home_cache | Out-Null
}

docker run -d --name vllm-server `
    --gpus all `
    -v "$($home_cache):/root/.cache/huggingface" `
    -p "$($PORT):8000" `
    --env HF_TOKEN="$($env:HF_TOKEN)" `
    vllm/vllm-openai:latest `
    --model "$MODEL"
