# Run tests for vlm_inference_lab

# Support UTF-8 emojis in PowerShell console
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$env:PYTHONPATH = "$(Get-Location)\source"

# Run pytest
Write-Host "--------------------------------------------------------"
Write-Host "Running pytest..."
Write-Host "--------------------------------------------------------"
pytest

# Optional: Run simulation demos to verify they work
Write-Host "`n--------------------------------------------------------"
Write-Host "Running Inference Simulator demo..."
Write-Host "--------------------------------------------------------"
python source/vlm_inference_lab/simulation/inference_simulator.py

Write-Host "`n--------------------------------------------------------"
Write-Host "Running VLM Embedding demo..."
Write-Host "--------------------------------------------------------"
python source/vlm_inference_lab/embeddings/demo.py
