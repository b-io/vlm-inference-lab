# Run tests for vlm_inference_lab
$env:PYTHONPATH = "source"
python -m unittest discover -s tests/vlm_inference_lab -p "test_*.py"
