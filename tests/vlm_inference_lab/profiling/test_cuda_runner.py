import unittest
import os
from vlm_inference_lab.profiling.cuda_runner import CudaRunner


class TestCudaRunner(unittest.TestCase):
    def setUp(self):
        self.runner = CudaRunner()

    def test_parse_output(self):
        mock_stdout = "Some random text\nRESULTS_JSON: {\"n\": 1024, \"ms\": 0.1}\nOther text"
        metrics = self.runner._parse_output(mock_stdout)
        self.assertEqual(metrics["n"], 1024)
        self.assertEqual(metrics["ms"], 0.1)

    def test_normpath_resource_dir(self):
        # Verify normalization
        self.assertTrue(os.path.isabs(self.runner.resource_dir) or "vlm_inference_lab" in self.runner.resource_dir)


if __name__ == "__main__":
    unittest.main()
