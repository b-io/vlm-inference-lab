import unittest
import time
from vlm_inference_lab.profiling.monitor import PerformanceMonitor


class TestPerformanceMonitor(unittest.TestCase):
    def test_timer_recording(self):
        monitor = PerformanceMonitor()
        with monitor.timer("test"):
            time.sleep(0.01)

        self.assertIn("test", monitor.metrics)
        self.assertEqual(len(monitor.metrics["test"]), 1)
        self.assertGreater(monitor.metrics["test"][0], 0.0)

    def test_multiple_recordings(self):
        monitor = PerformanceMonitor()
        for _ in range(3):
            with monitor.timer("test"):
                pass
        self.assertEqual(len(monitor.metrics["test"]), 3)


if __name__ == "__main__":
    unittest.main()
