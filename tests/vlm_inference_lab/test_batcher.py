import unittest
import time
from vlm_inference_lab.batching.dynamic_batcher import DynamicBatcher


class TestDynamicBatcher(unittest.TestCase):
    def test_batch_by_size(self):
        # Max size 2, timeout long
        batcher = DynamicBatcher(max_batch_size=2, timeout_ms=1000)
        batcher.add_request("req1")
        self.assertFalse(batcher.should_flush())

        batcher.add_request("req2")
        self.assertTrue(batcher.should_flush())

        batch = batcher.flush()
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch[0], "req1")

    def test_batch_by_timeout(self):
        # Max size 10, timeout 10ms
        batcher = DynamicBatcher(max_batch_size=10, timeout_ms=10)
        batcher.add_request("req1")

        # Immediate check (likely False unless machine is very slow)
        # We wait 20ms to be sure
        time.sleep(0.02)
        self.assertTrue(batcher.should_flush())

        batch = batcher.flush()
        self.assertEqual(len(batch), 1)


if __name__ == "__main__":
    unittest.main()
