import time
import random
import statistics
import contextlib
from typing import List, Optional, Dict

class PerformanceMonitor:
    """Helper to monitor and record execution times of various operations."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}

    @contextlib.contextmanager
    def timer(self, name: str):
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        duration = end - start
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(duration)

    def report(self):
        print(f"--- Performance Report ---")
        for name, values in self.metrics.items():
            print(f"[{name}]")
            print(f"  Count: {len(values)}")
            print(f"  Avg:   {statistics.mean(values):.6f}s")
            print(f"  Min:   {min(values):.6f}s")
            print(f"  Max:   {max(values):.6f}s")

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    for _ in range(5):
        with monitor.timer("test_operation"):
            time.sleep(random.random() * 0.1)
    monitor.report()
