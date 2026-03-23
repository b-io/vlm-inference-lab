import time
import random
import statistics
import contextlib
from typing import List, Dict


class PerformanceMonitor:
    """A helper class to monitor and record execution times of various operations."""

    def __init__(self):
        """Initializes the performance monitor with an empty metrics dictionary."""
        self.metrics: Dict[str, List[float]] = {}

    @contextlib.contextmanager
    def timer(self, name: str):
        """Yields a context manager that records the duration of the enclosed block."""
        # Record the start time using a high-resolution counter
        start = time.perf_counter()
        yield
        # Record the end time and calculate duration
        end = time.perf_counter()
        duration = end - start
        if name not in self.metrics:
            # Initialize the list for a new metric name
            self.metrics[name] = []
        self.metrics[name].append(duration)

    def report(self):
        """Prints a performance report with statistics for all recorded metrics."""
        print("--- Performance Report ---")
        for name, values in self.metrics.items():
            # Print statistics for each metric
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
