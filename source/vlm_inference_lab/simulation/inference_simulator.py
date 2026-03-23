import random
import statistics
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from vlm_inference_lab.simulation.arrival import Request, ArrivalSimulator
from vlm_inference_lab.batching.dynamic_batcher import DynamicBatcher
from vlm_inference_lab.profiling.monitor import PerformanceMonitor


@dataclass
class SimulationConfig:
    """A configuration container for inference simulation parameters."""
    arrival_rate: float = 10.0  # requests per second
    total_requests: int = 100
    max_batch_size: int = 8
    batch_timeout_ms: float = 50.0
    gpu_throughput_per_unit: float = 1000.0  # units per second
    gpu_base_latency_ms: float = 10.0  # constant overhead per batch
    seed: Optional[int] = 42
    # Prefill vs Decode simulation
    prefill_cost_per_unit: float = 1.0
    decode_cost_per_token: float = 0.1
    avg_tokens_per_request: int = 50


class InferenceSimulator:
    """A simulator that combines ArrivalSimulator, DynamicBatcher, and PerformanceMonitor to simulate inference
    server behavior."""

    def __init__(self, config: SimulationConfig):
        """Initializes the inference simulator with the given configuration."""
        self.config = config
        self.arrival_sim = ArrivalSimulator(rate=config.arrival_rate, total_requests=config.total_requests)
        self.batcher = DynamicBatcher(max_batch_size=config.max_batch_size, timeout_ms=config.batch_timeout_ms)
        self.monitor = PerformanceMonitor()
        self.processed_requests: List[Request] = []

    def run(self) -> Dict[str, Any]:
        """Runs the inference simulation and returns performance metrics."""
        if self.config.seed is not None:
            # Set the random seed for reproducibility
            random.seed(self.config.seed)

        requests = self.arrival_sim.run()
        if not requests:
            # Return zeroed metrics if no requests are generated
            return {"total_requests":      0, "throughput_rps": 0.0, "latency_avg_ms": 0.0, "latency_p50_ms": 0.0,
                    "latency_p95_ms":      0.0, "latency_p99_ms": 0.0, "queue_time_avg_ms": 0.0,
                    "service_time_avg_ms": 0.0, }

        # Sort requests by arrival time
        requests.sort(key=lambda r: r.arrival_time)

        current_time = 0.0
        pending_requests = requests[:]

        while pending_requests or self.batcher.queue:
            # 1. Add arriving requests to the batcher
            while pending_requests and pending_requests[0].arrival_time <= current_time:
                req = pending_requests.pop(0)
                self.batcher.add_request(req)

            # 2. Check if we should process a batch
            # Note: We override the real-time check in DynamicBatcher.should_flush 
            # for simulation time
            timeout_reached = (current_time - self.batcher.last_flush) >= self.batcher.timeout_sec
            batch_full = len(self.batcher.queue) >= self.batcher.max_batch_size

            if self.batcher.queue and (batch_full or timeout_reached or not pending_requests):
                batch = self.batcher.flush()

                # Simulate GPU processing
                with ((self.monitor.timer("gpu_inference"))):
                    # Total compute for the batch
                    # Simplified prefill + decode model
                    # Prefill is done once for the batch units
                    # Decode is done for each token
                    total_prefill_units = sum(r.compute_units for r in batch)
                    total_decode_tokens = len(batch) * self.config.avg_tokens_per_request

                    prefill_time = (
                        total_prefill_units * self.config.prefill_cost_per_unit
                    ) / self.config.gpu_throughput_per_unit
                    decode_time = (
                        total_decode_tokens * self.config.decode_cost_per_token
                    ) / self.config.gpu_throughput_per_unit

                    compute_time = (self.config.gpu_base_latency_ms / 1000.0) + prefill_time + decode_time

                    # Update request timestamps
                    batch_start_time = current_time
                    batch_end_time = batch_start_time + compute_time

                    for req in batch:
                        req.start_time = batch_start_time
                        req.end_time = batch_end_time
                        self.processed_requests.append(req)

                    # Advance simulation clock by compute time
                    current_time = batch_end_time
            else:
                # Advance simulation clock to next event (either next arrival or timeout)
                next_event_time = current_time + 0.001  # Smallest step 1ms
                if pending_requests:
                    next_event_time = min(next_event_time, pending_requests[0].arrival_time)

                timeout_time = self.batcher.last_flush + self.batcher.timeout_sec
                if self.batcher.queue:
                    next_event_time = min(next_event_time, timeout_time)

                current_time = max(current_time + 0.0001, next_event_time)

        # Summarize results after simulation ends
        return self._summarize()

    def _percentile(self, data: List[float], p: float) -> float:
        """Computes the percentile for the given data using nearest-rank."""
        if not data:
            return 0.0
        n = len(data)
        if n == 1:
            return data[0]

        # Use simple nearest-rank for efficiency in simulation
        idx = max(0, min(n - 1, int(n * p)))
        return data[idx]

    def _summarize(self) -> Dict[str, Any]:
        """Summarizes processed requests into a performance report."""
        if not self.processed_requests:
            return {"total_requests":      0, "throughput_rps": 0.0, "latency_avg_ms": 0.0, "latency_p50_ms": 0.0,
                    "latency_p95_ms":      0.0, "latency_p99_ms": 0.0, "queue_time_avg_ms": 0.0,
                    "service_time_avg_ms": 0.0, "queue_time_p95_ms": 0.0, "service_time_p95_ms": 0.0, }

        # Extract latencies, queue times, and service times in milliseconds
        latencies_ms = sorted([r.latency * 1000 for r in self.processed_requests])
        queue_times_ms = sorted([(r.start_time - r.arrival_time) * 1000 for r in self.processed_requests])
        service_times_ms = sorted([(r.end_time - r.start_time) * 1000 for r in self.processed_requests])

        # Calculate duration from first arrival to last completion
        total_duration = self.processed_requests[-1].end_time - self.processed_requests[0].arrival_time
        throughput = len(self.processed_requests) / total_duration if total_duration > 0 else 0

        # Aggregate final results
        results = {"total_requests":   len(self.processed_requests), "throughput_rps": throughput,
                "latency_avg_ms":      statistics.mean(latencies_ms), "latency_p50_ms": statistics.median(latencies_ms),
                "latency_p95_ms":      self._percentile(latencies_ms, 0.95),
                "latency_p99_ms":      self._percentile(latencies_ms, 0.99),
                "queue_time_avg_ms":   statistics.mean(queue_times_ms),
                "service_time_avg_ms": statistics.mean(service_times_ms),
                "queue_time_p95_ms":   self._percentile(queue_times_ms, 0.95),
                "service_time_p95_ms": self._percentile(service_times_ms, 0.95), }
        return results


if __name__ == "__main__":
    config = SimulationConfig(arrival_rate=20, total_requests=200, max_batch_size=16)
    sim = InferenceSimulator(config)
    stats = sim.run()

    print("--- Inference Simulation Results ---")
    print(f"Configuration: Arrival Rate={config.arrival_rate} req/s, Max Batch={config.max_batch_size}")
    for k, v in stats.items():
        print(f"{k:20}: {v:.4f}")

    sim.monitor.report()
