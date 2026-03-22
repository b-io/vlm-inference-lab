import time
import random
import statistics
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Request:
    """A single request in the simulation with arrival and service timing."""
    id: int
    arrival_time: float
    compute_units: int
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def latency(self) -> float:
        """Returns the total latency of the request from arrival to completion."""
        if self.end_time is not None and self.arrival_time is not None:
            return self.end_time - self.arrival_time
        return 0.0


class ArrivalSimulator:
    """A simulator that generates request arrivals and calculates latency/throughput."""

    def __init__(self, rate: float, total_requests: int):
        """Initializes the arrival simulator with request rate and total count."""
        self.rate = rate  # requests per second
        self.total_requests = total_requests

    def run(self) -> List[Request]:
        """Runs the arrival simulation and returns a list of requests."""
        requests = []
        current_time = 0.0
        for i in range(self.total_requests):
            # Use Poisson process where inter-arrival time is exponentially distributed
            inter_arrival = random.expovariate(self.rate)
            current_time += inter_arrival
            # Assign random compute cost to each request
            compute_units = random.randint(10, 100)
            requests.append(Request(id=i, arrival_time=current_time, compute_units=compute_units))
        return requests


class SimpleServer:
    """A single-threaded server that processes requests one by one."""

    def __init__(self, capacity_per_second: float):
        """Initializes the server with a processing capacity per second."""
        self.capacity = capacity_per_second

    def process(self, requests: List[Request]) -> List[Request]:
        """Processes a list of requests sequentially and updates their timing."""
        clock = 0.0
        for req in requests:
            # Wait for the request to arrive if the clock is behind
            clock = max(clock, req.arrival_time)
            req.start_time = clock
            # Calculate service time proportional to compute units
            service_time = req.compute_units / self.capacity
            clock += service_time
            req.end_time = clock
        return requests


def analyze_results(requests: List[Request]):
    """Analyzes simulation results and prints performance metrics."""
    latencies = [r.latency for r in requests]
    # Calculate total duration from first arrival to last completion
    total_time = requests[-1].end_time - requests[0].arrival_time
    throughput = len(requests) / total_time

    print(f"--- Simulation Results ---")
    print(f"Total Requests: {len(requests)}")
    print(f"Throughput: {throughput:.2f} req/s")
    print(f"Avg Latency: {statistics.mean(latencies):.4f}s")
    print(f"P50 Latency: {statistics.median(latencies):.4f}s")
    print(f"P99 Latency: {statistics.quantiles(latencies, n=100)[98]:.4f}s")


if __name__ == "__main__":
    sim = ArrivalSimulator(rate=10, total_requests=100)
    server = SimpleServer(capacity_per_second=500)
    results = server.process(sim.run())
    analyze_results(results)
