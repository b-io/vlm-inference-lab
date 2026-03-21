import unittest
from vlm_inference_lab.simulation.arrival import ArrivalSimulator, SimpleServer, Request

class TestArrivalSimulation(unittest.TestCase):
    def test_simulator_count(self):
        sim = ArrivalSimulator(rate=10, total_requests=50)
        requests = sim.run()
        self.assertEqual(len(requests), 50)
        for r in requests:
            self.assertIsInstance(r, Request)

    def test_server_latency(self):
        # 10 requests, each 100 units. Capacity 100 units/s.
        # Should take 1s per request.
        requests = [Request(id=i, arrival_time=0.0, compute_units=100) for i in range(5)]
        server = SimpleServer(capacity_per_second=100)
        results = server.process(requests)
        
        self.assertEqual(results[0].end_time, 1.0)
        self.assertEqual(results[-1].end_time, 5.0)
        self.assertEqual(results[-1].latency, 5.0)

if __name__ == "__main__":
    unittest.main()
