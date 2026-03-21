import time
from typing import List, Any

class DynamicBatcher:
    """Combines individual requests into batches based on max size or timeout."""
    
    def __init__(self, max_batch_size: int, timeout_ms: float):
        self.max_batch_size: int = max_batch_size
        self.timeout_sec: float = timeout_ms / 1000.0
        self.queue: List[Any] = []
        self.last_flush: float = time.time()

    def add_request(self, request: Any):
        self.queue.append(request)

    def should_flush(self) -> bool:
        if not self.queue:
            return False
        
        batch_full = len(self.queue) >= self.max_batch_size
        timeout_reached = (time.time() - self.last_flush) >= self.timeout_sec
        
        return batch_full or timeout_reached

    def flush(self) -> List[Any]:
        batch = self.queue[:]
        self.queue = []
        self.last_flush = time.time()
        return batch

def demo():
    batcher = DynamicBatcher(max_batch_size=4, timeout_ms=100)
    
    # Simulate arrivals
    for i in range(10):
        batcher.add_request(f"req_{i}")
        if batcher.should_flush():
            batch = batcher.flush()
            print(f"Flushed batch: {batch}")
        time.sleep(0.03) # 30ms

    # Final flush
    if batcher.queue:
        print(f"Final flush: {batcher.flush()}")

if __name__ == "__main__":
    demo()
