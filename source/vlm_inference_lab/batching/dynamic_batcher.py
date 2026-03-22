import time
from typing import List, Any

class DynamicBatcher:
    """A batcher that combines individual requests into batches based on max size or timeout."""
    
    def __init__(self, max_batch_size: int, timeout_ms: float):
        """Initializes the batcher with max batch size and timeout in milliseconds."""
        self.max_batch_size: int = max_batch_size
        self.timeout_sec: float = timeout_ms / 1000.0
        self.queue: List[Any] = []
        self.last_flush: float = time.time()

    def add_request(self, request: Any):
        """Adds a request to the queue."""
        self.queue.append(request)

    def should_flush(self) -> bool:
        """Checks if the batch should be flushed based on size or timeout."""
        if not self.queue:
            return False
        
        batch_full = len(self.queue) >= self.max_batch_size
        timeout_reached = (time.time() - self.last_flush) >= self.timeout_sec
        
        return batch_full or timeout_reached

    def flush(self) -> List[Any]:
        """Flushes the current queue and returns the batch."""
        # Create a copy of the queue for the batch
        batch = self.queue[:]
        # Reset the queue
        self.queue = []
        # Update the last flush timestamp
        self.last_flush = time.time()
        return batch

def demo():
    """Runs a simple demo of the DynamicBatcher."""
    batcher = DynamicBatcher(max_batch_size=4, timeout_ms=100)
    
    # Simulate arrivals
    for i in range(10):
        # Add a request to the batcher
        batcher.add_request(f"req_{i}")
        if batcher.should_flush():
            # Flush the batch if criteria are met
            batch = batcher.flush()
            print(f"Flushed batch: {batch}")
        # Sleep to simulate time passing
        time.sleep(0.03) # 30ms

    # Final flush
    if batcher.queue:
        # Perform a final flush if any requests remain
        print(f"Final flush: {batcher.flush()}")

if __name__ == "__main__":
    demo()
