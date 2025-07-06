import multiprocessing
import threading
import uuid
import asyncio
from asyncio import Future, AbstractEventLoop

from modelcache.embedding import EmbeddingModel
from modelcache.embedding.base import BaseEmbedding


def worker_func(embedding_model: EmbeddingModel, model_path, task_queue, result_queue, worker_id):
    """Worker function that runs in separate processes to generate embeddings."""
    base_embedding = BaseEmbedding.get(embedding_model, model_path=model_path)
    print(f"Embedding worker {worker_id} started.")
    try:
        while True:
            job_id, data = task_queue.get()  # Get task from queue
            try:
                result = base_embedding.to_embeddings(data)  # Generate embedding
            except Exception as e:
                result = e
            result_queue.put((job_id, result))  # Send result back
    except KeyboardInterrupt:
        print(f"Embedding worker {worker_id} stopped.")
    except Exception as e:
        print(f"Embedding worker {worker_id} encountered an error: {e}")


class EmbeddingDispatcher:
    """Manages a pool of worker processes for parallel embedding generation."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        model_path: str,
        event_loop: AbstractEventLoop,
        num_workers: int
    ):
        """Initialize the dispatcher with worker processes."""
        if num_workers <= 0:
            raise ValueError("Number of workers must be greater than 0.")

        self.task_queue = multiprocessing.Queue()  # Tasks to workers
        self.result_queue = multiprocessing.Queue()  # Results from workers
        self.futures: dict[str, asyncio.Future] = {}  # Pending futures
        self.event_loop = event_loop
        self._start_result_collector_thread()  # Start result collection thread

        # Start worker processes
        self.workers = []
        for i in range(num_workers):
            p = multiprocessing.Process(
                target=worker_func,
                args=(embedding_model, model_path, self.task_queue, self.result_queue, i)
            )
            p.daemon = True
            p.start()
            self.workers.append(p)

    def _start_result_collector_thread(self):
        """Start a thread to collect results from worker processes."""
        def collect():
            while True:
                job_id, result = self.result_queue.get()  # Get result from queue
                future = self.futures.pop(job_id, None)  # Retrieve future
                if future:
                    self.event_loop.call_soon_threadsafe(
                        future.set_exception if isinstance(result, Exception) else future.set_result,
                        result
                    )

        t = threading.Thread(target=collect, daemon=True)
        t.start()

    def embed(self, data: str) -> Future:
        """Submit a task for embedding generation."""
        job_id = str(uuid.uuid4())  # Generate unique job ID
        future = asyncio.get_running_loop().create_future()  # Create future
        self.futures[job_id] = future  # Store future
        self.task_queue.put((job_id, data))  # Add task to queue
        return future

