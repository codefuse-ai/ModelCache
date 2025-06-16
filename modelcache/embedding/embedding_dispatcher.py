import multiprocessing
import threading
import uuid
import asyncio
import psutil
from asyncio import Future, AbstractEventLoop

from modelcache.embedding import EmbeddingModel
from modelcache.embedding.base import BaseEmbedding


def worker_func(embedding_model: EmbeddingModel, model_path, task_queue, result_queue, worker_id):
    base_embedding = BaseEmbedding.get(embedding_model, model_path=model_path)
    print(f"Embedding worker {worker_id} started.")
    try:
        while True:
            job_id, data = task_queue.get()
            try:
                result = base_embedding.to_embeddings(data)
            except Exception as e:
                result = e
            result_queue.put((job_id, result))
    except KeyboardInterrupt:
        print(f"Embedding worker {worker_id} stopped.")
    except Exception as e:
        print(f"Embedding worker {worker_id} encountered an error: {e}")


class EmbeddingDispatcher:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        model_path: str,
        event_loop: AbstractEventLoop,
        num_workers: int
    ):
        if num_workers <= 0:
            raise ValueError("Number of workers must be greater than 0.")

        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.futures: dict[str, asyncio.Future] = {}
        self.event_loop = event_loop
        self._start_result_collector_thread()

        # Start worker processes
        self.workers = []
        for i in range(num_workers):
            p = multiprocessing.Process(
                target=worker_func,
                args=(embedding_model, model_path, self.task_queue, self.result_queue, i)
            )
            p.daemon = True
            p.start()
            psutil.Process(p.pid).nice(psutil.HIGH_PRIORITY_CLASS)
            self.workers.append(p)

    def _start_result_collector_thread(self):
        def collect():
            while True:
                job_id, result = self.result_queue.get()
                future = self.futures.pop(job_id, None)
                if future:
                    self.event_loop.call_soon_threadsafe(
                        future.set_exception if isinstance(result, Exception) else future.set_result,
                        result
                    )

        t = threading.Thread(target=collect, daemon=True)
        t.start()

    def embed(self, data: str) -> Future:
        job_id = str(uuid.uuid4())
        future = asyncio.get_running_loop().create_future()
        self.futures[job_id] = future
        self.task_queue.put((job_id, data))
        return future

