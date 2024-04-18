# -*- coding: utf-8 -*-
import atexit
from typing import Optional, List, Any
from modelcache_mm.processor.post import first
from modelcache_mm.similarity_evaluation import ExactMatchEvaluation
from modelcache_mm.similarity_evaluation import SimilarityEvaluation
from modelcache_mm.embedding.string import to_embeddings as string_embedding
from modelcache_mm.report import Report
from modelcache_mm.config import Config
from modelcache_mm.utils.cache_func import cache_all
from modelcache_mm.utils.log import modelcache_log
from modelcache_mm.manager import get_data_manager
from modelcache_mm.manager.data_manager import DataManager


class Cache:
    def __init__(self):
        self.has_init = False
        self.cache_enable_func = None
        self.query_pre_embedding_func = None
        self.insert_pre_embedding_func = None
        self.embedding_func = None
        self.embedding_concurrent_func = None
        self.data_manager: Optional[DataManager] = None
        self.similarity_evaluation: Optional[SimilarityEvaluation] = None
        self.post_process_messages_func = None
        self.config = Config()
        self.report = Report()
        self.next_cache = None

    def init(
            self,
            cache_enable_func=cache_all,
            query_pre_embedding_func=None,
            insert_pre_embedding_func=None,
            embedding_func=string_embedding,
            embedding_concurrent_func=string_embedding,
            data_manager: DataManager = get_data_manager(),
            similarity_evaluation=ExactMatchEvaluation(),
            post_process_messages_func=first,
            config=Config(),
            next_cache=None,
    ):
        self.has_init = True
        self.cache_enable_func = cache_enable_func
        self.query_pre_embedding_func = query_pre_embedding_func
        self.insert_pre_embedding_func = insert_pre_embedding_func
        self.embedding_func = embedding_func
        self.embedding_concurrent_func = embedding_concurrent_func
        self.data_manager: DataManager = data_manager
        self.similarity_evaluation = similarity_evaluation
        self.post_process_messages_func = post_process_messages_func
        self.config = config
        self.next_cache = next_cache

        @atexit.register
        def close():
            try:
                self.data_manager.close()
            except Exception as e:
                modelcache_log.error(e)

    def import_data(self, questions: List[Any], answers: List[Any]) -> None:
        self.data_manager.import_data(
            questions=questions,
            answers=answers,
            embedding_datas=[self.embedding_func(question) for question in questions],
        )

    def flush(self):
        self.data_manager.flush()
        if self.next_cache:
            self.next_cache.data_manager.flush()


cache = Cache()
