# -*- coding: utf-8 -*-
import atexit
from typing import Optional, List, Any
from modelcache.processor.post import first
from modelcache.similarity_evaluation import ExactMatchEvaluation
from modelcache.similarity_evaluation import SimilarityEvaluation
from modelcache.embedding.string import to_embeddings as string_embedding
from modelcache.report import Report
from modelcache.config import Config
from modelcache.utils.cache_func import cache_all
from modelcache.utils.log import modelcache_log
from modelcache.manager import get_data_manager
from modelcache.manager.data_manager import DataManager


class Cache:
    def __init__(self):
        self.has_init = False
        self.cache_enable_func = None
        self.mm_query_pre_embedding_func = None
        self.mm_insert_pre_embedding_func = None
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
            mm_query_pre_embedding_func=None,
            mm_insert_pre_embedding_func=None,
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
        self.mm_query_pre_embedding_func = mm_query_pre_embedding_func
        self.mm_insert_pre_embedding_func = mm_insert_pre_embedding_func
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
