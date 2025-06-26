# -*- coding: utf-8 -*-
import asyncio
import atexit
import json
import logging
import time
from asyncio import AbstractEventLoop
from typing import Callable, Optional, List, Any, Coroutine
from modelcache.adapter import adapter
from modelcache.embedding.embedding_dispatcher import EmbeddingDispatcher
from modelcache.utils.model_filter import model_blacklist_filter
from concurrent.futures import ThreadPoolExecutor, Future
import configparser
from modelcache.embedding.base import BaseEmbedding, EmbeddingModel, MetricType
from modelcache.manager.scalar_data.sql_storage import SQLStorage
from modelcache.manager.vector_data.base import VectorStorage
from modelcache.processor.post import first
from modelcache.processor.pre import query_with_role, query_multi_splicing, insert_multi_splicing
from modelcache.similarity_evaluation.base import SimilarityEvaluation
from modelcache.report import Report
from modelcache.similarity_evaluation.distance import SearchDistanceEvaluation
from modelcache.utils.error import CacheError
from modelcache.utils.log import modelcache_log
from modelcache.manager.data_manager import DataManager

            #=====================================================================#
            #==================== Cache class definition =========================#
            #=====================================================================#


# noinspection PyMethodMayBeStatic
class Cache:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        similarity_metric_type: MetricType,
        data_manager: DataManager,
        query_pre_embedding_func: Callable,
        insert_pre_embedding_func: Callable,
        embedding_func: Callable[[str], Future],
        report: Report, # TODO: figure out why this is needed
        similarity_evaluation: Optional[SimilarityEvaluation],
        post_process_messages_func: Callable,

        similarity_threshold: float = 0.95,
        similarity_threshold_long: float = 0.95,
        prompts: Optional[List[str]] = None,
        log_time_func: Callable[[str, float], None] = None,
    ):
        if similarity_threshold < 0 or similarity_threshold > 1:
            raise CacheError(
                "Invalid the similarity threshold param, reasonable range: 0-1"
            )
        self.data_manager: DataManager = data_manager
        self.embedding_model: EmbeddingModel = embedding_model
        self.similarity_metric_type: MetricType = similarity_metric_type
        self.report: Report = report
        self.query_pre_embedding_func: Callable = query_pre_embedding_func
        self.insert_pre_embedding_func: Callable = insert_pre_embedding_func
        self.embedding_func: Callable = embedding_func
        self.similarity_evaluation: Optional[SimilarityEvaluation] = similarity_evaluation
        self.post_process_messages_func: Callable = post_process_messages_func
        self.similarity_threshold = similarity_threshold
        self.similarity_threshold_long = similarity_threshold_long
        self.prompts = prompts
        self.log_time_func: Callable[[str, float], None] = log_time_func

        @atexit.register
        def close():
            try:
                self.data_manager.close()
            except Exception as e:
                modelcache_log.error(e)

    def save_query_resp(self, query_resp_dict, **kwargs):
        """
        Save query response asynchronously to avoid blocking main thread.
        Used for logging and analytics purposes.
        """
        # Execute save operation in a separate thread to maintain async performance
        asyncio.create_task(asyncio.to_thread(
            self.data_manager.save_query_resp,
            query_resp_dict, **kwargs
        ))

    def save_query_info(self, result, model, query, delta_time_log):
        """
        Save query information with execution timing for performance analysis.
        Serializes query data to JSON for storage.
        """
        # Convert query to JSON and save asynchronously
        asyncio.create_task(asyncio.to_thread(
            self.data_manager.save_query_resp,
            result, model=model, query=json.dumps(query, ensure_ascii=False), delta_time=delta_time_log
        ))

    async def handle_request(self, param_dict: dict):
        """
        Main entry point for processing cache requests.

        Routes requests to appropriate handlers based on request type.
        Supports: query, insert, remove, register operations.

        Args:
            param_dict: Request parameters containing type, scope, query, etc.

        Returns:
            dict: Response dictionary with errorCode, result data, and metadata
        """
        # Parse and validate request parameters
        try:
            request_type = param_dict.get("type")

            # Extract model information from scope
            scope = param_dict.get("scope")
            model = None
            if scope is not None:
                model = scope.get('model')
                # Normalize model name for consistent storage (replace special chars)
                model = model.replace('-', '_')
                model = model.replace('.', '_')
            query = param_dict.get("query")
            chat_info = param_dict.get("chat_info")

            # Validate request type against supported operations
            if request_type is None or request_type not in ['query', 'insert', 'remove', 'register']:
                result = {"errorCode": 102,
                          "errorDesc": "type exception, should one of ['query', 'insert', 'remove', 'register']",
                          "cacheHit": False, "delta_time": 0, "hit_query": '', "answer": ''}
                self.save_query_resp(result, model=model, query='', delta_time=0)
                return result
        except Exception as e:
            # Return error response for parameter parsing failures
            return {"errorCode": 103, "errorDesc": str(e), "cacheHit": False, "delta_time": 0, "hit_query": '',
                      "answer": ''}

        # Apply model-based filtering (blacklist check)
        filter_resp = model_blacklist_filter(model, request_type)
        if isinstance(filter_resp, dict):
            return filter_resp

        # Route to appropriate handler based on request type
        if request_type == 'query':
            return await self.handle_query(model, query)
        elif request_type == 'insert':
            return await self.handle_insert(chat_info, model)
        elif request_type == 'remove':
            return await self.handle_remove(model, param_dict)
        elif request_type == 'register':
            return await self.handle_register(model)
        else:
            return {"errorCode": 400, "errorDesc": "bad request"}

    async def handle_register(self, model):
        response = await adapter.ChatCompletion.create_register(
            model=model,
            cache_obj=self
        )
        # Process registration response and return standardized result
        if response in ['create_success', 'already_exists']:
            result = {"errorCode": 0, "errorDesc": "", "response": response, "writeStatus": "success"}
        else:
            result = {"errorCode": 502, "errorDesc": "", "response": response, "writeStatus": "exception"}
        return result

    async def handle_remove(self, model, param_dict):
        remove_type = param_dict.get("remove_type")
        id_list = param_dict.get("id_list", [])

        # Execute removal operation through adapter
        response = await adapter.ChatCompletion.create_remove(
            model=model,
            remove_type=remove_type,
            id_list=id_list,
            cache_obj=self
        )

        # Process removal response and standardize result format
        if not isinstance(response, dict):
            return {"errorCode": 401, "errorDesc": "", "response": response, "removeStatus": "exception"}
        state = response.get('status')
        if state == 'success':
            result = {"errorCode": 0, "errorDesc": "", "response": response, "writeStatus": "success"}
        else:
            result = {"errorCode": 402, "errorDesc": "", "response": response, "writeStatus": "exception"}
        return result

    async def handle_insert(self, chat_info, model):
        try:
            try:
                # Execute insertion through adapter with error handling
                response = await adapter.ChatCompletion.create_insert(
                    model=model,
                    chat_info=chat_info,
                    cache_obj=self
                )
            except Exception as e:
                return {"errorCode": 302, "errorDesc": str(e), "writeStatus": "exception"}

            # Process insertion response
            if response == 'success':
                result = {"errorCode": 0, "errorDesc": "", "writeStatus": "success"}
            else:
                result = {"errorCode": 301, "errorDesc": response, "writeStatus": "exception"}
            return result
        except Exception as e:
            return {"errorCode": 303, "errorDesc": str(e), "writeStatus": "exception"}

    async def handle_query(self, model, query):
        try:
            start_time = time.time()  # Start performance timer

            # Execute query through adapter system
            response = await adapter.ChatCompletion.create_query(
                scope={"model": model},
                query=query,
                cache_obj=self
            )

            # Calculate query execution time
            delta_time = '{}s'.format(round(time.time() - start_time, 2))

            # Process different response types
            if response is None:
                # No cache hit found
                result = {"errorCode": 0, "errorDesc": '', "cacheHit": False, "delta_time": delta_time, "hit_query": '',
                          "answer": ''}
            elif isinstance(response, str):
                # Error occurred during query processing
                result = {"errorCode": 201, "errorDesc": response, "cacheHit": False, "delta_time": delta_time,
                          "hit_query": '', "answer": ''}
            else:
                # Cache hit found - extract response data
                answer = response['data']
                hit_query = response['hitQuery']
                result = {"errorCode": 0, "errorDesc": '', "cacheHit": True, "delta_time": delta_time,
                          "hit_query": hit_query, "answer": answer}

            # Log query performance data asynchronously
            delta_time_log = round(time.time() - start_time, 2)
            self.save_query_info(result, model, query, delta_time_log)
        except Exception as e:
            result = {"errorCode": 202, "errorDesc": str(e), "cacheHit": False, "delta_time": 0,
                      "hit_query": '', "answer": ''}
            logging.info('result: {}'.format(result))
        return result

    def flush(self):
        """Flush all cached data to persistent storage backends."""
        self.data_manager.flush()

    @staticmethod
    async def init(
            sql_storage: str,
            vector_storage: str,
            embedding_model: EmbeddingModel,
            embedding_workers_num: int
    ) -> tuple['Cache' , AbstractEventLoop]:
        """
        Initialize a complete Cache system with all required components.

        Args:
            sql_storage: SQL backend type ("mysql", "sqlite", "elasticsearch")
            vector_storage: Vector backend type ("milvus", "faiss", "chromadb", "redis")
            embedding_model: Embedding model enum value
            embedding_workers_num: Number of parallel embedding worker processes

        Returns:
            tuple: (Cache instance, event loop) ready for async operations

        Raises:
            CacheError: If unsupported storage backends or model configuration issues

        Example:
            cache, loop = await Cache.init(
                sql_storage="mysql",
                vector_storage="milvus",
                embedding_model=EmbeddingModel.HUGGINGFACE_ALL_MPNET_BASE_V2,
                embedding_workers_num=4
            )
        """
        #================= Database configuration loading ===================#

        sql_config = configparser.ConfigParser()
        vector_config = configparser.ConfigParser()

        # Load SQL storage configuration from INI files
        if sql_storage == "mysql":
            sql_config.read('modelcache/config/mysql_config.ini')
        elif sql_storage == "elasticsearch":
            sql_config.read('modelcache/config/elasticsearch_config.ini')
        elif sql_storage == "sqlite":
            sql_config.read('modelcache/config/sqlite_config.ini')
        else:
            modelcache_log.error(f"Unsupported cache storage: {sql_storage}.")
            raise CacheError(f"Unsupported cache storage: {sql_storage}.")

        # Load vector storage configuration from INI files
        if vector_storage == "milvus" :
            vector_config.read('modelcache/config/milvus_config.ini')
        elif vector_storage == "chromadb" :
            vector_config.read('modelcache/config/chromadb_config.ini')
        elif vector_storage == "redis" :
            vector_config.read('modelcache/config/redis_config.ini')
        elif vector_storage == "faiss" :
            vector_config = None # FAISS does not require external configuration
        else:
            modelcache_log.error(f"Unsupported vector storage: {vector_storage}.")
            raise CacheError(f"Unsupported vector storage: {vector_storage}.")


        #=============== embedding-model-specific configuration =====================#

        event_loop = asyncio.get_running_loop()  # Get current async event loop
        model_path = embedding_model.value['model_path']
        dimension = embedding_model.value['dimension']

        # Validate that embedding model has required configuration
        if model_path is None or dimension is None:
            modelcache_log.error(f"Please set the model_path and dimension for {embedding_model} in modelcache/embedding/base.py.")
            raise CacheError(f"Please set the model_path and dimension for {embedding_model} in modelcache/embedding/base.py.")

        # Initialize parallel embedding generation system
        embedding_dispatcher = EmbeddingDispatcher(embedding_model, model_path, event_loop, embedding_workers_num)

        #=== These will be used to initialize the cache ===#
        query_pre_embedding_func: Callable = None
        insert_pre_embedding_func: Callable = None
        post_process_messages_func: Callable = None
        similarity_evaluation: Optional[SimilarityEvaluation] = None
        similarity_metric_type: MetricType = None
        similarity_threshold: float = None
        similarity_threshold_long: float = None
        normalize: bool = None
        #==================================================#

        # Configure cache behavior based on embedding model type
        if (embedding_model == EmbeddingModel.HUGGINGFACE_ALL_MPNET_BASE_V2
        or  embedding_model == EmbeddingModel.HUGGINGFACE_ALL_MINILM_L6_V2
        or  embedding_model == EmbeddingModel.HUGGINGFACE_ALL_MINILM_L12_V2):
            query_pre_embedding_func = query_with_role
            insert_pre_embedding_func = query_with_role
            post_process_messages_func = first
            similarity_evaluation = None # Uses the built-in cosine similarity evaluation in milvus
            similarity_metric_type = MetricType.COSINE
            similarity_threshold = 0.9
            similarity_threshold_long = 0.9
            normalize = False

        elif embedding_model == EmbeddingModel.DATA2VEC_AUDIO:
            query_pre_embedding_func = query_multi_splicing
            insert_pre_embedding_func = insert_multi_splicing
            post_process_messages_func = first
            similarity_evaluation = SearchDistanceEvaluation()
            similarity_metric_type = MetricType.L2
            similarity_threshold = 0.95
            similarity_threshold_long = 0.95
            normalize = True

        # Add configurations for additional embedding models as needed
        else:
            modelcache_log.error(f"Please add configuration for {embedding_model} in modelcache/cache.py.")
            raise CacheError(f"Please add configuration for {embedding_model} in modelcache/cache.py.")

        # ====================== Data manager ==============================#

        # Create coordinated data manager with all storage backends
        data_manager = DataManager.get(
            SQLStorage.get(sql_storage, config=sql_config),
            VectorStorage.get(
                name=vector_storage,
                dimension=dimension,
                config=vector_config,
                metric_type=similarity_metric_type,
            ),
            memory_cache_policy='ARC',
            max_size=10000,
            normalize=normalize,
        )

        #================== Cache Initialization ====================#

        # Create fully configured Cache instance
        cache = Cache(
            embedding_model = embedding_model,
            similarity_metric_type = similarity_metric_type,
            data_manager = data_manager,
            report = Report(),
            embedding_func = embedding_dispatcher.embed,
            query_pre_embedding_func = query_pre_embedding_func,
            insert_pre_embedding_func = insert_pre_embedding_func,
            similarity_evaluation = similarity_evaluation,
            post_process_messages_func = post_process_messages_func,
            similarity_threshold = similarity_threshold,
            similarity_threshold_long = similarity_threshold_long,
            prompts = None,
            log_time_func = None,
        )
        return cache, event_loop