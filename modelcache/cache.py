# -*- coding: utf-8 -*-
import atexit
import json
import logging
import time
from typing import Callable, Optional, List
from modelcache.adapter import adapter
from modelcache.utils.model_filter import model_blacklist_filter
from concurrent.futures import ThreadPoolExecutor
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

executor = ThreadPoolExecutor(max_workers=6)

def response_text(cache_resp):
    return cache_resp['data']

def response_hitquery(cache_resp):
    return cache_resp['hitQuery']

# noinspection PyMethodMayBeStatic
class Cache:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        similarity_metric_type: MetricType,
        data_manager: DataManager,
        query_pre_embedding_func: Callable,
        insert_pre_embedding_func: Callable,
        embedding_func: Callable,
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
        self.data_manager.save_query_resp(query_resp_dict, **kwargs)

    def save_query_info(self,result, model, query, delta_time_log):
        self.data_manager.save_query_resp(result, model=model, query=json.dumps(query, ensure_ascii=False),
                                          delta_time=delta_time_log)

    def handle_request(self, param_dict: dict):
        # param parsing
        try:
            request_type = param_dict.get("type")

            scope = param_dict.get("scope")
            model = None
            if scope is not None:
                model = scope.get('model')
                model = model.replace('-', '_')
                model = model.replace('.', '_')
            query = param_dict.get("query")
            chat_info = param_dict.get("chat_info")
            if request_type is None or request_type not in ['query', 'insert', 'remove', 'register']:
                result = {"errorCode": 102,
                          "errorDesc": "type exception, should one of ['query', 'insert', 'remove', 'register']",
                          "cacheHit": False, "delta_time": 0, "hit_query": '', "answer": ''}
                self.data_manager.save_query_resp(result, model=model, query='', delta_time=0)
                return result
        except Exception as e:
            return {"errorCode": 103, "errorDesc": str(e), "cacheHit": False, "delta_time": 0, "hit_query": '',
                      "answer": ''}

        # model filter
        filter_resp = model_blacklist_filter(model, request_type)
        if isinstance(filter_resp, dict):
            return filter_resp

        # handle request
        if request_type == 'query':
            return self.handle_query(model, query)
        elif request_type == 'insert':
            return self.handle_insert(chat_info, model)
        elif request_type == 'remove':
            return self.handle_remove(model, param_dict)
        elif request_type == 'register':
            return self.handle_register(model)
        else:
            return {"errorCode": 400, "errorDesc": "bad request"}

    def handle_register(self, model):
        response = adapter.ChatCompletion.create_register(
            model=model,
            cache_obj=self
        )
        if response in ['create_success', 'already_exists']:
            result = {"errorCode": 0, "errorDesc": "", "response": response, "writeStatus": "success"}
        else:
            result = {"errorCode": 502, "errorDesc": "", "response": response, "writeStatus": "exception"}
        return result

    def handle_remove(self, model, param_dict):
        remove_type = param_dict.get("remove_type")
        id_list = param_dict.get("id_list", [])
        response = adapter.ChatCompletion.create_remove(
            model=model,
            remove_type=remove_type,
            id_list=id_list,
            cache_obj=self
        )
        if not isinstance(response, dict):
            return {"errorCode": 401, "errorDesc": "", "response": response, "removeStatus": "exception"}
        state = response.get('status')
        if state == 'success':
            result = {"errorCode": 0, "errorDesc": "", "response": response, "writeStatus": "success"}
        else:
            result = {"errorCode": 402, "errorDesc": "", "response": response, "writeStatus": "exception"}
        return result

    def handle_insert(self, chat_info, model):
        try:
            try:
                response = adapter.ChatCompletion.create_insert(
                    model=model,
                    chat_info=chat_info,
                    cache_obj=self
                )
            except Exception as e:
                return {"errorCode": 302, "errorDesc": str(e), "writeStatus": "exception"}

            if response == 'success':
                result = {"errorCode": 0, "errorDesc": "", "writeStatus": "success"}
            else:
                result = {"errorCode": 301, "errorDesc": response, "writeStatus": "exception"}
            return result
        except Exception as e:
            return {"errorCode": 303, "errorDesc": str(e), "writeStatus": "exception"}

    def handle_query(self, model, query):
        try:
            start_time = time.time()
            response = adapter.ChatCompletion.create_query(
                scope={"model": model},
                query=query,
                cache_obj=self
            )
            delta_time = '{}s'.format(round(time.time() - start_time, 2))
            if response is None:
                result = {"errorCode": 0, "errorDesc": '', "cacheHit": False, "delta_time": delta_time, "hit_query": '',
                          "answer": ''}
            # elif response in ['adapt_query_exception']:
            elif isinstance(response, str):
                result = {"errorCode": 201, "errorDesc": response, "cacheHit": False, "delta_time": delta_time,
                          "hit_query": '', "answer": ''}
            else:
                answer = response_text(response)
                hit_query = response_hitquery(response)
                result = {"errorCode": 0, "errorDesc": '', "cacheHit": True, "delta_time": delta_time,
                          "hit_query": hit_query, "answer": answer}
            delta_time_log = round(time.time() - start_time, 2)
            executor.submit(self.save_query_info, result, model, query, delta_time_log)
        except Exception as e:
            result = {"errorCode": 202, "errorDesc": str(e), "cacheHit": False, "delta_time": 0,
                      "hit_query": '', "answer": ''}
            logging.info('result: {}'.format(result))
        return result

    def flush(self):
        self.data_manager.flush()

    @staticmethod
    def init(sql_storage: str, vector_storage: str) -> 'Cache':
        #================= configurations for databases ===================#

        sql_config = configparser.ConfigParser()
        vector_config = configparser.ConfigParser()

        if sql_storage == "mysql":
            sql_config.read('modelcache/config/mysql_config.ini')
        elif sql_storage == "elasticsearch":
            sql_config.read('modelcache/config/elasticsearch_config.ini')
        elif sql_storage == "sqlite":
            sql_config.read('modelcache/config/sqlite_config.ini')
        else:
            modelcache_log.error(f"Unsupported cache storage: {sql_storage}.")
            raise CacheError(f"Unsupported cache storage: {sql_storage}.")

        if vector_storage == "milvus" :
            vector_config.read('modelcache/config/milvus_config.ini')
        elif vector_storage == "chromadb" :
            vector_config.read('modelcache/config/chromadb_config.ini')
        elif vector_storage == "redis" :
            vector_config.read('modelcache/config/redis_config.ini')
        elif vector_storage == "faiss" :
            vector_config = None # faiss does not require additional configuration
        else:
            modelcache_log.error(f"Unsupported vector storage: {vector_storage}.")
            raise CacheError(f"Unsupported vector storage: {vector_storage}.")


        #=============== model-specific configuration =====================#

        embedding_model = EmbeddingModel.HUGGINGFACE
        model_path = "sentence-transformers/all-mpnet-base-v2"
        base_embedding = BaseEmbedding.get(embedding_model, model_path=model_path)

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

        # switching based on embedding_model
        if embedding_model == EmbeddingModel.HUGGINGFACE:
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

        # add more configurations for other embedding models as needed
        else:
            modelcache_log.error(f"Please add configuration for {embedding_model} in modelcache/__init__.py.")
            raise CacheError(f"Please add configuration for {embedding_model} in modelcache/__init__.py.")

        # ====================== Data manager ==============================#

        data_manager = DataManager.get(
            SQLStorage.get(sql_storage, config=sql_config),
            VectorStorage.get(
                name=vector_storage,
                dimension=base_embedding.dimension,
                config=vector_config,
                metric_type=similarity_metric_type,
            ),
            eviction='WTINYLFU',
            max_size=100000,
            normalize=normalize,
        )


        #================== Cache Initialization ====================#

        cache = Cache(
            embedding_model = embedding_model,
            similarity_metric_type = similarity_metric_type,
            data_manager = data_manager,
            report = Report(),
            embedding_func = base_embedding.to_embeddings,
            query_pre_embedding_func = query_pre_embedding_func,
            insert_pre_embedding_func = insert_pre_embedding_func,
            similarity_evaluation = similarity_evaluation,
            post_process_messages_func = post_process_messages_func,
            similarity_threshold = similarity_threshold,
            similarity_threshold_long = similarity_threshold_long,
            prompts = None,
            log_time_func = None,
        )
        return cache