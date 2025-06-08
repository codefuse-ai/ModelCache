# -*- coding: utf-8 -*-
import time
from flask import Flask, request
import logging
import configparser
import json
from modelcache import cache
from modelcache.adapter import adapter
from modelcache.embedding.mpnet_base import MPNet_Base
from modelcache.manager.vector_data import manager
from modelcache.manager import CacheBase, VectorBase, get_data_manager, data_manager
from modelcache.similarity_evaluation.distance import SearchDistanceEvaluation
from modelcache.processor.pre import query_multi_splicing,insert_multi_splicing, query_with_role
from concurrent.futures import ThreadPoolExecutor
from modelcache.utils.model_filter import model_blacklist_filter
from modelcache.embedding import Data2VecAudio

# 创建一个Flask实例
app = Flask(__name__)


def response_text(cache_resp):
    return cache_resp['data']


def save_query_info(result, model, query, delta_time_log):
    cache.data_manager.save_query_resp(result, model=model, query=json.dumps(query, ensure_ascii=False),
                                       delta_time=delta_time_log)


def response_hitquery(cache_resp):
    return cache_resp['hitQuery']

manager.MPNet_base = True

if manager.MPNet_base:
    mpnet_base = MPNet_Base()
    embedding_func = mpnet_base.to_embeddings
    dimension =  mpnet_base.dimension
    data_manager.NORMALIZE = False
    query_pre_embedding_func=query_with_role
    insert_pre_embedding_func=query_with_role
else:
    data2vec = Data2VecAudio()
    embedding_func = data2vec.to_embeddings
    dimension = data2vec.dimension
    query_pre_embedding_func=query_multi_splicing
    insert_pre_embedding_func=insert_multi_splicing

mysql_config = configparser.ConfigParser()
mysql_config.read('modelcache/config/mysql_config.ini')

milvus_config = configparser.ConfigParser()
milvus_config.read('modelcache/config/milvus_config.ini')

# es_config = configparser.ConfigParser()
# es_config.read('modelcache/config/elasticsearch_config.ini')

# redis_config = configparser.ConfigParser()
# redis_config.read('modelcache/config/redis_config.ini')

# chromadb_config = configparser.ConfigParser()
# chromadb_config.read('modelcache/config/chromadb_config.ini')

data_manager = get_data_manager(
    CacheBase("mysql", config=mysql_config),
    VectorBase("milvus",
               dimension=dimension,
               milvus_config=milvus_config,
               index_params={
                   "metric_type": "COSINE",
                   "index_type": "HNSW",
                   "params": {"M": 16, "efConstruction": 64},
                } if manager.MPNet_base else None,
                search_params={
                    "IVF_FLAT": {"metric_type": "COSINE", "params": {"nprobe": 10}},
                    "IVF_SQ8": {"metric_type": "COSINE", "params": {"nprobe": 10}},
                    "IVF_PQ": {"metric_type": "COSINE", "params": {"nprobe": 10}},
                    "HNSW": {"metric_type": "COSINE", "params": {"ef": 10}},
                    "RHNSW_FLAT": {"metric_type": "COSINE", "params": {"ef": 10}},
                    "RHNSW_SQ": {"metric_type": "COSINE", "params": {"ef": 10}},
                    "RHNSW_PQ": {"metric_type": "COSINE", "params": {"ef": 10}},
                    "IVF_HNSW": {"metric_type": "COSINE", "params": {"nprobe": 10, "ef": 10}},
                    "ANNOY": {"metric_type": "COSINE", "params": {"search_k": 10}},
                    "AUTOINDEX": {"metric_type": "COSINE", "params": {}},
                } if manager.MPNet_base else None
    )
)


# data_manager = get_data_manager(CacheBase("mysql", config=mysql_config),
#                                 VectorBase("chromadb", dimension=data2vec.dimension, chromadb_config=chromadb_config))

# data_manager = get_data_manager(CacheBase("mysql", config=mysql_config),
#                                 VectorBase("redis", dimension=data2vec.dimension, redis_config=redis_config))

cache.init(
    embedding_func=embedding_func,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
    query_pre_embedding_func=query_pre_embedding_func,
    insert_pre_embedding_func=insert_pre_embedding_func,
)

global executor
executor = ThreadPoolExecutor(max_workers=6)


@app.route('/welcome')
def first_flask():  # 视图函数
    return 'hello, modelcache!'


@app.route('/modelcache', methods=['GET', 'POST'])
def user_backend():
    param_dict = []
    try:
        if request.method == 'POST':
            param_dict = request.json
        elif request.method == 'GET':
            param_dict = request.args
    except Exception as e:
        result = {"errorCode": 101, "errorDesc": str(e), "cacheHit": False, "delta_time": 0, "hit_query": '',
                  "answer": ''}
        cache.data_manager.save_query_resp(result, model='', query='', delta_time=0)
        return json.dumps(result)


    # param parsing
    try:
        request_type = param_dict.get("type")

        scope = param_dict.get("scope")
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
            cache.data_manager.save_query_resp(result, model=model, query='', delta_time=0)
            return json.dumps(result)
    except Exception as e:
        result = {"errorCode": 103, "errorDesc": str(e), "cacheHit": False, "delta_time": 0, "hit_query": '',
                  "answer": ''}
        return json.dumps(result)

    # model filter
    filter_resp = model_blacklist_filter(model, request_type)
    if isinstance(filter_resp, dict):
        return json.dumps(filter_resp)

    if request_type == 'query':
        try:
            start_time = time.time()
            response = adapter.ChatCompletion.create_query(
                scope={"model": model},
                query=query
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
            future = executor.submit(save_query_info, result, model, query, delta_time_log)
        except Exception as e:
            result = {"errorCode": 202, "errorDesc": str(e), "cacheHit": False, "delta_time": 0,
                      "hit_query": '', "answer": ''}
            logging.info('result: {}'.format(result))

        return json.dumps(result, ensure_ascii=False)

    if request_type == 'insert':
        try:
            try:
                response = adapter.ChatCompletion.create_insert(
                    model=model,
                    chat_info=chat_info
                )
            except Exception as e:
                result = {"errorCode": 302, "errorDesc": str(e), "writeStatus": "exception"}
                return json.dumps(result, ensure_ascii=False)

            if response == 'success':
                result = {"errorCode": 0, "errorDesc": "", "writeStatus": "success"}
            else:
                result = {"errorCode": 301, "errorDesc": response, "writeStatus": "exception"}
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            result = {"errorCode": 303, "errorDesc": str(e), "writeStatus": "exception"}
            return json.dumps(result, ensure_ascii=False)

    if request_type == 'remove':
        remove_type = param_dict.get("remove_type")
        id_list = param_dict.get("id_list", [])

        response = adapter.ChatCompletion.create_remove(
            model=model,
            remove_type=remove_type,
            id_list=id_list
        )
        if not isinstance(response, dict):
            result = {"errorCode": 401, "errorDesc": "", "response": response, "removeStatus": "exception"}
            return json.dumps(result)

        state = response.get('status')
        if state == 'success':
            result = {"errorCode": 0, "errorDesc": "", "response": response, "writeStatus": "success"}
        else:
            result = {"errorCode": 402, "errorDesc": "", "response": response, "writeStatus": "exception"}
        return json.dumps(result)

    if request_type == 'register':
        response = adapter.ChatCompletion.create_register(
            model=model
        )
        if response in ['create_success', 'already_exists']:
            result = {"errorCode": 0, "errorDesc": "", "response": response, "writeStatus": "success"}
        else:
            result = {"errorCode": 502, "errorDesc": "", "response": response, "writeStatus": "exception"}
        return json.dumps(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
