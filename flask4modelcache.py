# -*- coding: utf-8 -*-
import time
from datetime import datetime
from flask import Flask, request
import logging
import configparser
import json
from modelcache import cache
from modelcache.adapter import adapter
from modelcache.manager import CacheBase, VectorBase, get_data_manager
from modelcache.similarity_evaluation.distance import SearchDistanceEvaluation
from modelcache.processor.pre import query_multi_splicing
from modelcache.processor.pre import insert_multi_splicing
from concurrent.futures import ThreadPoolExecutor
from modelcache.utils.model_filter import model_blacklist_filter
from modelcache.embedding import Data2VecAudio
# from modelcache.maya_embedding_service.maya_embedding_service import get_cache_embedding_text2vec


# 创建一个Flask实例
app = Flask(__name__)


def response_text(cache_resp):
    return cache_resp['data']


def save_query_info(result, model, query, delta_time_log):
    cache.data_manager.save_query_resp(result, model=model, query=json.dumps(query, ensure_ascii=False),
                                       delta_time=delta_time_log)


def response_hitquery(cache_resp):
    return cache_resp['hitQuery']


data2vec = Data2VecAudio()
mysql_config = configparser.ConfigParser()
mysql_config.read('modelcache/config/mysql_config.ini')
milvus_config = configparser.ConfigParser()
milvus_config.read('modelcache/config/milvus_config.ini')
data_manager = get_data_manager(CacheBase("mysql", config=mysql_config),
                                VectorBase("milvus", dimension=data2vec.dimension, milvus_config=milvus_config))


cache.init(
    embedding_func=data2vec.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
    query_pre_embedding_func=query_multi_splicing,
    insert_pre_embedding_func=insert_multi_splicing,
)

# cache.set_openai_key()
global executor
executor = ThreadPoolExecutor(max_workers=6)


@app.route('/welcome')
def first_flask():  # 视图函数
    return 'hello, modelcache!'


@app.route('/modelcache', methods=['GET', 'POST'])
def user_backend():
    try:
        if request.method == 'POST':
            request_data = request.json
        elif request.method == 'GET':
            request_data = request.args
        param_dict = json.loads(request_data)
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
        if request_type is None or request_type not in ['query', 'insert', 'detox', 'remove']:
            result = {"errorCode": 102,
                      "errorDesc": "type exception, should one of ['query', 'insert', 'detox', 'remove']",
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
            elif response in ['adapt_query_exception']:
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
            result = {"errorCode": 202, "errorDesc": e, "cacheHit": False, "delta_time": 0,
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
                result = {"errorCode": 303, "errorDesc": e, "writeStatus": "exception"}
                return json.dumps(result, ensure_ascii=False)

            if response in ['adapt_insert_exception']:
                result = {"errorCode": 301, "errorDesc": response, "writeStatus": "exception"}
            elif response == 'success':
                result = {"errorCode": 0, "errorDesc": "", "writeStatus": "success"}
            else:
                result = {"errorCode": 302, "errorDesc": response,
                          "writeStatus": "exception"}
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            result = {"errorCode": 304, "errorDesc": e, "writeStatus": "exception"}
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
