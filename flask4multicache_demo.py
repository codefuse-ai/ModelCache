# -*- coding: utf-8 -*-
import time
from flask import Flask, request
import logging
import json
from concurrent.futures import ThreadPoolExecutor
from modelcache_mm import cache
from modelcache_mm.adapter import adapter
from modelcache_mm.manager import CacheBase, VectorBase, get_data_manager
from modelcache_mm.similarity_evaluation.distance import SearchDistanceEvaluation
# from modelcache.processor.pre import query_multi_splicing
# from modelcache.processor.pre import insert_multi_splicing
# from modelcache.utils.model_filter import model_blacklist_filter
# from modelcache.embedding import Data2VecAudio
from modelcache_mm.processor.pre import mm_insert_dict
from modelcache_mm.processor.pre import mm_query_dict
from modelcache_mm.embedding import Clip2Vec

# 创建一个Flask实例
app = Flask(__name__)


def response_text(cache_resp):
    return cache_resp['data']


def save_query_info(result, model, query, delta_time_log):
    cache.data_manager.save_query_resp(result, model=model, query=json.dumps(query, ensure_ascii=False),
                                       delta_time=delta_time_log)


def response_hitquery(cache_resp):
    return cache_resp['hitQuery']


# data2vec = Data2VecAudio()
# data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=data2vec.dimension))

image_dimension = 512
text_dimension = 512
clip2vec = Clip2Vec()
data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss",
                                                                mm_dimension=image_dimension+text_dimension,
                                                                i_dimension=image_dimension,
                                                                t_dimension=text_dimension))


cache.init(
            embedding_func=clip2vec.to_embeddings,
            data_manager=data_manager,
            similarity_evaluation=SearchDistanceEvaluation(),
            insert_pre_embedding_func=mm_insert_dict,
            query_pre_embedding_func=mm_query_dict,
        )

# cache.set_openai_key()
global executor
executor = ThreadPoolExecutor(max_workers=6)


@app.route('/welcome')
def first_flask():  # 视图函数
    return 'hello, llms_cache!'


@app.route('/llms_cache', methods=['GET', 'POST'])
def user_backend():
    try:
        if request.method == 'POST':
            request_data = request.json
        elif request.method == 'GET':
            request_data = request.args
        param_dict = json.loads(request_data)
    except Exception as e:
        result = {"errorCode": 301, "errorDesc": str(e), "cacheHit": False, "delta_time": 0, "hit_query": '',
                  "answer": ''}
        cache.data_manager.save_query_resp(result, model='', query='', delta_time=0)
        return json.dumps(result)

    # param parsing
    try:
        request_type = param_dict.get("request_type")
        scope = param_dict.get("scope")
        if scope is not None:
            model = scope.get('model')
            model = model.replace('-', '_')
            model = model.replace('.', '_')

        if request_type in ['query', 'insert']:
            if request_type == 'query':
                query = param_dict.get("query")
            elif request_type == 'insert':
                chat_info = param_dict.get("chat_info")
                query = chat_info[-1]['query']

        if request_type is None or request_type not in ['query', 'remove', 'insert', 'register']:
            result = {"errorCode": 102,
                      "errorDesc": "type exception, should one of ['query', 'insert', 'remove', 'register']",
                      "cacheHit": False, "delta_time": 0, "hit_query": '', "answer": ''}
            cache.data_manager.save_query_resp(result, model=model, query='', delta_time=0)
            return json.dumps(result)
    except Exception as e:
        result = {"errorCode": 103, "errorDesc": str(e), "cacheHit": False, "delta_time": 0, "hit_query": '',
                  "answer": ''}
        return json.dumps(result)

    if request_type == 'query':
        try:
            start_time = time.time()
            response = adapter.ChatCompletion.create_query(
                scope={"model": model},
                query=query,
            )
            delta_time = '{}s'.format(round(time.time() - start_time, 2))
            if response is None:
                result = {"errorCode": 0, "errorDesc": '', "cacheHit": False, "delta_time": delta_time,
                          "hit_query": '', "answer": ''}
            elif isinstance(response, dict):
                answer = response_text(response)
                hit_query = response_hitquery(response)
                result = {"errorCode": 0, "errorDesc": '', "cacheHit": True, "delta_time": delta_time,
                          "hit_query": hit_query, "answer": answer}
            else:
                result = {"errorCode": 201, "errorDesc": response, "cacheHit": False, "delta_time": delta_time,
                          "hit_query": '', "answer": ''}
            delta_time_log = round(time.time() - start_time, 3)

            future = executor.submit(save_query_info, result, model, query, delta_time_log)
        except Exception as e:
            raise e
        return json.dumps(result, ensure_ascii=False)

    if request_type == 'insert':
        try:
            start_time = time.time()
            try:
                response = adapter.ChatCompletion.create_insert(
                    model=model,
                    chat_info=chat_info,
                )
            except Exception as e:
                raise e

            if response == 'success':
                result = {"errorCode": 0, "errorDesc": "", "writeStatus": "success"}
            else:
                result = {"errorCode": 301, "errorDesc": response, "writeStatus": "exception"}
            insert_time = round(time.time() - start_time, 2)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            raise e

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
        # if response == 'success':
        if state == 'success':
            result = {"errorCode": 0, "errorDesc": "", "response": response, "writeStatus": "success"}
        else:
            result = {"errorCode": 402, "errorDesc": "", "response": response, "writeStatus": "exception"}
        return json.dumps(result)

    if request_type == 'register':
        type = param_dict.get("type")
        response = adapter.ChatCompletion.create_register(
            model=model,
            type=type
        )
        if response in ['create_success', 'already_exists']:
            result = {"errorCode": 0, "errorDesc": "", "response": response, "writeStatus": "success"}
        else:
            result = {"errorCode": 502, "errorDesc": "", "response": response, "writeStatus": "exception"}
        return json.dumps(result)


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000, debug=True)
    app.run(host='0.0.0.0', port=5000)
