# -*- coding: utf-8 -*-
"""
   Alipay.com Inc.
   Copyright (c) 2004-2021 All Rights Reserved.
   ------------------------------------------------------
   File Name : gptcache_serving.py
   Author : fuhui.phe
   Email: hongen.phe@antfin.com
   Create Time : 2023/5/28 11:03
   Description : description what the main function of this file
   Change Activity:
        version0 : 2023/5/28 11:03 by fuhui.phe  init
"""
from datetime import datetime
from typing import Dict
import time
import json
import uuid
import configparser
from concurrent.futures import ThreadPoolExecutor
from modelcache_mm import cache
from modelcache_mm.adapter import adapter
from modelcache_mm.manager import CacheBase, VectorBase, get_data_manager
from modelcache_mm.similarity_evaluation.distance import SearchDistanceEvaluation
from modelcache_mm.processor.pre import mm_insert_dict
from modelcache_mm.processor.pre import mm_query_dict
from modelcache_mm.maya_embedding_service.maya_multi_embedding_service import get_embedding_multi
from modelcache_mm.maya_embedding_service.maya_multi_embedding_service import get_embedding_multi_concurrent_sin


def save_query_info(result, model, query, delta_time_log):
    print('执行 save_query_info!')
    cache.data_manager.save_query_resp(result, model=model, query=query,
                                       delta_time=delta_time_log)


def response_text(cache_resp):
    # print('cache_resp: {}'.format(cache_resp))
    return cache_resp['data']


def response_hitquery(cache_resp):
    # print('cache_resp: {}'.format(cache_resp))
    return cache_resp['hitQuery']


# timm2vec = Timm()
# text2vec = Data2VecAudio()


# python类示例
class UserBackend:
    def __init__(self):
        image_dimension = 768
        text_dimension = 768

        mysql_config = configparser.ConfigParser()
        mysql_config.read('modelcache_mm/config/mysql_config.ini')

        # milvus_config = configparser.ConfigParser()
        # milvus_config.read('modelcache/config/milvus_config.ini')

        redis_config = configparser.ConfigParser()
        redis_config.read('modelcache_mm/config/redis_config.ini')

        data_manager = get_data_manager(CacheBase("mysql", config=mysql_config),
                                        VectorBase("redis", mm_dimension=image_dimension+text_dimension,
                                                   i_dimension=image_dimension, t_dimension=text_dimension,
                                                   redis_config=redis_config))
        cache.init(
            embedding_func=get_embedding_multi,
            embedding_concurrent_func=get_embedding_multi_concurrent_sin,
            data_manager=data_manager,
            similarity_evaluation=SearchDistanceEvaluation(),
            insert_pre_embedding_func=mm_insert_dict,
            query_pre_embedding_func=mm_query_dict,
        )
        self.gptcache_version = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.executor = ThreadPoolExecutor(max_workers=6)

    def __call__(self, param):
        print('version: {}'.format(self.gptcache_version))
        print('call_time: {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        try:
            param_dict = json.loads(param)
        except Exception as e:
            result = {"errorCode": 101, "errorDesc": str(e), "cacheHit": False, "delta_time": 0, "hit_query": '',
                      "answer": ''}
            cache.data_manager.save_query_resp(result, model='', query='', delta_time=0)
            return json.dumps(result)

        request_type = param_dict.get("request_type")
        UUID = param_dict.get("UUID", None)
        print('request_type: {}'.format(request_type))
        # param parsing
        try:
            scope = param_dict.get("scope")
            print('scope: {}'.format(scope))
            if scope is not None:
                model = scope.get('model')
                model = model.replace('-', '_')
                model = model.replace('.', '_')
                print('model: {}'.format(model))

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
            if UUID:
                try:
                    uuid_list = UUID.split('==>')
                    user_start = float(uuid_list[1])
                    ray_http_cost = time.time()-user_start
                    print('ray_http_cost: {}'.format(ray_http_cost))
                except Exception as e:
                    print('uuid_e: {}'.format(e))
            try:
                start_time = time.time()
                response = adapter.ChatCompletion.create_query(
                    scope={"model": model},
                    query=query,
                )
                # print('response: {}'.format(response))
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
                print('delta_time_log: {}'.format(delta_time_log))

                # modify at 20230807 20:51
                future = self.executor.submit(save_query_info, result, model, query, delta_time_log)
                query_time = round(time.time() - start_time, 2)
                print('query_time: {}'.format(query_time))
            except Exception as e:
                # result = {"errorCode": 202, "errorDesc": str(e), "cacheHit": False, "delta_time": 0,
                #           "hit_query": '', "answer": ''}
                raise e
            print('result: {}'.format(result))
            return json.dumps(result, ensure_ascii=False)

        if request_type == 'insert':
            if UUID:
                try:
                    uuid_list = UUID.split('==>')
                    user_start = float(uuid_list[1])
                    ray_http_cost = time.time()-user_start
                    print('ray_http_cost: {}'.format(ray_http_cost))
                except Exception as e:
                    print('uuid_e: {}'.format(e))
            try:
                start_time = time.time()
                try:
                    response = adapter.ChatCompletion.create_insert(
                        model=model,
                        chat_info=chat_info,
                    )
                except Exception as e:
                    # result = {"errorCode": 303, "errorDesc": str(e), "writeStatus": "exception"}
                    # return json.dumps(result, ensure_ascii=False)
                    raise e

                if response == 'success':
                    result = {"errorCode": 0, "errorDesc": "", "writeStatus": "success"}
                else:
                    result = {"errorCode": 301, "errorDesc": response, "writeStatus": "exception"}
                insert_time = round(time.time() - start_time, 2)
                print('insert_time: {}'.format(insert_time))
                return json.dumps(result, ensure_ascii=False)
            except Exception as e:
                # result = {"errorCode": 304, "errorDesc": str(e), "writeStatus": "exception"}
                # print('result: {}'.format(result))
                # return json.dumps(result, ensure_ascii=False)
                raise e

        if request_type == 'remove':
            remove_type = param_dict.get("remove_type")
            id_list = param_dict.get("id_list", [])
            print('remove_type: {}'.format(remove_type))

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

    def __update_config__(self, config: Dict[str, object]):
        """
        可选
        """
        pass

    def __health_check__(self):
        """
        可选
        """
        # logging.info(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        return True


if __name__ == '__main__':
    # ============01
    # request_type = 'insert'
    # scope = {"model": "test_0313"}
    # # UUID = "820b0052-d9d8-11ee-95f1-52775e3e6fd1" + "==>" + str(time.time())
    # UUID = str(uuid.uuid1()) + "==>" + str(time.time())
    # print('UUID: {}'.format(UUID))
    # img_data = "http://resarch.oss-cn-hangzhou-zmf.aliyuncs.com/transFile%2Ftmp%2FLMM_test_image_coco%2FCOCO_train2014_000000332345.jpg"
    # query = {'text': ['父母带着孩子来这个地方可能会有什么顾虑'],
    #          'imageRaw': '',
    #          'imageUrl': img_data,
    #          'imageId': 'ccc'}
    # answer = "应该注意小孩不要跑到铁轨上"
    # chat_info = [{"query": query, "answer": answer}]
    # data_dict = {'request_type': request_type, 'scope': scope, 'chat_info': chat_info, 'UUID': UUID}
    # r1 = json.dumps(data_dict)

    # ============02
    # request_type = 'query'
    # UUID = str(uuid.uuid1()) + "==>" + str(time.time())
    # scope = {"model": "test_0313"}
    # img_data = 'http://resarch.oss-cn-hangzhou-zmf.aliyuncs.com/transFile%2Ftmp%2FLMM_test_image_coco%2FCOCO_train2014_000000332345.jpg'
    # query = {'text': ['父母带着孩子来这个地方可能会有什么顾虑'],
    #          'imageRaw': '',
    #          'imageUrl': img_data,
    #          'multiType': 'IMG_TEXT'}
    # r1 = json.dumps({'request_type': request_type, 'scope': scope, 'query': query, 'UUID': UUID})

    # ============03
    request_type = 'remove'
    scope = {"model": "test_0313"}
    remove_type = 'truncate_by_model'
    r1 = json.dumps({'request_type': request_type, 'scope': scope, 'remove_type': remove_type})

    # ============04
    # request_type = 'register'
    # scope = {"model": "test_0313"}
    # type = 'IMG_TEXT'
    # r1 = json.dumps({'request_type': request_type, 'scope': scope, 'type': type})

    user_backend = UserBackend()
    resp = user_backend(r1)
    print('resp: {}'.format(resp))
