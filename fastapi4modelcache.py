# -*- coding: utf-8 -*-
import time
import uvicorn
import asyncio
import logging
import configparser
import json
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from starlette.responses import PlainTextResponse  
import functools

from modelcache import cache
from modelcache.adapter import adapter
from modelcache.manager import CacheBase, VectorBase, get_data_manager
from modelcache.similarity_evaluation.distance import SearchDistanceEvaluation
from modelcache.processor.pre import query_multi_splicing
from modelcache.processor.pre import insert_multi_splicing
from modelcache.utils.model_filter import model_blacklist_filter
from modelcache.embedding import Data2VecAudio

#创建一个FastAPI实例
app = FastAPI()

class RequestData(BaseModel):
    type: str
    scope: dict = None
    query: str = None
    chat_info: dict = None
    remove_type: str = None
    id_list: list = []

data2vec = Data2VecAudio()
mysql_config = configparser.ConfigParser()
mysql_config.read('modelcache/config/mysql_config.ini')

milvus_config = configparser.ConfigParser()
milvus_config.read('modelcache/config/milvus_config.ini')

# redis_config = configparser.ConfigParser()
# redis_config.read('modelcache/config/redis_config.ini')

# 初始化datamanager
data_manager = get_data_manager(
    CacheBase("mysql", config=mysql_config),
    VectorBase("milvus", dimension=data2vec.dimension, milvus_config=milvus_config)
)

# # 使用redis初始化datamanager
# data_manager = get_data_manager(
#     CacheBase("mysql", config=mysql_config),
#     VectorBase("redis", dimension=data2vec.dimension, redis_config=redis_config)
# )

cache.init(
    embedding_func=data2vec.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
    query_pre_embedding_func=query_multi_splicing,
    insert_pre_embedding_func=insert_multi_splicing,
)

executor = ThreadPoolExecutor(max_workers=6)

# 异步保存查询信息
async def save_query_info(result, model, query, delta_time_log):
    loop = asyncio.get_running_loop()
    func = functools.partial(cache.data_manager.save_query_resp, result, model=model, query=json.dumps(query, ensure_ascii=False), delta_time=delta_time_log)
    await loop.run_in_executor(None, func)



@app.get("/welcome", response_class=PlainTextResponse)
async def first_fastapi():
    return "hello, modelcache!"

@app.post("/modelcache")
async def user_backend(request: Request):
    try:
        raw_body = await request.body()
        # 解析字符串为JSON对象
        if isinstance(raw_body, bytes):
            raw_body = raw_body.decode("utf-8")
        if isinstance(raw_body, str):
            try:
                # 尝试将字符串解析为JSON对象
                request_data = json.loads(raw_body)
            except json.JSONDecodeError as e:
                # 如果无法解析，返回格式错误
                result = {"errorCode": 101, "errorDesc": str(e), "cacheHit": False, "delta_time": 0, "hit_query": '',
                  "answer": ''}
                asyncio.create_task(save_query_info(result, model='', query='', delta_time_log=0))
                raise HTTPException(status_code=101, detail="Invalid JSON format")
        else:
            request_data = raw_body

        # 确保request_data是字典对象
        if isinstance(request_data, str):
            try:
                request_data = json.loads(request_data)
            except json.JSONDecodeError:
                raise HTTPException(status_code=101, detail="Invalid JSON format")

        request_type = request_data.get('type')
        model = None
        if 'scope' in request_data:
            model = request_data['scope'].get('model', '').replace('-', '_').replace('.', '_')
        query = request_data.get('query')
        chat_info = request_data.get('chat_info')

        if not request_type or request_type not in ['query', 'insert', 'remove', 'register']:
            result = {"errorCode": 102,
                      "errorDesc": "type exception, should one of ['query', 'insert', 'remove', 'register']",
                      "cacheHit": False, "delta_time": 0, "hit_query": '', "answer": ''}
            asyncio.create_task(save_query_info(result, model=model, query='', delta_time_log=0))
            raise HTTPException(status_code=102, detail="Type exception, should be one of ['query', 'insert', 'remove', 'register']")

    except Exception as e:
        request_data = raw_body if 'raw_body' in locals() else None
        result = {
            "errorCode": 103,
            "errorDesc": str(e),
            "cacheHit": False,
            "delta_time": 0,
            "hit_query": '',
            "answer": '',
            "para_dict": request_data
        }
        return result


    # model filter
    filter_resp = model_blacklist_filter(model, request_type)
    if isinstance(filter_resp, dict):
        return filter_resp

    if request_type == 'query':
        try:
            start_time = time.time()
            response = adapter.ChatCompletion.create_query(scope={"model": model}, query=query)
            delta_time = f"{round(time.time() - start_time, 2)}s"

            if response is None:
                result = {"errorCode": 0, "errorDesc": '', "cacheHit": False, "delta_time": delta_time, "hit_query": '', "answer": ''}
            elif response in ['adapt_query_exception']:
                result = {"errorCode": 201, "errorDesc": response, "cacheHit": False, "delta_time": delta_time,
                          "hit_query": '', "answer": ''}
            else:
                answer = response['data']
                hit_query = response['hitQuery']
                result = {"errorCode": 0, "errorDesc": '', "cacheHit": True, "delta_time": delta_time, "hit_query": hit_query, "answer": answer}

            delta_time_log = round(time.time() - start_time, 2)
            asyncio.create_task(save_query_info(result, model, query, delta_time_log))
            return result
        except Exception as e:
            result = {"errorCode": 202, "errorDesc": str(e), "cacheHit": False, "delta_time": 0,
                      "hit_query": '', "answer": ''}
            logging.info(f'result: {str(result)}')
            return result

    if request_type == 'insert':
        try:
            response = adapter.ChatCompletion.create_insert(model=model, chat_info=chat_info)
            if response == 'success':
                return {"errorCode": 0, "errorDesc": "", "writeStatus": "success"}
            else:
                return {"errorCode": 301, "errorDesc": response, "writeStatus": "exception"}
        except Exception as e:
            return {"errorCode": 303, "errorDesc": str(e), "writeStatus": "exception"}

    if request_type == 'remove':
        response = adapter.ChatCompletion.create_remove(model=model, remove_type=request_data.get("remove_type"), id_list=request_data.get("id_list"))
        if not isinstance(response, dict):
            return {"errorCode": 401, "errorDesc": "", "response": response, "removeStatus": "exception"}

        state = response.get('status')
        if state == 'success':
            return {"errorCode": 0, "errorDesc": "", "response": response, "writeStatus": "success"}
        else:
            return {"errorCode": 402, "errorDesc": "", "response": response, "writeStatus": "exception"}

    if request_type == 'register':
        response = adapter.ChatCompletion.create_register(model=model)
        if response in ['create_success', 'already_exists']:
            return {"errorCode": 0, "errorDesc": "", "response": response, "writeStatus": "success"}
        else:
            return {"errorCode": 502, "errorDesc": "", "response": response, "writeStatus": "exception"}

# TODO: 可以修改为在命令行中使用`uvicorn your_module_name:app --host 0.0.0.0 --port 5000 --reload`的命令启动
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)