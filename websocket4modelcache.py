# -*- coding: utf-8 -*-
from contextlib import asynccontextmanager
import uvicorn
import json
import asyncio
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect
from modelcache.cache import Cache
from modelcache.embedding import EmbeddingModel

@asynccontextmanager
async def lifespan(app: FastAPI):
    global cache
    cache, _ = await Cache.init(
        sql_storage="mysql",
        vector_storage="milvus",
        embedding_model=EmbeddingModel.HUGGINGFACE_ALL_MPNET_BASE_V2,
        embedding_workers_num=6
    )
    yield

app = FastAPI(lifespan=lifespan)
cache: Cache = None

@app.websocket("/modelcache")
async def user_backend(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            asyncio.create_task(handle_message(websocket, data))
    except WebSocketDisconnect as e:
        print(e)


async def handle_message(websocket,message):
    try:
        param_dict = json.loads(message)
    except Exception:
        await websocket.send_json({"errorCode": 400, "errorDesc": "bad request", "cacheHit": False, "delta_time": 0, "hit_query": '', "answer": ''})
        return

    request_id = param_dict.get("requestId")
    request_payload = param_dict.get("payload")
    if not request_id or not request_payload:
        await websocket.send_json({"errorCode": 400, "errorDesc": "bad request", "cacheHit": False, "delta_time": 0, "hit_query": '', "answer": ''})
        return
    try:
        result = await cache.handle_request(request_payload)
        await websocket.send_json({"requestId": request_id,"result": result})
    except Exception as e:
        error_result = {"errorCode": 500, "errorDesc": str(e), "cacheHit": False, "delta_time": 0, "hit_query": '', "answer": ''}
        cache.save_query_resp(error_result, model='', query='', delta_time=0)
        await websocket.send_json(error_result)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000, loop="asyncio", http="httptools")
