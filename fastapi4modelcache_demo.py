# -*- coding: utf-8 -*-
import asyncio
from contextlib import asynccontextmanager
import uvicorn
import json
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request
from modelcache.cache import Cache
from modelcache.embedding import EmbeddingModel

@asynccontextmanager
async def lifespan(app: FastAPI):
    global cache
    cache, _ = await Cache.init(
        sql_storage="sqlite",
        vector_storage="faiss",
        embedding_model=EmbeddingModel.HUGGINGFACE_ALL_MPNET_BASE_V2,
        embedding_workers_num=2
    )
    yield

app = FastAPI(lifespan=lifespan)
cache: Cache = None

@app.get("/welcome")
async def first_fastapi():
    return "hello, modelcache!"

@app.post("/modelcache")
async def user_backend(request: Request):

    try:
        request_data = await request.json()
    except Exception:
        result = {"errorCode": 400, "errorDesc": "bad request", "cacheHit": False, "delta_time": 0, "hit_query": '', "answer": ''}
        return JSONResponse(status_code=400, content=result)

    try:
        return await cache.handle_request(request_data)
    except Exception as e:
        result = {"errorCode": 500, "errorDesc": str(e), "cacheHit": False, "delta_time": 0, "hit_query": '', "answer": ''}
        cache.save_query_resp(result, model='', query='', delta_time=0)
        return JSONResponse(status_code=500, content=result)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000, loop="asyncio", http="httptools")
