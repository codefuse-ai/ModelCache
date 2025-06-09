# -*- coding: utf-8 -*-
import uvicorn
import json
from fastapi import FastAPI, Request, HTTPException
from modelcache.cache import Cache

#创建一个FastAPI实例
app = FastAPI()

cache = Cache.init("mysql", "milvus")

@app.get("/welcome")
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
                cache.save_query_info(result, model='', query='', delta_time_log=0)
                raise HTTPException(status_code=101, detail="Invalid JSON format")
        else:
            request_data = raw_body

        # 确保request_data是字典对象
        if isinstance(request_data, str):
            try:
                request_data = json.loads(request_data)
            except json.JSONDecodeError:
                raise HTTPException(status_code=101, detail="Invalid JSON format")

        return cache.handle_request(request_data)

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

# TODO: 可以修改为在命令行中使用`uvicorn your_module_name:app --host 0.0.0.0 --port 5000 --reload`的命令启动
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)