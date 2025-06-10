# -*- coding: utf-8 -*-
import asyncio
import websockets
import json
from modelcache.cache import Cache

# Initialize the cache
cache = Cache.init("mysql", "milvus")


async def handle_client(websocket):
    async for message in websocket:
        # Parse JSON
        try:
            param_dict = json.loads(message)
        except json.JSONDecodeError:
            await websocket.send(json.dumps({"errorCode": 400, "errorDesc": "bad request"}))
            continue

        request_id = param_dict.get("requestId")
        request_payload = param_dict.get("payload")
        if not request_id or not request_payload:
            await websocket.send(json.dumps({"errorCode": 400, "errorDesc": "bad request"}))
            continue
        asyncio.create_task(process_and_respond(websocket, request_id, request_payload))


async def process_and_respond(websocket,request_id, request_payload):
    try:
        result = cache.handle_request(request_payload)
        await websocket.send(json.dumps({"requestId": request_id,"result": result}))
    except Exception as e:
        error_result = {"errorCode": 102, "errorDesc": str(e), "cacheHit": False, "delta_time": 0, "hit_query": '',
                        "answer": ''}
        cache.save_query_resp(error_result, model='', query='', delta_time=0)
        await websocket.send(json.dumps(error_result))


async def main():
    print("WebSocket server starting on ws://0.0.0.0:5000")
    async with websockets.serve(handle_client, "0.0.0.0", 5000):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
