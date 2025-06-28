# -*- coding: utf-8 -*-
import asyncio

from flask import Flask, request, jsonify
from modelcache.cache import Cache
from modelcache.embedding import EmbeddingModel


async def main():

    # 创建一个Flask实例
    app = Flask(__name__)

    cache,loop = await Cache.init(
        sql_storage="mysql",
        vector_storage="milvus",
        embedding_model=EmbeddingModel.HUGGINGFACE_ALL_MPNET_BASE_V2,
        embedding_workers_num=2
    )

    @app.route('/welcome')
    def first_flask():  # 视图函数
        return 'hello, modelcache!'


    @app.post('/modelcache')
    def user_backend():
        try:
            param_dict = request.json
        except Exception:
            result = {"errorCode": 400, "errorDesc": "bad request", "cacheHit": False, "delta_time": 0, "hit_query": '',"answer": ''}
            return jsonify(result), 400

        try:
            result = asyncio.run_coroutine_threadsafe(
                cache.handle_request(param_dict), loop
            ).result()
            return jsonify(result), 200
        except Exception as e:
            result = {"errorCode": 500, "errorDesc": str(e), "cacheHit": False, "delta_time": 0, "hit_query": '',"answer": ''}
            cache.save_query_resp(result, model='', query='', delta_time=0)
            return jsonify(result), 500

    await asyncio.to_thread(app.run, host='0.0.0.0', port=5000)


if __name__ == '__main__':
    asyncio.run(main())
