# -*- coding: utf-8 -*-
from flask import Flask, request
import json
from modelcache.cache import Cache

# 创建一个Flask实例
app = Flask(__name__)

cache = Cache.init("sqlite","faiss")

@app.route('/welcome')
def first_flask():  # 视图函数
    return 'hello, modelcache!'


@app.route('/modelcache', methods=['GET', 'POST'])
def user_backend():
    param_dict = {}
    try:
        if request.method == 'POST':
            param_dict = request.json
        elif request.method == 'GET':
            param_dict = request.args

        return json.dumps(cache.handle_request(param_dict))
    except Exception as e:
        result = {"errorCode": 101, "errorDesc": str(e), "cacheHit": False, "delta_time": 0, "hit_query": '',
                  "answer": ''}
        cache.save_query_resp(result, model='', query='', delta_time=0)
        return json.dumps(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
