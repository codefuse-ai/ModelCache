# -*- coding: utf-8 -*-
import json
import requests


def run():
    url = 'http://127.0.0.1:5000/modelcache'
    type = 'insert'
    scope = {"model": "CODEGPT-1117"}
    chat_info = [{"query": [{"role": "system", "content": "你是一个python助手"}, {"role": "user", "content": "hello"}],
                  "answer": "你好，我是智能助手，请问有什么能帮您!"}]
    data = {'type': type, 'scope': scope, 'chat_info': chat_info}
    headers = {"Content-Type": "application/json"}
    res = requests.post(url, headers=headers, json=json.dumps(data))
    res_text = res.text
    print('res_text: {}'.format(res_text))


if __name__ == '__main__':
    run()
