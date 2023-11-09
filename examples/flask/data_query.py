# -*- coding: utf-8 -*-
import json
import requests


def run():
    url = 'http://127.0.0.1:5000/modelcache'
    type = 'query'
    scope = {"model": "CODEGPT-1109"}
    query = [{"role": "system", "content": "你是一个python助手"}, {"role": "user", "content": "hello"}]
    data = {'type': type, 'scope': scope, 'query': query}

    headers = {"Content-Type": "application/json"}
    res = requests.post(url, headers=headers, json=json.dumps(data))
    res_text = res.text
    print('res_text: {}'.format(res_text))


if __name__ == '__main__':
    run()
