# -*- coding: utf-8 -*-
"""
register index for redis
"""
import json
import requests


def run():
    url = 'http://127.0.0.1:5000/multicache'
    request_type = 'register'
    scope = {"model": "multimodal_test"}
    type = 'IMG_TEXT'
    data = {'request_type': request_type, 'scope': scope, 'type': type}
    headers = {"Content-Type": "application/json"}
    res = requests.post(url, headers=headers, json=json.dumps(data))
    res_text = res.text
    print('res_text: {}'.format(res_text))


if __name__ == '__main__':
    run()