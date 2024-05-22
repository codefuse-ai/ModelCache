# -*- coding: utf-8 -*-
"""
register index for redis
"""
import json
import requests


def run():
    url = 'http://127.0.0.1:5000/multicache'
    request_type = 'remove'
    scope = {"model": "multimodal_test"}
    remove_type = 'truncate_by_model'
    data = {'request_type': request_type, 'scope': scope, 'remove_type': remove_type}

    headers = {"Content-Type": "application/json"}
    res = requests.post(url, headers=headers, json=json.dumps(data))
    res_text = res.text
    print('res_text: {}'.format(res_text))


if __name__ == '__main__':
    run()
