# -*- coding: utf-8 -*-
"""
register index for redis
"""
import json
import requests


def run():
    url = 'http://127.0.0.1:5000/modelcache'
    type = 'register'
    scope = {"model": "CODEGPT-1117"}
    data = {'type': type, 'scope': scope}
    headers = {"Content-Type": "application/json"}
    res = requests.post(url, headers=headers, json=json.dumps(data))
    res_text = res.text


if __name__ == '__main__':
    run()