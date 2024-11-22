# -*- coding: utf-8 -*-
import json
import requests


def run():
    url = 'http://127.0.0.1:5000/modelcache'
    type = 'query'
    scope = {"model": "CODEGPT-1109"}
    system_conten = """
    """
    user_content = """
    """

    query = [{"role": "system", "content": system_conten}, {"role": "user", "content": user_content}]
    data = {'type': type, 'scope': scope, 'query': query}

    headers = {"Content-Type": "application/json"}
    res = requests.post(url, headers=headers, json=json.dumps(data))
    res_text = res.text

    print("data_query_long:", res.status_code, res_text)

if __name__ == '__main__':
    run()
