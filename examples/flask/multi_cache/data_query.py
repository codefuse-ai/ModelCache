# -*- coding: utf-8 -*-
import json
import requests
import uuid
import time


def run():
    url = 'http://127.0.0.1:5000/multicache'
    request_type = 'query'
    UUID = str(uuid.uuid1()) + "==>" + str(time.time())
    scope = {"model": "multimodal_test"}
    img_data = "https://img0.baidu.com/it/u=1436460262,4166266890&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=282"
    query = {'text': ['父母带着孩子来这个地方可能会有什么顾虑'],
             'imageRaw': '',
             'imageUrl': img_data,
             'multiType': 'IMG_TEXT'}

    data = {'request_type': request_type, 'scope': scope, 'query': query, 'UUID': UUID}

    headers = {"Content-Type": "application/json"}
    res = requests.post(url, headers=headers, json=json.dumps(data))
    res_text = res.text
    print('res_text: {}'.format(res_text))


if __name__ == '__main__':
    run()
