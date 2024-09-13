# -*- coding: utf-8 -*-
import time
import json
import uuid
import requests


def run():
    url = 'http://127.0.0.1:5000/multicache'

    request_type = 'insert'
    scope = {"model": "multimodal_test"}
    # UUID = "820b0052-d9d8-11ee-95f1-52775e3e6fd1" + "==>" + str(time.time())
    UUID = str(uuid.uuid1()) + "==>" + str(time.time())
    img_data = "https://img0.baidu.com/it/u=1436460262,4166266890&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=282"
    query = {'text': ['父母带着孩子来这个地方可能会有什么顾虑'],
             'imageRaw': '',
             'imageUrl': img_data,
             'imageId': 'ccc'}
    answer = "应该注意小孩不要跑到铁轨上"
    chat_info = [{"query": query, "answer": answer}]
    data_dict = {'request_type': request_type, 'scope': scope, 'chat_info': chat_info, 'UUID': UUID}

    headers = {"Content-Type": "application/json"}
    res = requests.post(url, headers=headers, json=json.dumps(data_dict))
    res_text = res.text
    print('res_text: {}'.format(res_text))


if __name__ == '__main__':
    run()
