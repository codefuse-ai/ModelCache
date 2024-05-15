# -*- coding: utf-8 -*-
def model_blacklist_filter(model, request_type):
    black_list = ['DI_COPILOT_SECOND', 'DI_COPILOT_LAB', 'DI_COPILOT_THIRD']
    result = None
    if model in black_list:
        if request_type == 'query':
            result = {"errorCode": 105,
                      "errorDesc": "model: {} in blacklist".format(model),
                      "cacheHit": False, "delta_time": 0, "hit_query": '', "answer": ''}
        elif request_type == 'insert':
            result = {"errorCode": 305, "errorDesc": "model: {} in blacklist".format(model), "writeStatus": ""}

    return result


