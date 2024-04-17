# -*- coding: utf-8 -*-
from modelcache import cache


def adapt_register(*args, **kwargs):
    chat_cache = kwargs.pop("cache_obj", cache)
    model = kwargs.pop("model", None)
    mm_type = kwargs.pop("mm_type", None)
    if model is None or len(model) == 0:
        return ValueError('')

    print('mm_type: {}'.format(mm_type))
    print('model: {}'.format(model))
    register_resp = chat_cache.data_manager.create_index(model, mm_type)
    print('register_resp: {}'.format(register_resp))
    return register_resp
