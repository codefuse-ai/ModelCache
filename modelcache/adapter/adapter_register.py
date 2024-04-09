# -*- coding: utf-8 -*-
from modelcache import cache


def adapt_register(*args, **kwargs):
    chat_cache = kwargs.pop("cache_obj", cache)
    model = kwargs.pop("model", None)
    if model is None or len(model) == 0:
        return ValueError('')

    register_resp = chat_cache.data_manager.create_index(model)
    print('register_resp: {}'.format(register_resp))
    return register_resp
