# -*- coding: utf-8 -*-
from modelcache import cache
from modelcache.utils.error import NotInitError
from modelcache.utils.time import time_cal


def adapt_insert(*args, **kwargs):
    chat_cache = kwargs.pop("cache_obj", cache)
    model = kwargs.pop("model", None)
    require_object_store = kwargs.pop("require_object_store", False)
    if require_object_store:
        assert chat_cache.data_manager.o, "Object store is required for adapter."
    if not chat_cache.has_init:
        raise NotInitError()
    cache_enable = chat_cache.cache_enable_func(*args, **kwargs)
    context = kwargs.pop("cache_context", {})
    embedding_data = None
    pre_embedding_data = chat_cache.insert_pre_embedding_func(
        kwargs,
        extra_param=context.get("pre_embedding_func", None),
        prompts=chat_cache.config.prompts,
    )
    chat_info = kwargs.pop("chat_info", [])
    llm_data = chat_info[-1]['answer']

    if cache_enable:
        embedding_data = time_cal(
            chat_cache.embedding_func,
            func_name="embedding",
            report_func=chat_cache.report.embedding,
        )(pre_embedding_data)

    chat_cache.data_manager.save(
        pre_embedding_data,
        llm_data,
        embedding_data,
        model=model,
        extra_param=context.get("save_func", None)
    )
    return 'success'
