# -*- coding: utf-8 -*-
import logging
from modelcache.adapter.adapter_query import adapt_query
from modelcache.adapter.adapter_insert import adapt_insert
from modelcache.adapter.adapter_remove import adapt_remove
from modelcache.adapter.adapter_register import adapt_register


class ChatCompletion(object):
    """Openai ChatCompletion Wrapper"""

    @classmethod
    async def create_query(cls, *args, **kwargs):
        def cache_data_convert(cache_data, cache_query):
            return construct_resp_from_cache(cache_data, cache_query)
        try:
            return await adapt_query(
                cache_data_convert,
                *args,
                **kwargs
            )
        except Exception as e:
            print(e)
            return str(e)

    @classmethod
    async def create_insert(cls, *args, **kwargs):
        try:
            return await adapt_insert(
                *args,
                **kwargs
            )
        except Exception as e:
            print(e)
            return str(e)

    @classmethod
    async def create_remove(cls, *args, **kwargs):
        try:
            return await adapt_remove(
                *args,
                **kwargs
            )
        except Exception as e:
            print(e)
            return str(e)

    @classmethod
    async def create_register(cls, *args, **kwargs):
        try:
            return await adapt_register(
                *args,
                **kwargs
            )
        except Exception as e:
            print(e)
            return str(e)


def construct_resp_from_cache(return_message, return_query):
    return {
        "modelcache": True,
        "hitQuery": return_query,
        "data": return_message,
        "errorCode": 0
    }
