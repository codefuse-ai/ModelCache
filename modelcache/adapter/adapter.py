# -*- coding: utf-8 -*-
import logging

import openai
from modelcache.adapter.adapter_query import adapt_query
from modelcache.adapter.adapter_insert import adapt_insert
from modelcache.adapter.adapter_remove import adapt_remove
from modelcache.adapter.adapter_register import adapt_register


class ChatCompletion(openai.ChatCompletion):
    """Openai ChatCompletion Wrapper"""

    @classmethod
    def create_query(cls, *args, **kwargs):
        def cache_data_convert(cache_data, cache_query):
            return construct_resp_from_cache(cache_data, cache_query)
        try:
            return adapt_query(
                cache_data_convert,
                *args,
                **kwargs
            )
        except Exception as e:
            return str(e)

    @classmethod
    def create_insert(cls, *args, **kwargs):
        try:
            return adapt_insert(
                *args,
                **kwargs
            )
        except Exception as e:
            return str(e)

    @classmethod
    def create_remove(cls, *args, **kwargs):
        try:
            return adapt_remove(
                *args,
                **kwargs
            )
        except Exception as e:
            logging.info('adapt_remove_e: {}'.format(e))
            return str(e)

    @classmethod
    def create_register(cls, *args, **kwargs):
        try:
            return adapt_register(
                *args,
                **kwargs
            )
        except Exception as e:
            return str(e)


def construct_resp_from_cache(return_message, return_query):
    return {
        "modelcache": True,
        "hitQuery": return_query,
        "data": return_message,
        "errorCode": 0
    }
