# -*- coding: utf-8 -*-
import logging

from modelcache.adapter_mm.adapter_query import adapt_query
from modelcache.adapter_mm.adapter_insert import adapt_insert
from modelcache.adapter_mm.adapter_remove import adapt_remove
from modelcache.adapter_mm.adapter_register import adapt_register


class ChatCompletion(object):
    """Openai ChatCompletion Wrapper"""
    @classmethod
    def create_mm_query(cls, *args, **kwargs):
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
    def create_mm_insert(cls, *args, **kwargs):
        try:
            return adapt_insert(
                *args,
                **kwargs
            )
        except Exception as e:
            # return str(e)
            raise e

    @classmethod
    def create_mm_remove(cls, *args, **kwargs):
        try:
            return adapt_remove(
                *args,
                **kwargs
            )
        except Exception as e:
            logging.info('adapt_remove_e: {}'.format(e))
            return str(e)

    @classmethod
    def create_mm_register(cls, *args, **kwargs):
        try:
            return adapt_register(
                *args,
                **kwargs
            )
        except Exception as e:
            raise e


def construct_resp_from_cache(return_message, return_query):
    return {
        "modelcache": True,
        "hitQuery": return_query,
        "data": return_message,
        "errorCode": 0
    }
