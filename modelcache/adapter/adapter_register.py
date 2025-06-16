# -*- coding: utf-8 -*-
import asyncio


async def adapt_register(*args, **kwargs):
    chat_cache = kwargs.pop("cache_obj")
    model = kwargs.pop("model", None)
    if model is None or len(model) == 0:
        return ValueError('')

    register_resp = await asyncio.to_thread(
        chat_cache.data_manager.create_index,
        model
    )

    return register_resp
