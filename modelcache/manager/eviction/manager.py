# -*- coding: utf-8 -*-
from typing import Callable, List, Any
from modelcache.utils.error import NotFoundError


class EvictionBase:
    """
    EvictionBase to evict the cache data.
    """

    def __init__(self):
        raise EnvironmentError(
            "EvictionBase is designed to be instantiated, "
            "please using the `EvictionBase.get(name, policy, maxsize, clean_size)`."
        )

    @staticmethod
    def get(name: str, policy: str, maxsize: int, clean_size: int, on_evict: Callable[[List[Any]], None], **kwargs):
        if name in "memory":
            from modelcache.manager.eviction.memory_cache import MemoryCacheEviction

            eviction_base = MemoryCacheEviction(policy, maxsize, clean_size, on_evict, **kwargs)
        else:
            raise NotFoundError("eviction base", name)
        return eviction_base
