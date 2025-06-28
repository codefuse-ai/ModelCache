# -*- coding: utf-8 -*-
from typing import Any, Callable, List, Tuple
import cachetools

from modelcache.manager.eviction.base import EvictionBase
from .arc_cache import ARC
from .wtinylfu_cache import W2TinyLFU


def popitem_wrapper(func, wrapper_func, clean_size):
    def wrapper(*args, **kwargs):
        keys = []
        try:
            keys = [func(*args, **kwargs)[0] for _ in range(clean_size)]
        except KeyError:
            pass
        wrapper_func(keys)
    return wrapper


class MemoryCacheEviction(EvictionBase):
    def __init__(self, policy: str, maxsize: int, clean_size: int, **kwargs):
        self._policy = policy.upper()
        self.model_to_cache = dict()
        self.maxsize = maxsize
        self.clean_size = clean_size
        self.kwargs = kwargs

    def create_cache(self, model: str):
        if self._policy == "LRU":
            cache = cachetools.LRUCache(maxsize=self.maxsize, **self.kwargs)
        elif self._policy == "LFU":
            cache = cachetools.LFUCache(maxsize=self.maxsize, **self.kwargs)
        elif self._policy == "FIFO":
            cache = cachetools.FIFOCache(maxsize=self.maxsize, **self.kwargs)
        elif self._policy == "RR":
            cache = cachetools.RRCache(maxsize=self.maxsize, **self.kwargs)
        elif self._policy == "WTINYLFU":
            cache = W2TinyLFU(maxsize=self.maxsize)
        elif self._policy == "ARC":
            cache = ARC(maxsize=self.maxsize)
        else:
            raise ValueError(f"Unknown policy {self.policy}")
        return cache

    def put(self, objs: List[Tuple[Any, Any]], model: str):
        cache = self.get_cache(model)
        for key, value in objs:
            cache[key] = value


    def get(self, obj: Any, model: str):
        cache = self.get_cache(model)
        return cache.get(obj)


    def clear(self, model: str):
        self.model_to_cache.pop(model, None)


    def get_cache(self, model: str):
        if not model in self.model_to_cache:
            self.model_to_cache[model] = self.create_cache(model)
        return self.model_to_cache[model]


    @property
    def policy(self) -> str:
        return self._policy

