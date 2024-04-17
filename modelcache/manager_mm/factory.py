# -*- coding: utf-8 -*-
from typing import Union, Callable
from modelcache.manager_mm import CacheBase, VectorBase, ObjectBase
from modelcache.manager_mm.data_manager import SSDataManager, MapDataManager


def get_data_manager(
    cache_base: Union[CacheBase, str] = None,
    vector_base: Union[VectorBase, str] = None,
    object_base: Union[ObjectBase, str] = None,
    max_size: int = 1000,
    clean_size: int = None,
    eviction: str = "LRU",
    data_path: str = "data_map.txt",
    get_data_container: Callable = None,
):
    if not cache_base and not vector_base:
        return MapDataManager(data_path, max_size, get_data_container)

    if isinstance(cache_base, str):
        cache_base = CacheBase(name=cache_base)
    if isinstance(vector_base, str):
        vector_base = VectorBase(name=vector_base)
    if isinstance(object_base, str):
        object_base = ObjectBase(name=object_base)
    assert cache_base and vector_base
    return SSDataManager(cache_base, vector_base, object_base, max_size, clean_size, eviction)
