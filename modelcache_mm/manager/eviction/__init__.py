# -*- coding: utf-8 -*-
from modelcache.utils.lazy_import import LazyImport

eviction_manager = LazyImport(
    "eviction_manager", globals(), "modelcache.manager.eviction.manager"
)


def EvictionBase(name: str, **kwargs):
    return eviction_manager.EvictionBase.get(name, **kwargs)
