# -*- coding: utf-8 -*-
from modelcache.utils.lazy_import import LazyImport
scalar_manager = LazyImport(
    "scalar_manager", globals(), "modelcache.manager.scalar_data.manager"
)


def CacheBase(name: str, **kwargs):
    return scalar_manager.CacheBase.get(name, **kwargs)
