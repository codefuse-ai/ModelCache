# -*- coding: utf-8 -*-
from modelcache_mm.utils.lazy_import import LazyImport

vector_manager = LazyImport(
    "vector_manager", globals(), "modelcache_mm.manager.vector_data.manager"
)


def VectorBase(name: str, **kwargs):
    return vector_manager.VectorBase.get(name, **kwargs)
