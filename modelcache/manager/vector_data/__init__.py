# -*- coding: utf-8 -*-
from modelcache.utils.lazy_import import LazyImport

vector_manager = LazyImport(
    "vector_manager", globals(), "modelcache.manager.vector_data.manager"
)


def VectorBase(name: str, **kwargs):
    return vector_manager.VectorBase.get(name, **kwargs)
