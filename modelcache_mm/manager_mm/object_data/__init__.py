# -*- coding: utf-8 -*-
from modelcache.utils.lazy_import import LazyImport
object_manager = LazyImport(
    "object_manager", globals(), "modelcache.manager.object_data.manager"
)


def ObjectBase(name: str, **kwargs):
    return object_manager.ObjectBase.get(name, **kwargs)
