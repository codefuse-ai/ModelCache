# -*- coding: utf-8 -*-
from modelcache import cache
from modelcache.utils.error import NotInitError


def adapt_remove(*args, **kwargs):
    chat_cache = kwargs.pop("cache_obj", cache)
    model = kwargs.pop("model", None)
    remove_type = kwargs.pop("remove_type", None)
    require_object_store = kwargs.pop("require_object_store", False)
    if require_object_store:
        assert chat_cache.data_manager.o, "Object store is required for adapter."
    if not chat_cache.has_init:
        raise NotInitError()

    # delete data
    if remove_type == 'delete_by_id':
        id_list = kwargs.pop("id_list", [])
        resp = chat_cache.data_manager.delete(id_list, model=model)

    elif remove_type == 'truncate_by_model':
        resp = chat_cache.data_manager.truncate(model)

    else:
        resp = "remove_type_error"
    return resp

