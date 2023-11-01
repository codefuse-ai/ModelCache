# -*- coding: utf-8 -*-
from modelcache.utils import import_sql_client
from modelcache.utils.error import NotFoundError


class CacheBase:
    """
    CacheBase to manager the cache storage.
    """

    def __init__(self):
        raise EnvironmentError(
            "CacheBase is designed to be instantiated, please using the `CacheBase.get(name)`."
        )

    @staticmethod
    def get(name, **kwargs):
        if name in ["sqlite", "mysql"]:
            from modelcache.manager.scalar_data.sql_storage import SQLStorage
            config = kwargs.get("config")
            # db_name = kwargs.get("db_name")
            import_sql_client(name)
            cache_base = SQLStorage(db_type=name, config=config)
        else:
            raise NotFoundError("cache store", name)
        return cache_base
