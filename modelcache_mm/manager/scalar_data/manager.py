# -*- coding: utf-8 -*-
from modelcache_mm.utils import import_sql_client
from modelcache_mm.utils.error import NotFoundError

SQL_URL = {"sqlite": "./sqlite.db"}


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

        if name in ["mysql", "oceanbase"]:
            from modelcache_mm.manager.scalar_data.sql_storage import SQLStorage
            config = kwargs.get("config")
            import_sql_client(name)
            cache_base = SQLStorage(db_type=name, config=config)
        elif name == 'sqlite':
            from modelcache_mm.manager.scalar_data.sql_storage_sqlite import SQLStorage
            sql_url = kwargs.get("sql_url", SQL_URL[name])
            cache_base = SQLStorage(db_type=name, url=sql_url)
        else:
            raise NotFoundError("cache store", name)
        return cache_base
