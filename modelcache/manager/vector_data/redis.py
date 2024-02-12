# -*- coding: utf-8 -*-
from typing import List

import numpy as np
from modelcache.manager.vector_data.base import VectorBase, VectorData
from modelcache.utils import import_redis
from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from modelcache.utils.log import modelcache_log

import_redis()
#
# from redis.commands.search.indexDefinition import IndexDefinition, IndexType
# from redis.commands.search.query import Query
# from redis.commands.search.field import TagField, VectorField
# from redis.client import Redis


class RedisVectorStore(VectorBase):
    def __init__(
        self,
        host: str = "localhost",
        port: str = "6379",
        username: str = "",
        password: str = "",
        dimension: int = 0,
        collection_name: str = "gptcache",
        top_k: int = 1,
        namespace: str = "",
    ):
        self._client = Redis(
            host=host, port=int(port), username=username, password=password
        )
        self.top_k = top_k
        self.dimension = dimension
        self.collection_name = collection_name
        self.namespace = namespace
        self.doc_prefix = f"{self.namespace}doc:"  # Prefix with the specified namespace
        self._create_collection(collection_name)

    def _check_index_exists(self, index_name: str) -> bool:
        """Check if Redis index exists."""
        try:
            self._client.ft(index_name).info()
        except:  # pylint: disable=W0702
            gptcache_log.info("Index does not exist")
            return False
        gptcache_log.info("Index already exists")
        return True

    def _create_collection(self, collection_name):
        if self._check_index_exists(collection_name):
            gptcache_log.info(
                "The %s already exists, and it will be used directly", collection_name
            )
        else:
            schema = (
                TagField("tag"),  # Tag Field Name
                VectorField(
                    "vector",  # Vector Field Name
                    "FLAT",
                    {  # Vector Index Type: FLAT or HNSW
                        "TYPE": "FLOAT32",  # FLOAT32 or FLOAT64
                        "DIM": self.dimension,  # Number of Vector Dimensions
                        "DISTANCE_METRIC": "COSINE",  # Vector Search Distance Metric
                    },
                ),
            )
            definition = IndexDefinition(
                prefix=[self.doc_prefix], index_type=IndexType.HASH
            )

            # create Index
            self._client.ft(collection_name).create_index(
                fields=schema, definition=definition
            )

    def mul_add(self, datas: List[VectorData]):
        pipe = self._client.pipeline()

        for data in datas:
            key: int = data.id
            obj = {
                "vector": data.data.astype(np.float32).tobytes(),
            }
            pipe.hset(f"{self.doc_prefix}{key}", mapping=obj)

        pipe.execute()

    def search(self, data: np.ndarray, top_k: int = -1):
        query = (
            Query(
                f"*=>[KNN {top_k if top_k > 0 else self.top_k} @vector $vec as score]"
            )
            .sort_by("score")
            .return_fields("id", "score")
            .paging(0, top_k if top_k > 0 else self.top_k)
            .dialect(2)
        )
        query_params = {"vec": data.astype(np.float32).tobytes()}
        results = (
            self._client.ft(self.collection_name)
            .search(query, query_params=query_params)
            .docs
        )
        return [(float(result.score), int(result.id[len(self.doc_prefix):])) for result in results]

    def rebuild(self, ids=None) -> bool:
        pass

    def delete(self, ids) -> None:
        pipe = self._client.pipeline()
        for data_id in ids:
            pipe.delete(f"{self.doc_prefix}{data_id}")
        pipe.execute()