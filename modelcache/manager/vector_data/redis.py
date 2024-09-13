# -*- coding: utf-8 -*-
from typing import List
<<<<<<< HEAD

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
=======
import numpy as np
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.commands.search.field import TagField, VectorField, NumericField
from redis.client import Redis

from modelcache.manager.vector_data.base import VectorBase, VectorData
from modelcache.utils import import_redis
from modelcache.utils.log import modelcache_log
from modelcache.utils.index_util import get_index_name
from modelcache.utils.index_util import get_index_prefix
import_redis()
>>>>>>> main


class RedisVectorStore(VectorBase):
    def __init__(
        self,
        host: str = "localhost",
        port: str = "6379",
        username: str = "",
        password: str = "",
        dimension: int = 0,
<<<<<<< HEAD
        collection_name: str = "gptcache",
        top_k: int = 1,
        namespace: str = "",
    ):
=======
        top_k: int = 1,
        namespace: str = "",
    ):
        if dimension <= 0:
            raise ValueError(
                f"invalid `dim` param: {dimension} in the Redis vector store."
            )
>>>>>>> main
        self._client = Redis(
            host=host, port=int(port), username=username, password=password
        )
        self.top_k = top_k
        self.dimension = dimension
<<<<<<< HEAD
        self.collection_name = collection_name
        self.namespace = namespace
        self.doc_prefix = f"{self.namespace}doc:"  # Prefix with the specified namespace
        self._create_collection(collection_name)
=======
        self.namespace = namespace
        self.doc_prefix = f"{self.namespace}doc:"
>>>>>>> main

    def _check_index_exists(self, index_name: str) -> bool:
        """Check if Redis index exists."""
        try:
            self._client.ft(index_name).info()
<<<<<<< HEAD
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
=======
        except:
            modelcache_log.info("Index does not exist")
            return False
        modelcache_log.info("Index already exists")
        return True

    def create_index(self, index_name, index_prefix):
        dimension = self.dimension
        if self._check_index_exists(index_name):
            modelcache_log.info(
                "The %s already exists, and it will be used directly", index_name
            )
            return 'already_exists'
        else:
            id_field_name = "data_id"
            embedding_field_name = "data_vector"

            id = NumericField(name=id_field_name)
            embedding = VectorField(embedding_field_name,
                                    "HNSW", {
                                        "TYPE": "FLOAT32",
                                        "DIM": dimension,
                                        "DISTANCE_METRIC": "L2",
                                        "INITIAL_CAP": 1000,
                                    }
                                    )
            fields = [id, embedding]
            definition = IndexDefinition(prefix=[index_prefix], index_type=IndexType.HASH)

            # create Index
            self._client.ft(index_name).create_index(
                fields=fields, definition=definition
            )
            return 'create_success'

    def mul_add(self, datas: List[VectorData], model=None):
        # pipe = self._client.pipeline()
        for data in datas:
            id: int = data.id
            embedding = data.data.astype(np.float32).tobytes()
            id_field_name = "data_id"
            embedding_field_name = "data_vector"
            obj = {id_field_name: id, embedding_field_name: embedding}
            index_prefix = get_index_prefix(model)
            self._client.hset(f"{index_prefix}{id}", mapping=obj)

    def search(self, data: np.ndarray, top_k: int = -1, model=None):
        index_name = get_index_name(model)
        id_field_name = "data_id"
        embedding_field_name = "data_vector"
        base_query = f'*=>[KNN 2 @{embedding_field_name} $vector AS distance]'
        query = (
            Query(base_query)
            .sort_by("distance")
            .return_fields(id_field_name, "distance")
            .dialect(2)
        )

        query_params = {"vector": data.astype(np.float32).tobytes()}
        results = (
            self._client.ft(index_name)
            .search(query, query_params=query_params)
            .docs
        )
        return [(float(result.distance), int(getattr(result, id_field_name))) for result in results]
>>>>>>> main

    def rebuild(self, ids=None) -> bool:
        pass

<<<<<<< HEAD
=======
    def rebuild_col(self, model):
        index_name_model = get_index_name(model)
        if self._check_index_exists(index_name_model):
            try:
                self._client.ft(index_name_model).dropindex(delete_documents=True)
            except Exception as e:
                raise ValueError(str(e))
        try:
            index_prefix = get_index_prefix(model)
            self.create_index(index_name_model, index_prefix)
        except Exception as e:
            raise ValueError(str(e))
        # return 'rebuild success'

>>>>>>> main
    def delete(self, ids) -> None:
        pipe = self._client.pipeline()
        for data_id in ids:
            pipe.delete(f"{self.doc_prefix}{data_id}")
<<<<<<< HEAD
        pipe.execute()
=======
        pipe.execute()

    def create(self, model=None):
        index_name = get_index_name(model)
        index_prefix = get_index_prefix(model)
        return self.create_index(index_name, index_prefix)

    def get_index_by_name(self, index_name):
        pass
>>>>>>> main
