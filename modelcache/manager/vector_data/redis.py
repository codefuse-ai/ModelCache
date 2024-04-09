# -*- coding: utf-8 -*-
from typing import List
import numpy as np
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.commands.search.field import TagField, VectorField, NumericField
from redis.client import Redis

from gptcache.manager.vector_data.base import VectorBase, VectorData
from gptcache.utils import import_redis
from gptcache.utils.log import gptcache_log
from gptcache.utils.collection_util import get_collection_name
from gptcache.utils.collection_util import get_collection_prefix
import_redis()


class RedisVectorStore(VectorBase):
    def __init__(
        self,
        host: str = "localhost",
        port: str = "6379",
        username: str = "",
        password: str = "",
        table_suffix: str = "",
        dimension: int = 0,
        collection_prefix: str = "gptcache",
        top_k: int = 1,
        namespace: str = "",
    ):
        if dimension <= 0:
            raise ValueError(
                f"invalid `dim` param: {dimension} in the Milvus vector store."
            )
        self._client = Redis(
            host=host, port=int(port), username=username, password=password
        )
        self.top_k = top_k
        self.dimension = dimension
        self.collection_prefix = collection_prefix
        self.table_suffix = table_suffix
        self.namespace = namespace
        self.doc_prefix = f"{self.namespace}doc:"  # Prefix with the specified namespace
        # self._create_collection(collection_name)

    def _check_index_exists(self, index_name: str) -> bool:
        """Check if Redis index exists."""
        try:
            self._client.ft(index_name).info()
        except:  # pylint: disable=W0702
            gptcache_log.info("Index does not exist")
            return False
        gptcache_log.info("Index already exists")
        return True

    def create_collection(self, collection_name, index_prefix):
        dimension = self.dimension
        print('dimension: {}'.format(dimension))
        if self._check_index_exists(collection_name):
            gptcache_log.info(
                "The %s already exists, and it will be used directly", collection_name
            )
            return 'already_exists'
        else:
            # id_field_name = collection_name + '_' + "id"
            # embedding_field_name = collection_name + '_' + "vec"
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
            # definition = IndexDefinition(index_type=IndexType.HASH)
            definition = IndexDefinition(prefix=[index_prefix], index_type=IndexType.HASH)

            # create Index
            self._client.ft(collection_name).create_index(
                fields=fields, definition=definition
            )
            return 'create_success'

    def mul_add(self, datas: List[VectorData], model=None):
        # pipe = self._client.pipeline()
        for data in datas:
            id: int = data.id
            embedding = data.data.astype(np.float32).tobytes()
            # id_field_name = collection_name + '_' + "id"
            # embedding_field_name = collection_name + '_' + "vec"
            id_field_name = "data_id"
            embedding_field_name = "data_vector"
            obj = {id_field_name: id, embedding_field_name: embedding}
            index_prefix = get_collection_prefix(model, self.table_suffix)
            self._client.hset(f"{index_prefix}{id}", mapping=obj)

        #     obj = {
        #         "vector": data.data.astype(np.float32).tobytes(),
        #     }
        #     pipe.hset(f"{self.doc_prefix}{key}", mapping=obj)
        # pipe.execute()

    def search(self, data: np.ndarray, top_k: int = -1, model=None):
        collection_name = get_collection_name(model, self.table_suffix)
        print('collection_name: {}'.format(collection_name))
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
        # print('query_params: {}'.format(query_params))
        results = (
            self._client.ft(collection_name)
            .search(query, query_params=query_params)
            .docs
        )
        print('results: {}'.format(results))
        for i, doc in enumerate(results):
            print('doc: {}'.format(doc))
            print("id_field_name", getattr(doc, id_field_name), ", distance: ", doc.distance)
        return [(float(result.distance), int(getattr(result, id_field_name))) for result in results]

    def rebuild(self, ids=None) -> bool:
        pass

    def rebuild_col(self, model):
        resp_info = 'failed'
        if len(self.table_suffix) == 0:
            raise ValueError('table_suffix is none error,please check!')

        collection_name_model = get_collection_name(model, self.table_suffix)
        print('collection_name_model: {}'.format(collection_name_model))
        if self._check_index_exists(collection_name_model):
            try:
                self._client.ft(collection_name_model).dropindex(delete_documents=True)
            except Exception as e:
                raise ValueError(str(e))
        try:
            index_prefix = get_collection_prefix(model, self.table_suffix)
            self.create_collection(collection_name_model, index_prefix)
        except Exception as e:
            raise ValueError(str(e))
        return 'rebuild success'

        # print('remove collection_name_model: {}'.format(collection_name_model))
        # try:
        #     self._client.ft(collection_name_model).dropindex(delete_documents=True)
        #     resp_info = 'rebuild success'
        # except Exception as e:
        #     print('exception: {}'.format(e))
        #     resp_info = 'create only'
        # try:
        #     self.create_collection(collection_name_model)
        # except Exception as e:
        #     raise ValueError(str(e))
        # return resp_info

    def delete(self, ids) -> None:
        pipe = self._client.pipeline()
        for data_id in ids:
            pipe.delete(f"{self.doc_prefix}{data_id}")
        pipe.execute()

    def create(self, model=None):
        collection_name = get_collection_name(model, self.table_suffix)
        index_prefix = get_collection_prefix(model, self.table_suffix)
        return self.create_collection(collection_name, index_prefix)

    def get_collection_by_name(self, collection_name, table_suffix):
        pass
