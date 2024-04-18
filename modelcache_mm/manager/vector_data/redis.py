# -*- coding: utf-8 -*-
from typing import List
import numpy as np
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, NumericField
from redis.client import Redis

from modelcache_mm.manager.vector_data.base import VectorBase, VectorData
from modelcache_mm.utils import import_redis
from modelcache_mm.utils.log import modelcache_log
from modelcache_mm.utils.index_util import get_mm_index_name
from modelcache_mm.utils.index_util import get_mm_index_prefix
import_redis()


class RedisVectorStore(VectorBase):
    def __init__(
        self,
        host: str = "localhost",
        port: str = "6379",
        username: str = "",
        password: str = "",
        mm_dimension: int = 0,
        i_dimension: int = 0,
        t_dimension: int = 0,
        top_k: int = 1,
        namespace: str = "",
    ):
        if mm_dimension <= 0:
            raise ValueError(
                f"invalid `dim` param: {mm_dimension} in the Milvus vector store."
            )
        self._client = Redis(
            host=host, port=int(port), username=username, password=password
        )
        self.top_k = top_k
        self.mm_dimension = mm_dimension
        self.i_dimension = i_dimension
        self.t_dimension = t_dimension
        self.namespace = namespace
        self.doc_prefix = f"{self.namespace}doc:"

    def _check_index_exists(self, index_name: str) -> bool:
        """Check if Redis index exists."""
        try:
            self._client.ft(index_name).info()
        except:
            modelcache_log.info("Index does not exist")
            return False
        modelcache_log.info("Index already exists")
        return True

    def create_index(self, index_name, type, index_prefix):
        # dimension = self.dimension
        if type == 'IMG_TEXT':
            dimension = self.mm_dimension
        elif type == 'IMG':
            dimension = self.i_dimension
        elif type == 'TEXT':
            dimension = self.t_dimension
        else:
            raise ValueError('dimension type exception')
        print('dimension: {}'.format(dimension))
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

    def add(self, datas: List[VectorData], model=None, mm_type=None):
        for data in datas:
            id: int = data.id
            embedding = data.data.astype(np.float32).tobytes()
            index_prefix = get_mm_index_prefix(model, mm_type)
            id_field_name = "data_id"
            embedding_field_name = "data_vector"
            obj = {id_field_name: id, embedding_field_name: embedding}
            self._client.hset(f"{index_prefix}{id}", mapping=obj)

    def search(self, data: np.ndarray, top_k: int = -1, model=None, mm_type=None):
        index_name = get_mm_index_name(model, mm_type)
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

    def create(self, model=None, mm_type=None):
        collection_name_model = get_mm_index_name(model, mm_type)
        try:
            index_prefix = get_mm_index_prefix(model, mm_type)
            self.create_index(collection_name_model, mm_type, index_prefix)
        except Exception as e:
            raise ValueError(str(e))
        return 'success'


    def rebuild(self, ids=None) -> bool:
        pass

    def rebuild_idx(self, model, mm_type=None):
        for mm_type in ['IMG_TEXT', 'TEXT']:
            index_name = get_mm_index_name(model, mm_type)
            print('remove index_name: {}'.format(index_name))
            if self._check_index_exists(index_name):
                try:
                    self._client.ft(index_name).dropindex(delete_documents=True)
                except Exception as e:
                    raise ValueError(str(e))
            try:
                index_prefix = get_mm_index_prefix(model, mm_type)
                self.create_index(index_name, mm_type, index_prefix)
            except Exception as e:
                raise ValueError(str(e))

    def delete(self, ids) -> None:
        pipe = self._client.pipeline()
        for data_id in ids:
            pipe.delete(f"{self.doc_prefix}{data_id}")
        pipe.execute()

    def create(self, model=None, type=None):
        index_name = get_mm_index_name(model, type)
        index_prefix = get_mm_index_prefix(model, type)
        return self.create_index(index_name, type, index_prefix)

    def get_index_by_name(self, index_name):
        pass
