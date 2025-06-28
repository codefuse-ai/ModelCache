# -*- coding: utf-8 -*-
import logging
from typing import List
from uuid import uuid4
import numpy as np

from modelcache.embedding import MetricType
from modelcache.utils import import_pymilvus
from modelcache.utils.log import modelcache_log
from modelcache.manager.vector_data.base import VectorStorage, VectorData


import_pymilvus()

from pymilvus import (  # pylint: disable=C0413
    connections,
    utility,
    FieldSchema,
    DataType,
    CollectionSchema,
    Collection,
    MilvusException,
)


class Milvus(VectorStorage):

    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        user: str = "",
        password: str = "",
        secure: bool = False,
        collection_name: str = "modelcache",
        dimension: int = 0,
        top_k: int = 1,
        index_params: dict = None,
        search_params: dict = None,
        local_mode: bool = False,
        local_data: str = "./milvus_data",
        metric_type: MetricType = MetricType.COSINE,
    ):
        if dimension <= 0:
            raise ValueError(
                f"invalid `dim` param: {dimension} in the Milvus vector store."
            )
        self._local_mode = local_mode
        self._local_data = local_data
        self.dimension = dimension
        self.top_k = top_k
        if self._local_mode:
            self._create_local(port, local_data)
        self._connect(host, port, user, password, secure)
        self.collection_name = collection_name
        self.search_params = {
            "IVF_FLAT": {"metric_type": metric_type.value, "params": {"nprobe": 10}},
            "IVF_SQ8": {"metric_type": metric_type.value, "params": {"nprobe": 10}},
            "IVF_PQ": {"metric_type": metric_type.value, "params": {"nprobe": 10}},
            "HNSW": {"metric_type": metric_type.value, "params": {"ef": 10}},
            "RHNSW_FLAT": {"metric_type": metric_type.value, "params": {"ef": 10}},
            "RHNSW_SQ": {"metric_type": metric_type.value, "params": {"ef": 10}},
            "RHNSW_PQ": {"metric_type": metric_type.value, "params": {"ef": 10}},
            "IVF_HNSW": {"metric_type": metric_type.value, "params": {"nprobe": 10, "ef": 10}},
            "ANNOY": {"metric_type": metric_type.value, "params": {"search_k": 10}},
            "AUTOINDEX": {"metric_type": metric_type.value, "params": {}},
        }
        self.index_params ={
            "metric_type": metric_type.value,
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 64},
        }
        self.collections = dict()


    def _connect(self, host, port, user, password, secure):
        try:
            i = [
                connections.get_connection_addr(x[0])
                for x in connections.list_connections()
            ].index({"host": host, "port": port})
            self.alias = connections.list_connections()[i][0]
        except ValueError:
            # Connect to the Milvus instance using the passed in Environment variables
            self.alias = uuid4().hex
            connections.connect(
                alias=self.alias,
                host=host,
                port=port,
                user=user,  # type: ignore
                password=password,  # type: ignore
                secure=secure,
                timeout=10
            )


    def _create_collection(self, collection_name):
        if not utility.has_collection(collection_name, using=self.alias):
            schema = [
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    max_length=36,
                    is_primary=True,
                    auto_id=False,
                ),
                FieldSchema(
                    name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension
                ),
            ]
            schema = CollectionSchema(schema)

            new_collection = Collection(
                collection_name,
                schema=schema,
                consistency_level="Session",
                using=self.alias,
            )
        else:
            modelcache_log.warning("The %s collection already exists, and it will be used directly.", collection_name)
            new_collection = Collection(
                collection_name, consistency_level="Session", using=self.alias
            )

        self.collections[collection_name] = new_collection

        if len(new_collection.indexes) == 0:
            try:
                modelcache_log.info("Attempting creation of Milvus index.")
                new_collection.create_index("embedding", index_params=self.index_params)
                modelcache_log.info("Creation of Milvus index successful.")
            except MilvusException as e:
                modelcache_log.warning("Error with building index: %s, and attempting creation of default index.", e)
                i_p = {"metric_type": "L2", "index_type": "AUTOINDEX", "params": {}}
                new_collection.create_index("embedding", index_params=i_p)
                self.index_params = i_p
        else:
            self.index_params = new_collection.indexes[0].to_dict()["index_param"]

        new_collection.load()


    def _get_collection(self, collection_name):
        if collection_name not in self.collections:
            self._create_collection(collection_name)
        return self.collections[collection_name]

    def mul_add(self, datas: List[VectorData], model=None):
        collection_name_model = self.collection_name + '_' + model
        col = self._get_collection(collection_name_model)
        data_array, id_array = map(list, zip(*((data.data, data.id) for data in datas)))
        np_data = np.array(data_array).astype("float32")
        entities = [id_array, np_data]
        col.insert(entities)


    def search(self, data: np.ndarray, top_k: int = -1, model=None):
        if top_k == -1:
            top_k = self.top_k
        collection_name_model = self.collection_name + '_' + model
        col = self._get_collection(collection_name_model)
        search_result = col.search(
            data=data.reshape(1, -1).tolist(),
            anns_field="embedding",
            param=self.search_params,
            limit=top_k,
        )
        return list(zip(search_result[0].distances, search_result[0].ids))


    def delete(self, ids, model=None):
        collection_name_model = self.collection_name + '_' + model
        col = self._get_collection(collection_name_model)

        del_ids = ",".join([f'"{x}"' for x in ids])
        resp = col.delete(f"id in [{del_ids}]")
        delete_count = resp.delete_count
        return delete_count

    def rebuild_col(self, model):
        collection_name_model = self.collection_name + '_' + model

        # if col exist, drop col
        if not utility.has_collection(collection_name_model, using=self.alias):
            return 'model collection not found, please check!'
        utility.drop_collection(collection_name_model, using=self.alias)
        try:
            self._create_collection(collection_name_model)
        except Exception as e:
            logging.info('create_collection: {}'.format(e))

    def rebuild(self, ids=None):  # pylint: disable=unused-argument
        for col in self.collections.values():
            col.compact()

    def flush(self):
        for col in self.collections.values():
            col.flush(_async=True)

    def close(self):
        self.flush()
        if self._local_mode:
            self._server.stop()
