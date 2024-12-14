from typing import List

import numpy as np
import logging
from modelcache.manager.vector_data.base import VectorBase, VectorData
from modelcache.utils import import_chromadb, import_torch

import_torch()
import_chromadb()

import chromadb


class Chromadb(VectorBase):

    def __init__(
            self,
            persist_directory="./chromadb",
            top_k: int = 1,
    ):
        self.collection_name = "modelcache"
        self.top_k = top_k

        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = None

    def mul_add(self, datas: List[VectorData], model=None):
        collection_name_model = self.collection_name + '_' + model
        self._collection = self._client.get_or_create_collection(name=collection_name_model)

        data_array, id_array = map(list, zip(*((data.data.tolist(), str(data.id)) for data in datas)))
        self._collection.add(embeddings=data_array, ids=id_array)

    def search(self, data: np.ndarray, top_k: int = -1, model=None):
        collection_name_model = self.collection_name + '_' + model
        self._collection = self._client.get_or_create_collection(name=collection_name_model)

        if self._collection.count() == 0:
            return []
        if top_k == -1:
            top_k = self.top_k
        results = self._collection.query(
            query_embeddings=[data.tolist()],
            n_results=top_k,
            include=["distances"],
        )
        return list(zip(results["distances"][0], [int(x) for x in results["ids"][0]]))

    def rebuild(self, ids=None):
        pass

    def delete(self, ids, model=None):
        try:
            collection_name_model = self.collection_name + '_' + model
            self._collection = self._client.get_or_create_collection(name=collection_name_model)
            # 查询集合中实际存在的 ID
            ids_str = [str(x) for x in ids]
            existing_ids = set(self._collection.get(ids=ids_str).ids)

            # 删除存在的 ID
            if existing_ids:
                self._collection.delete(list(existing_ids))

            # 返回实际删除的条目数量
            return len(existing_ids)

        except Exception as e:
            logging.error('Error during deletion: {}'.format(e))
            raise ValueError(str(e))

    def rebuild_col(self, model):
        collection_name_model = self.collection_name + '_' + model

        # 检查集合是否存在，如果存在则删除
        collections = self._client.list_collections()
        if any(col.name == collection_name_model for col in collections):
            self._client.delete_collection(collection_name_model)
        else:
            return 'model collection not found, please check!'

        try:
            self._client.create_collection(collection_name_model)
        except Exception as e:
            logging.info(f'rebuild_collection: {e}')
            raise ValueError(str(e))

    def flush(self):
        # chroma无flush方法
        pass

    def close(self):
        # chroma无flush方法
        pass
