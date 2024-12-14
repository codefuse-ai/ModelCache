from typing import List

import numpy as np
import logging
from modelcache_mm.manager.vector_data.base import VectorBase, VectorData
from modelcache_mm.utils import import_chromadb, import_torch
from modelcache_mm.utils.index_util import get_mm_index_name

import_torch()
import_chromadb()

import chromadb


class Chromadb(VectorBase):

    def __init__(
            self,
            persist_directory="./chromadb",
            top_k: int = 1,
    ):
        # self.collection_name = "modelcache"
        self.top_k = top_k

        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = None

    def create(self, model=None, mm_type=None):
        try:
            collection_name_model = get_mm_index_name(model, mm_type)
            # collection_name_model = self.collection_name + '_' + model
            self._client.get_or_create_collection(name=collection_name_model)
        except Exception as e:
            raise ValueError(str(e))

    def add(self, datas: List[VectorData], model=None, mm_type=None):
        collection_name_model = get_mm_index_name(model, mm_type)
        self._collection = self._client.get_or_create_collection(name=collection_name_model)

        data_array, id_array = map(list, zip(*((data.data.tolist(), str(data.id)) for data in datas)))
        self._collection.add(embeddings=data_array, ids=id_array)

    def search(self, data: np.ndarray, top_k: int = -1, model=None, mm_type='mm'):
        collection_name_model = get_mm_index_name(model, mm_type)
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

    def delete(self, ids, model=None, mm_type=None):
        try:
            collection_name_model = get_mm_index_name(model, mm_type)
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

    def rebuild_idx(self, model, mm_type=None):
        collection_name_model = get_mm_index_name(model, mm_type)

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

    def rebuild(self, ids=None):
        pass

    def flush(self):
        pass

    def close(self):
        pass
