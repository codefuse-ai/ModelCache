# -*- coding: utf-8 -*-
import os
from typing import List
import numpy as np
from modelcache.manager.vector_data.base import VectorBase, VectorData
from modelcache.utils import import_faiss
import_faiss()
import faiss  # pylint: disable=C0413


class Faiss(VectorBase):
    def __init__(self, index_file_path, dimension, top_k):
        self._index_file_path = index_file_path
        self._dimension = dimension
        self._index = faiss.index_factory(self._dimension, "IDMap,Flat", faiss.METRIC_L2)
        self._top_k = top_k
        self.index_file_path = index_file_path
        if os.path.isfile(self.index_file_path):
            self._index = faiss.read_index(self.index_file_path)

    def mul_add(self, datas: List[VectorData], model=None):
        data_array, id_array = map(list, zip(*((data.data, data.id) for data in datas)))
        np_data = np.array(data_array).astype("float32")
        ids = np.array(id_array)
        self._index.add_with_ids(np_data, ids)

    def search(self, data: np.ndarray, top_k: int = -1, model=None):
        if self._index.ntotal == 0:
            return None
        if top_k == -1:
            top_k = self._top_k
        np_data = np.array(data).astype("float32").reshape(1, -1)
        dist, ids = self._index.search(np_data, top_k)
        ids = [int(i) for i in ids[0]]
        return list(zip(dist[0], ids))

    def rebuild_col(self, ids=None):
        try:
            self._index.reset()
        except Exception as e:
            return f"An error occurred during index rebuild: {e}"

    def rebuild(self, ids=None):
        return True

    def delete(self, ids):
        ids_to_remove = np.array(ids)
        self._index.remove_ids(faiss.IDSelectorBatch(ids_to_remove.size, faiss.swig_ptr(ids_to_remove)))

    def flush(self):
        faiss.write_index(self._index, self._index_file_path)

    def close(self):
        self.flush()

    def count(self):
        return self._index.ntotal

    def create(self, model=None):
        if os.path.isfile(self.index_file_path):
            self._index = faiss.read_index(self.index_file_path)

        return 'create_success'
