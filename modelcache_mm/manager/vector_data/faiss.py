# -*- coding: utf-8 -*-
import os
from typing import List
import numpy as np
from modelcache_mm.manager.vector_data.base import VectorBase, VectorData
from modelcache_mm.utils import import_faiss
import_faiss()
import faiss  # pylint: disable=C0413


class Faiss(VectorBase):
    def __init__(self,
                 index_file_path,
                 dimension: int = 0,
                 top_k: int = 1
                 ):
        self._dimension = dimension
        self._index_file_path = index_file_path
        self._index = faiss.index_factory(self._dimension, "IDMap,Flat", faiss.METRIC_L2)
        self._top_k = top_k
        if os.path.isfile(index_file_path):
            self._index = faiss.read_index(index_file_path)

    def add(self, datas: List[VectorData], model=None, mm_type=None):
        data_array, id_array = map(list, zip(*((data.data, data.id) for data in datas)))
        np_data = np.array(data_array).astype("float32")
        ids = np.array(id_array)
        print('insert_np_data: {}'.format(np_data))
        print('insert_np_data: {}'.format(np_data.shape))
        self._index.add_with_ids(np_data, ids)

    def search(self, data: np.ndarray, top_k: int, model, mm_type='mm'):
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

    def create(self, model=None, mm_type=None):
        pass
        # collection_name_model = get_mm_index_name(model, mm_type)
        # try:
        #     index_prefix = get_mm_index_prefix(model, mm_type)
        #     self.create_index(collection_name_model, mm_type, index_prefix)
        # except Exception as e:
        #     raise ValueError(str(e))
        # return 'success'

    def flush(self):
        faiss.write_index(self._index, self._index_file_path)

    def close(self):
        self.flush()

    def rebuild_idx(self, model):
        pass

    def count(self):
        return self._index.ntotal
