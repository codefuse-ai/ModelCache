# -*- coding: utf-8 -*-
import logging
import time
import requests
import pickle
import numpy as np
import cachetools
from abc import abstractmethod, ABCMeta
from typing import List, Any, Optional
from typing import Union, Callable
from modelcache.manager.scalar_data.base import CacheStorage,CacheData,DataType,Answer,Question
from modelcache.utils.error import CacheError, ParamError
from modelcache.manager.vector_data.base import VectorStorage, VectorData
from modelcache.manager.object_data.base import ObjectBase
from modelcache.manager.eviction.memory_cache import MemoryCacheEviction
from modelcache.utils.log import modelcache_log


class DataManager(metaclass=ABCMeta):
    """DataManager manage the cache data, including save and search"""

    @abstractmethod
    def save(self, question, answer, embedding_data, **kwargs):
        pass

    @abstractmethod
    def save_query_resp(self, query_resp_dict, **kwargs):
        pass

    @abstractmethod
    def import_data(self, questions: List[Any], answers: List[Any], embedding_datas: List[Any], model:Any):
        pass

    @abstractmethod
    def get_scalar_data(self, res_data, **kwargs) -> CacheData:
        pass

    @abstractmethod
    def update_hit_count(self, primary_id, **kwargs):
        pass

    def hit_cache_callback(self, res_data, **kwargs):
        pass

    @abstractmethod
    def search(self, embedding_data, **kwargs):
        pass

    @abstractmethod
    def delete(self, id_list, **kwargs):
        pass

    @abstractmethod
    def truncate(self, model_name):
        pass

    @abstractmethod
    def flush(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @staticmethod
    def get(
            cache_base: Union[CacheStorage, str] = None,
            vector_base: Union[VectorStorage, str] = None,
            object_base: Union[ObjectBase, str] = None,
            max_size: int = 3,
            clean_size: int = 1,
            eviction: str = "ARC",
            data_path: str = "data_map.txt",
            get_data_container: Callable = None,
            normalize: bool = True
    ):
        if not cache_base and not vector_base:
            return MapDataManager(data_path, max_size, get_data_container)

        if isinstance(cache_base, str):
            cache_base = CacheStorage.get(name=cache_base)
        if isinstance(vector_base, str):
            vector_base = VectorStorage.get(name=vector_base)
        if isinstance(object_base, str):
            object_base = ObjectBase.get(name=object_base)
        assert cache_base and vector_base
        return SSDataManager(cache_base, vector_base, object_base, max_size, clean_size,normalize, eviction)


class MapDataManager(DataManager):
    def __init__(self, data_path, max_size, get_data_container=None):
        if get_data_container is None:
            self.data = cachetools.LRUCache(max_size)
        else:
            self.data = get_data_container(max_size)
        self.data_path = data_path
        self.init()

    def init(self):
        try:
            with open(self.data_path, "rb") as f:
                self.data = pickle.load(f)
        except FileNotFoundError:
            return
        except PermissionError:
            raise CacheError(  # pylint: disable=W0707
                f"You don't have permission to access this file <{self.data_path}>."
            )

    def save(self, question, answer, embedding_data, **kwargs):
        if isinstance(question, Question):
            question = question.content
        self.data[embedding_data] = (question, answer, embedding_data)

    def save_query_resp(self, query_resp_dict, **kwargs):
        pass

    def import_data(
        self, questions: List[Any], answers: List[Any], embedding_datas: List[Any], model: Any
    ):
        if len(questions) != len(answers) or len(questions) != len(embedding_datas):
            raise ParamError("Make sure that all parameters have the same length")
        for i, embedding_data in enumerate(embedding_datas):
            self.data[embedding_data] = (questions[i], answers[i], embedding_datas[i])

    def get_scalar_data(self, res_data, **kwargs) -> CacheData:
        return CacheData(question=res_data[0], answers=res_data[1])

    def update_hit_count(self, primary_id, **kwargs):
        pass

    def search(self, embedding_data, **kwargs):
        try:
            return [self.data[embedding_data]]
        except KeyError:
            return []

    def delete(self, id_list, **kwargs):
        pass

    def truncate(self, model_name):
        pass

    def flush(self):
        try:
            with open(self.data_path, "wb") as f:
                pickle.dump(self.data, f)
        except PermissionError:
            modelcache_log.error(
                "You don't have permission to access this file %s.", self.data_path
            )

    def close(self):
        self.flush()


def normalize(vec):
    magnitude = np.linalg.norm(vec)
    normalized_v = vec / magnitude
    return normalized_v


class SSDataManager(DataManager):
    def __init__(
        self,
        s: CacheStorage,
        v: VectorStorage,
        o: Optional[ObjectBase],
        max_size,
        clean_size,
        normalize: bool,
        policy="LRU",
    ):
        self.max_size = max_size
        self.clean_size = clean_size
        self.s = s
        self.v = v
        self.o = o
        self.normalize = normalize

        # added
        self.eviction_base = MemoryCacheEviction(
            policy=policy,
            maxsize=max_size,
            clean_size=clean_size,
            on_evict=self._evict_ids)

    def save(self, questions: List[any], answers: List[any], embedding_datas: List[any], **kwargs):
        model = kwargs.pop("model", None)
        self.import_data(questions, answers, embedding_datas, model)


    def save_query_resp(self, query_resp_dict, **kwargs):
        save_query_start_time = time.time()
        self.s.insert_query_resp(query_resp_dict, **kwargs)
        save_query_delta_time = '{}s'.format(round(time.time() - save_query_start_time, 2))

    def _process_answer_data(self, answers: Union[Answer, List[Answer]]):
        if isinstance(answers, Answer):
            answers = [answers]
        new_ans = []
        for ans in answers:
            if ans.answer_type != DataType.STR:
                new_ans.append(Answer(self.o.put(ans.answer), ans.answer_type))
            else:
                new_ans.append(ans)
        return new_ans

    def _process_question_data(self, question: Union[str, Question]):
        if isinstance(question, Question):
            if question.deps is None:
                return question

            for dep in question.deps:
                if dep.dep_type == DataType.IMAGE_URL:
                    dep.dep_type.data = self.o.put(requests.get(dep.data).content)
            return question

        return Question(question)

    def import_data(
        self, questions: List[Any], answers: List[Answer], embedding_datas: List[Any], model: Any
    ):
        if len(questions) != len(answers) or len(questions) != len(embedding_datas):
            raise ParamError("Make sure that all parameters have the same length")
        cache_datas = []

        if self.normalize:
            embedding_datas = [
                normalize(embedding_data) for embedding_data in embedding_datas
            ]

        for i, embedding_data in enumerate(embedding_datas):
            if self.o is not None:
                ans = self._process_answer_data(answers[i])
            else:
                ans = answers[i]

            question = questions[i]
            embedding_data = embedding_data.astype("float32")
            cache_datas.append([ans, question, embedding_data, model])

        ids = self.s.batch_insert(cache_datas)
        datas = []
        for i, embedding_data in enumerate(embedding_datas):
            _id = ids[i]
            datas.append(VectorData(id=_id, data=embedding_data.astype("float32")))
            self.eviction_base.put([(_id, cache_datas[i])],model=model)
        self.v.mul_add(datas,model)

    def get_scalar_data(self, res_data, **kwargs) -> Optional[CacheData]:
        model = kwargs.pop("model")
        #Get Data from RAM Cache
        _id = res_data[1]
        cache_hit = self.eviction_base.get(_id, model=model)
        if cache_hit is not None:
            return cache_hit
        cache_data = self.s.get_data_by_id(res_data[1])
        if cache_data is None:
            return None
        return cache_data

    def update_hit_count(self, primary_id, **kwargs):
        self.s.update_hit_count_by_id(primary_id)

    def hit_cache_callback(self, res_data, **kwargs):
        self.eviction_base.get(res_data[1])

    def search(self, embedding_data, **kwargs):
        model = kwargs.pop("model", None)
        if self.normalize:
            embedding_data = normalize(embedding_data)
        top_k = kwargs.get("top_k", -1)
        return self.v.search(data=embedding_data, top_k=top_k, model=model)

    def delete(self, id_list, **kwargs):
        model = kwargs.pop("model")
        try:
            for id in id_list:
                self.eviction_base.get_cache(model).pop(id, None)  # Remove from in-memory LRU too
            v_delete_count = self.v.delete(ids=id_list, model=model)
        except Exception as e:
            return {'status': 'failed', 'milvus': 'delete milvus data failed, please check! e: {}'.format(e),
                    'mysql': 'unexecuted'}
        try:
            s_delete_count = self.s.mark_deleted(id_list)
        except Exception as e:
            return {'status': 'failed', 'milvus': 'success',
                    'mysql': 'delete mysql data failed, please check! e: {}'.format(e)}

        return {'status': 'success', 'milvus': 'delete_count: '+str(v_delete_count),
                'mysql': 'delete_count: '+str(s_delete_count)}

    def create_index(self, model, **kwargs):
        return self.v.create(model)

    def truncate(self, model):
        # drop memory cache data
        self.eviction_base.clear(model)

        # drop vector base data
        try:
            vector_resp = self.v.rebuild_col(model)
        except Exception as e:
            return {'status': 'failed', 'VectorDB': 'truncate VectorDB data failed, please check! e: {}'.format(e),
                    'ScalarDB': 'unexecuted'}
        if vector_resp:
            return {'status': 'failed', 'VectorDB': vector_resp, 'ScalarDB': 'unexecuted'}
        # drop scalar base data
        try:
            delete_count = self.s.model_deleted(model)
        except Exception as e:
            return {'status': 'failed', 'VectorDB': 'rebuild',
                    'ScalarDB': 'truncate scalar data failed, please check! e: {}'.format(e)}
        return {'status': 'success', 'VectorDB': 'rebuild', 'ScalarDB': 'delete_count: ' + str(delete_count)}

    # added
    def _evict_ids(self, ids, **kwargs):
        model = kwargs.get("model")
        if not ids or any(i is None for i in ids):
            modelcache_log.warning("Skipping eviction for invalid IDs: %s", ids)
            return

        if isinstance(ids,str):
            ids = [ids]

        for _id in ids:
            self.eviction_base.get_cache(model).pop(_id, None)

        try:
            self.s.mark_deleted(ids)
            modelcache_log.info("Evicted from scalar storage: %s", ids)
        except Exception as e:
            modelcache_log.error("Failed to delete from scalar storage: %s", str(e))

        try:
            self.v.delete(ids, model=model)
            modelcache_log.info("Evicted from vector storage (model=%s): %s", model, ids)
        except Exception as e:
            modelcache_log.error("Failed to delete from vector storage (model=%s): %s", model, str(e))

    def flush(self):
        self.s.flush()
        self.v.flush()

    def close(self):
        self.s.close()
        self.v.close()

