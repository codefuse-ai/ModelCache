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
            memory_cache_policy: str = "ARC",
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
        return SSDataManager(cache_base, vector_base, object_base, max_size, clean_size,normalize, memory_cache_policy)


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
        self.s = s  # SQL storage
        self.v = v  # Vector storage
        self.o = o  # Object storage (optional)
        self.normalize = normalize

        # Initialize memory cache with specified eviction policy
        self.eviction_base = MemoryCacheEviction(
            policy=policy,
            maxsize=max_size,
            clean_size=clean_size)

    def save(self, questions: List[any], answers: List[any], embedding_datas: List[any], **kwargs):
        """Save multiple questions, answers, and embeddings to storage."""
        model = kwargs.pop("model", None)
        self.import_data(questions, answers, embedding_datas, model)

    def save_query_resp(self, query_resp_dict, **kwargs):
        """Save query response log to SQL storage for analytics."""
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
        """
        Add multiple cache entries into all storage backends.

        Coordinates data insertion across SQL, vector, and object storage,
        with memory cache population and optional vector normalization.
        """
        if len(questions) != len(answers) or len(questions) != len(embedding_datas):
            raise ParamError("Make sure that all parameters have the same length")
        cache_datas = []

        # Normalize embedding vectors if configured
        if self.normalize:
            embedding_datas = [
                normalize(embedding_data) for embedding_data in embedding_datas
            ]

        for embedding_data, answer, question in zip(embedding_datas,answers,questions):
            if self.o is not None:
                answer = self._process_answer_data(answer)

            embedding_data = embedding_data.astype("float32")
            cache_datas.append([answer, question, embedding_data, model])

        # Insert into SQL storage and get generated IDs
        ids = self.s.batch_insert(cache_datas)

        # Prepare vector data and populate memory cache
        datas = []
        for _id,embedding_data,cache_data in zip(ids,embedding_datas,cache_datas):
            datas.append(VectorData(id=_id, data=embedding_data.astype("float32")))
            self.eviction_base.put([(_id, cache_data)],model=model)
        self.v.mul_add(datas,model)

    def get_scalar_data(self, res_data, **kwargs) -> Optional[CacheData]:
        """
        Retrieve scalar data with multi-level caching strategy.

        First checks memory cache, then falls back to SQL storage.
        """
        model = kwargs.pop("model")
        _id = res_data[1]

        # Try to get from memory cache first (fastest)
        cache_hit = self.eviction_base.get(_id, model=model)
        if cache_hit is not None:
            return cache_hit
        cache_data = self.s.get_data_by_id(_id)
        if cache_data is None:
            return None
        self.eviction_base.put([(_id, cache_data)], model=model)
        return cache_data

    def update_hit_count(self, primary_id, **kwargs):
        """Update hit count statistics in SQL storage."""
        self.s.update_hit_count_by_id(primary_id)

    def hit_cache_callback(self, res_data, **kwargs):
        """Callback executed on cache hit to update memory cache."""
        self.eviction_base.get(res_data[1])

    def search(self, embedding_data, **kwargs):
        """
        Search for similar vectors in vector storage.

        Applies normalization if configured and delegates to vector backend.
        """
        model = kwargs.pop("model", None)
        if self.normalize:
            embedding_data = normalize(embedding_data)
        top_k = kwargs.get("top_k", -1)
        return self.v.search(data=embedding_data, top_k=top_k, model=model)

    def delete(self, id_list, **kwargs):
        """
        Delete cache entries from all storage backends.

        Removes from memory cache, vector storage, and marks as deleted in SQL.
        Returns detailed status of deletion operations.
        """
        model = kwargs.pop("model")
        try:
            # Remove from memory cache
            for id in id_list:
                self.eviction_base.get_cache(model).pop(id, None)
            # Delete from vector storage
            v_delete_count = self.v.delete(ids=id_list, model=model)
        except Exception as e:
            return {'status': 'failed', 'milvus': 'delete milvus data failed, please check! e: {}'.format(e),
                    'mysql': 'unexecuted'}
        try:
            # Mark as deleted in SQL storage
            s_delete_count = self.s.mark_deleted(id_list)
        except Exception as e:
            return {'status': 'failed', 'milvus': 'success',
                    'mysql': 'delete mysql data failed, please check! e: {}'.format(e)}

        return {'status': 'success', 'milvus': 'delete_count: '+str(v_delete_count),
                'mysql': 'delete_count: '+str(s_delete_count)}

    def create_index(self, model, **kwargs):
        """Create vector index for a specific model."""
        return self.v.create(model)

    def truncate(self, model):
        """
        Truncate all data for a specific model across all storage backends.

        Clears memory cache, rebuilds vector storage, and deletes SQL data.
        Returns detailed status of truncation operations.
        """
        # Clear memory cache data
        self.eviction_base.clear(model)

        # Rebuild vector storage (drops and recreates collection)
        try:
            vector_resp = self.v.rebuild_col(model)
        except Exception as e:
            return {'status': 'failed', 'VectorDB': 'truncate VectorDB data failed, please check! e: {}'.format(e),
                    'ScalarDB': 'unexecuted'}
        if vector_resp:
            return {'status': 'failed', 'VectorDB': vector_resp, 'ScalarDB': 'unexecuted'}

        # Delete scalar data from SQL storage
        try:
            delete_count = self.s.model_deleted(model)
        except Exception as e:
            return {'status': 'failed', 'VectorDB': 'rebuild',
                    'ScalarDB': 'truncate scalar data failed, please check! e: {}'.format(e)}
        return {'status': 'success', 'VectorDB': 'rebuild', 'ScalarDB': 'delete_count: ' + str(delete_count)}

    def flush(self):
        """Flush all storage backends to ensure data persistence."""
        self.s.flush()
        self.v.flush()

    def close(self):
        """Close all storage connections and release resources."""
        self.s.close()
        self.v.close()

