# -*- coding: utf-8 -*-
import logging
import time
import requests
import pickle
import numpy as np
import cachetools
from abc import abstractmethod, ABCMeta
from typing import List, Any, Optional, Union
from modelcache_mm.manager.scalar_data.base import (
    CacheStorage,
    CacheData,
    DataType,
    Answer,
    Question
)
from modelcache_mm.utils.error import CacheError, ParamError
from modelcache_mm.manager.vector_data.base import VectorBase, VectorData
from modelcache_mm.manager.object_data.base import ObjectBase
from modelcache_mm.utils.log import modelcache_log


class DataManager(metaclass=ABCMeta):
    """DataManager manage the cache data, including save and search"""

    @abstractmethod
    def save(self, text, image_url, image_id,  answer, embedding, **kwargs):
        pass

    @abstractmethod
    def save_query_resp(self, query_resp_dict, **kwargs):
        pass

    @abstractmethod
    def import_data(
        self, texts: List[Any], image_urls: List[Any], image_ids: List[Any], answers: List[Answer],
            embeddings: List[Any], model: Any, mm_type: Any
    ):
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

    def truncate(self, model_name):
        pass

    def flush(self):
        pass

    @abstractmethod
    def close(self):
        pass


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

    def save(self, text, image_url, image_id,  answer, embedding, **kwargs):
        pass

    def save_query_resp(self, query_resp_dict, **kwargs):
        pass

    def import_data(
        self, texts: List[Any], image_urls: List[Any], image_ids: List[Any], answers: List[Answer],
            embeddings: List[Any], model: Any, mm_type: Any
    ):
        pass

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
        v: VectorBase,
        o: Optional[ObjectBase],
        max_size,
        clean_size,
        policy="LRU",
    ):
        self.max_size = max_size
        self.clean_size = clean_size
        self.s = s
        self.v = v
        self.o = o

    def save(self, text, image_url, image_id,  answer, embedding, **kwargs):
        model = kwargs.pop("model", None)
        mm_type = kwargs.pop("mm_type", None)
        self.import_data([text], [image_url], [image_id], [answer], [embedding], model, mm_type)

    def save_query_resp(self, query_resp_dict, **kwargs):
        self.s.insert_query_resp(query_resp_dict, **kwargs)

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
        self, texts: List[Any], image_urls: List[Any], image_ids: List[Any], answers: List[Answer],
            embeddings: List[Any], model: Any, mm_type: Any
    ):
        if len(texts) != len(answers):
            raise ParamError("Make sure that all parameters have the same length")
        cache_datas = []

        embeddings = [
            normalize(text_embedding) for text_embedding in embeddings
        ]

        for i, embedding in enumerate(embeddings):
            if self.o is not None:
                ans = self._process_answer_data(answers[i])
            else:
                ans = answers[i]
            text = texts[i]
            image_url = image_urls[i]
            image_id = image_ids[i]
            cache_datas.append([ans, text, image_url, image_id, model])

        ids = self.s.batch_insert(cache_datas)
        self.v.add(
            [
                VectorData(id=ids[i], data=embedding)
                for i, embedding in enumerate(embeddings)
            ],
            model,
            mm_type
        )

    def get_scalar_data(self, res_data, **kwargs) -> Optional[CacheData]:
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
        mm_type = kwargs.pop("mm_type", None)
        embedding_data = normalize(embedding_data)
        top_k = kwargs.get("top_k", -1)
        try:
            search_result = self.v.search(data=embedding_data, top_k=top_k, model=model, mm_type=mm_type)
        except Exception as e:
            try:
                message = str(e)
                if "no such index" in message:
                    logging.info('no such index except，creating...')
                self.v.create(model, mm_type)
                search_result = self.v.search(data=embedding_data, top_k=top_k, model=model, mm_type=mm_type)
            except Exception as e:
                raise e
        return search_result

    def delete(self, id_list, **kwargs):
        model = kwargs.pop("model", None)
        try:
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

    def create_index(self, model, type, **kwargs):
        return self.v.create(model, type)

    def truncate(self, model_name):
        try:
            resp = self.v.rebuild_idx(model_name)
        except Exception as e:
            return {'status': 'failed', 'VectorDB': 'truncate VectorDB failed, please check! e: {}'.format(str(e)),
                    'ScalarDB': 'unexecuted'}

        if resp:
            return {'status': 'failed', 'VectorDB': resp, 'ScalarDB': 'unexecuted'}
        try:
            delete_count = self.s.model_deleted(model_name)
        except Exception as e:
            # return 'truncate milvus data failed, please check!'
            return {'status': 'failed', 'VectorDB': 'rebuild',
                    'ScalarDB': 'truncate scalardb data failed, please check! e: {}'.format(e)}

        return {'status': 'success', 'VectorDB': 'rebuild', 'ScalarDB': 'delete_count: ' + str(delete_count)}

    def flush(self):
        self.s.flush()
        self.v.flush()

    def close(self):
        self.s.close()
        self.v.close()


# if __name__ == '__main__':
#     from modelcache.manager import CacheBase, VectorBase, get_data_manager
#     data_manager = get_data_manager(CacheBase('mysql'), VectorBase('milvus', dimension=128))
#     data_manager.save('hello', 'hi', np.random.random((128,)).astype('float32'), model='gptcode_6b')
