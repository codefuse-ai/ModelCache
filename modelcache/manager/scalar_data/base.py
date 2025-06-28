# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Union, Dict, List, Optional, Any
from enum import IntEnum
import numpy as np

from modelcache.utils import import_sql_client
from modelcache.utils.error import NotFoundError


class DataType(IntEnum):
    STR = 0
    IMAGE_BASE64 = 1
    IMAGE_URL = 2


@dataclass
class QuestionDep:
    """
    QuestionDep
    """

    name: str
    data: str
    dep_type: int = DataType.STR

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(
            name=d["name"],
            data=d["data"],
            dep_type=d["dep_type"]
        )


@dataclass
class Question:
    """
    Question
    """

    content: str
    deps: Optional[List[QuestionDep]] = None

    @classmethod
    def from_dict(cls, d: Dict):
        deps = []
        for dep in d["deps"]:
            deps.append(QuestionDep.from_dict(dep))
        return cls(d["content"], deps)


@dataclass
class Answer:
    """
    data_type:
        0: str
        1: base64 image
    """

    answer: Any
    answer_type: int = DataType.STR


@dataclass
class CacheData:
    """
    CacheData
    """

    question: Union[str, Question]
    answers: List[Answer]
    embedding_data: Optional[np.ndarray] = None

    def __init__(self, question, answers, embedding_data=None):
        self.question = question
        self.answers = []
        if isinstance(answers, (str, Answer)):
            answers = [answers]
        for data in answers:
            if isinstance(data, (list, tuple)):
                self.answers.append(Answer(*data))
            elif isinstance(data, Answer):
                self.answers.append(data)
            else:
                self.answers.append(Answer(answer=data))
        self.embedding_data = embedding_data


class CacheStorage(metaclass=ABCMeta):
    """
    BaseStorage for scalar data.
    """

    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def insert_query_resp(self, query_resp, **kwargs):
        pass

    @abstractmethod
    def get_data_by_id(self, key):
        pass

    @abstractmethod
    def mark_deleted(self, keys):
        pass

    @abstractmethod
    def model_deleted(self, model):
        pass

    @abstractmethod
    def clear_deleted_data(self):
        pass

    @abstractmethod
    def get_ids(self, deleted=True):
        pass

    @abstractmethod
    def count(self):
        pass

    @abstractmethod
    def flush(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def batch_insert(self, all_data: List[CacheData]):
        pass

    @abstractmethod
    def update_hit_count_by_id(self, primary_id):
        pass

    @staticmethod
    def get(name, **kwargs):
        if name in ["mysql", "oceanbase"]:
            from modelcache.manager.scalar_data.sql_storage import SQLStorage
            config = kwargs.get("config")
            import_sql_client(name)
            cache_base = SQLStorage(db_type=name, config=config)
        elif name == 'sqlite':
            SQL_URL = {"sqlite": "./sqlite.db"}
            from modelcache.manager.scalar_data.sql_storage_sqlite import SQLStorage
            sql_url = kwargs.get("sql_url", SQL_URL[name])
            cache_base = SQLStorage(db_type=name, url=sql_url)
        elif name == 'elasticsearch':
            from modelcache.manager.scalar_data.sql_storage_es import SQLStorage
            config = kwargs.get("config")
            cache_base = SQLStorage(db_type=name, config=config)
        else:
            raise NotFoundError("cache store", name)
        return cache_base

