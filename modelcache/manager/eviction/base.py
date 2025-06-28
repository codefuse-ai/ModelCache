# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from typing import Any, List


class EvictionBase(metaclass=ABCMeta):
    """
    Eviction base.
    """

    @abstractmethod
    def put(self, objs: List[Any], model:str):
        pass

    @abstractmethod
    def get(self, obj: Any, model:str):
        pass

    @property
    @abstractmethod
    def policy(self) -> str:
        pass
