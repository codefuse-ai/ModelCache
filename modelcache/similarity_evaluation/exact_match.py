# -*- coding: utf-8 -*-
from typing import Tuple, Dict, Any
from modelcache.similarity_evaluation.similarity_evaluation import SimilarityEvaluation


class ExactMatchEvaluation(SimilarityEvaluation):

    def __init__(self):
        pass

    def evaluation(
        self, src_dict: Dict[str, Any], cache_dict: Dict[str, Any], **_
    ) -> float:
        return 1 if cache_dict["question"] == src_dict["question"] else 0

    def range(self) -> Tuple[float, float]:
        return 0, 1
