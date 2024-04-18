# -*- coding: utf-8 -*-
from modelcache.similarity_evaluation.similarity_evaluation import SimilarityEvaluation
from modelcache.utils.lazy_import import LazyImport

exact_match = LazyImport(
    "exact_match", globals(), "modelcache.similarity_evaluation.exact_match"
)


def ExactMatchEvaluation():
    return exact_match.ExactMatchEvaluation()
