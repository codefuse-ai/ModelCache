# -*- coding: utf-8 -*-
import numpy as np
import os
from modelcache.utils import import_fasttext
from modelcache.embedding.base import BaseEmbedding
import_fasttext()
import fasttext.util


class FastText(BaseEmbedding):
    def __init__(self, model: str = "en", dim: int = None):
        self.model_path = os.path.abspath(fasttext.util.download_model(model))
        self.ft = fasttext.load_model(self.model_path)

        if dim:
            fasttext.util.reduce_model(self.ft, dim)
        self.__dimension = self.ft.get_dimension()

    def to_embeddings(self, data, **_):
        assert isinstance(data, str), "Only allow string as input."
        emb = self.ft.get_sentence_vector(data)
        return np.array(emb).astype("float32")

    @property
    def dimension(self):
        return self.__dimension

