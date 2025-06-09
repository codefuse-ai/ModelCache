# -*- coding: utf-8 -*-
from modelcache.embedding.base import BaseEmbedding
from sentence_transformers import SentenceTransformer

class Huggingface(BaseEmbedding):
    def __init__(self, model: str):
        self.model = SentenceTransformer(model)
        try:
            self.__dimension = self.model.config.hidden_size
        except Exception:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model)
            self.__dimension = config.hidden_size

    def to_embeddings(self, data: str, **_):
        """Generate embedding given text input

        :param data: text in string.
        :type data: str

        :return: a text embedding in shape of (dim,).
        """

        if not data:
            raise ValueError("No data provided for embedding.")
        embeddings = self.model.encode(data)
        return embeddings[0] if len(data) == 1 else embeddings

    @property
    def dimension(self):
        """Embedding dimension.

        :return: embedding dimension
        """
        return self.__dimension
