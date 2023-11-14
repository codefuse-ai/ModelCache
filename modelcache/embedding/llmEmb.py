# -*- coding: utf-8 -*-
import numpy as np
from modelcache.embedding.base import BaseEmbedding
from transformers import AutoTokenizer
from transformers import AutoConfig


class LlmEmb2Vec(BaseEmbedding):
    def __init__(self):

        self.model_name = ''  # 13b-mft-embedding.npy
        model_path = ''  # .npy file storage path
        model_file = model_path + self.model_name  # .npy file
        config = AutoConfig.from_pretrained(model_path)
        dimension = config.hidden_size
        self.__dimension = dimension
        self.model = np.load(model_file)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    def to_embeddings(self, data, **_):
        """Generate embedding given text input

        :param data: text in string.
        :return: a text embedding in shape of (dim,).
        """
        input_ids = self.tokenizer.encode(data, add_special_tokens=True)
        embedding_array = self.model[input_ids].mean(axis=0)
        return embedding_array

    def post_proc(self, token_embeddings, inputs):
        pass

    @property
    def dimension(self):
        """Embedding dimension.
        :return: embedding dimension
        """
        return self.__dimension
