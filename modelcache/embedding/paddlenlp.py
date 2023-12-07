# -*- coding: utf-8 -*-
"""
   Alipay.com Inc.
   Copyright (c) 2004-2023 All Rights Reserved.
   ------------------------------------------------------
   File Name : paddlenlp.py
   Author : fuhui.phe
   Create Time : 2023/12/7 20:43
   Description : description what the main function of this file
   Change Activity: 
        version0 : 2023/12/7 20:43 by fuhui.phe  init
"""
import numpy as np

from modelcache.embedding.base import BaseEmbedding
from modelcache.utils import import_paddlenlp, import_paddle

import_paddle()
import_paddlenlp()


import paddle  # pylint: disable=C0413
from paddlenlp.transformers import AutoModel, AutoTokenizer  # pylint: disable=C0413


class PaddleNLP(BaseEmbedding):
    def __init__(self, model: str = "ernie-3.0-medium-zh"):
        self.model = AutoModel.from_pretrained(model)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = "<pad>"
        self.__dimension = None

    def to_embeddings(self, data, **_):
        """Generate embedding given text input

        :param data: text in string.
        :type data: str

        :return: a text embedding in shape of (dim,).
        """
        if not isinstance(data, list):
            data = [data]
        inputs = self.tokenizer(
            data, padding=True, truncation=True, return_tensors="pd"
        )
        outs = self.model(**inputs)[0]
        emb = self.post_proc(outs, inputs).squeeze(0).detach().numpy()
        return np.array(emb).astype("float32")

    def post_proc(self, token_embeddings, inputs):
        attention_mask = paddle.ones(inputs["token_type_ids"].shape)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.shape).astype("float32")
        )
        sentence_embs = paddle.sum(
            token_embeddings * input_mask_expanded, 1
        ) / paddle.clip(input_mask_expanded.sum(1), min=1e-9)
        return sentence_embs

    @property
    def dimension(self):
        """Embedding dimension.

        :return: embedding dimension
        """
        if not self.__dimension:
            self.__dimension = len(self.to_embeddings("foo"))
        return self.__dimension
