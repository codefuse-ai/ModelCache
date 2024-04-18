# -*- coding: utf-8 -*-
import numpy as np

from modelcache.utils import import_huggingface, import_torch
from modelcache.embedding.base import BaseEmbedding

import_torch()
import_huggingface()

import torch  # pylint: disable=C0413
from transformers import AutoTokenizer, AutoModel  # pylint: disable=C0413


class Huggingface(BaseEmbedding):
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = AutoModel.from_pretrained(model, local_files_only=True)
        self.model.eval()

        # self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = "[PAD]"
        try:
            self.__dimension = self.model.config.hidden_size
        except Exception:  # pylint: disable=W0703
            from transformers import AutoConfig  # pylint: disable=C0415

            config = AutoConfig.from_pretrained(model)
            self.__dimension = config.hidden_size

    def to_embeddings(self, data, **_):
        """Generate embedding given text input

        :param data: text in string.
        :type data: str

        :return: a text embedding in shape of (dim,).
        """
        if not isinstance(data, list):
            data = [data]
        inputs = self.tokenizer(
            data, padding=True, truncation=True, return_tensors="pt"
        )
        outs = self.model(**inputs).last_hidden_state
        emb = self.post_proc(outs, inputs).squeeze(0).detach().numpy()
        return np.array(emb).astype("float32")

    def post_proc(self, token_embeddings, inputs):
        attention_mask = inputs["attention_mask"]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sentence_embs = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sentence_embs

    @property
    def dimension(self):
        """Embedding dimension.

        :return: embedding dimension
        """
        return self.__dimension
