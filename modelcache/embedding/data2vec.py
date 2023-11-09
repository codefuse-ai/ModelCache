# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from modelcache.embedding.base import BaseEmbedding


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class Data2VecAudio(BaseEmbedding):
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        model_dir = os.path.dirname(parent_dir)
        model = os.path.join(model_dir, 'model/text2vec-base-chinese/')

        try:
            self.__dimension = self.model.config.hidden_size
        except Exception:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model)
            self.__dimension = config.hidden_size

        self.tokenizer = BertTokenizer.from_pretrained(model, local_files_only=True)
        self.model = BertModel.from_pretrained(model, local_files_only=True)

    def to_embeddings(self, data, **_):
        encoded_input = self.tokenizer(data, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = sentence_embeddings.squeeze(0).detach().numpy()
        embedding_array = np.array(sentence_embeddings).astype("float32")
        return embedding_array

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
