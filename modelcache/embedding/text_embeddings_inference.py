# -*- coding: utf-8 -*-
import requests
import numpy as np
from modelcache.embedding.base import BaseEmbedding

class TextEmbeddingsInference(BaseEmbedding):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        self.headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
        }
        self.__dimension = self.to_embeddings('test').shape[0]
    def to_embeddings(self, data, **_):
        json_data = {
            'input': data,
            'model': self.model,
        }

        response = requests.post(self.base_url, headers=self.headers, json=json_data)
        embedding = response.json()['data'][0]['embedding']
        return np.array(embedding)

    @property
    def dimension(self):
        """Embedding dimension.

        :return: embedding dimension
        """
        return self.__dimension
