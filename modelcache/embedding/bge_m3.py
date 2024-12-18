# -*- coding: utf-8 -*-
import numpy as np
from modelcache.embedding.base import BaseEmbedding
from transformers import AutoTokenizer, AutoModel
from FlagEmbedding import BGEM3FlagModel

class BgeM3Embedding(BaseEmbedding):
    def __init__(self, model_path: str = "model/bge-m3"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        
        self.bge_model = BGEM3FlagModel(model_name_or_path=model_path, 
                                        model=self.model, 
                                        tokenizer=self.tokenizer, 
                                        use_fp16=False)
        
        self.__dimension = 768 

    def to_embeddings(self, data, **_):
        if not isinstance(data, list):
            data = [data]
        
        embeddings = self.bge_model.encode(data, batch_size=12, max_length=8192)['dense_vecs']
        return np.array(embeddings).astype("float32")

    @property
    def dimension(self):
        return self.__dimension