# -*- coding: utf-8 -*-
import numpy as np

from modelcache.embedding.base import BaseEmbedding
from modelcache.utils import (
    import_onnxruntime,
    import_huggingface_hub,
    import_huggingface,
)

import_huggingface()
import_onnxruntime()
import_huggingface_hub()

from transformers import AutoTokenizer, AutoConfig  # pylint: disable=C0413
import onnxruntime
from modelcache.utils.env_config import get_onnx_tokenizer_path, get_onnx_model


class Onnx(BaseEmbedding):

    def __init__(self, model="modelcache_open/paraphrase-albert-onnx"):
        # 本地加载
        onnx_tokenizer = get_onnx_tokenizer_path()
        self.tokenizer = AutoTokenizer.from_pretrained(onnx_tokenizer, local_files_only=True)
        # 本地加载
        onnx_model = get_onnx_model()
        self.ort_session = onnxruntime.InferenceSession(onnx_model)

        config = AutoConfig.from_pretrained(onnx_tokenizer, local_files_only=True)
        self.__dimension = config.hidden_size

    def to_embeddings(self, data, **_):
        """Generate embedding given text input.

        :param data: text in string.
        :type data: str

        :return: a text embedding in shape of (dim,).
        """
        encoded_text = self.tokenizer.encode_plus(data, padding="max_length")
        ort_inputs = {
            "input_ids": np.array(encoded_text["input_ids"]).reshape(1, -1),
            "attention_mask": np.array(encoded_text["attention_mask"]).reshape(1, -1),
            "token_type_ids": np.array(encoded_text["token_type_ids"]).reshape(1, -1),
        }

        ort_outputs = self.ort_session.run(None, ort_inputs)
        ort_feat = ort_outputs[0]
        emb = self.post_proc(ort_feat, ort_inputs["attention_mask"])
        return emb.flatten()

    def post_proc(self, token_embeddings, attention_mask):
        input_mask_expanded = (
            np.expand_dims(attention_mask, -1)
            .repeat(token_embeddings.shape[-1], -1)
            .astype(float)
        )
        sentence_embs = np.sum(token_embeddings * input_mask_expanded, 1) / np.maximum(
            input_mask_expanded.sum(1), 1e-9
        )
        return sentence_embs

    @property
    def dimension(self):
        """Embedding dimension.

        :return: embedding dimension
        """
        return self.__dimension
