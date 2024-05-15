# -*- coding: utf-8 -*-
import numpy as np

from modelcache.utils import import_timm, import_torch, import_pillow
from modelcache.embedding.base import BaseEmbedding

import_torch()
import_timm()
import_pillow()

import torch
from timm.models import create_model
from timm.data import create_transform, resolve_data_config
from PIL import Image


class Timm(BaseEmbedding):
    def __init__(self, model: str = "resnet18", device: str = "default"):
        if device == "default":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model_name = model
        self.model = create_model(model_name=model, pretrained=True)
        self.model.eval()

        try:
            self.__dimension = self.model.embed_dim
        except Exception:
            self.__dimension = None

    def to_embeddings(self, data, skip_preprocess: bool = False, **_):
        if not skip_preprocess:
            data = self.preprocess(data)
        if data.dim() == 3:
            data = data.unsqueeze(0)
        feats = self.model.forward_features(data)
        emb = self.post_proc(feats).squeeze(0).detach().numpy()

        return np.array(emb).astype("float32")

    def post_proc(self, features):
        features = features.to("cpu")
        if features.dim() == 3:
            features = features[:, 0]
        if features.dim() == 4:
            global_pool = torch.nn.AdaptiveAvgPool2d(1)
            features = global_pool(features)
            features = features.flatten(1)
        assert features.dim() == 2, f"Invalid output dim {features.dim()}"
        return features

    def preprocess(self, image_path):
        data_cfg = resolve_data_config(self.model.pretrained_cfg)
        transform = create_transform(**data_cfg)

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image)
        return image_tensor

    @property
    def dimension(self):
        """Embedding dimension.
        :return: embedding dimension
        """
        if not self.__dimension:
            input_size = self.model.pretrained_cfg["input_size"]
            dummy_input = torch.rand((1,) + input_size)
            feats = self.to_embeddings(dummy_input, skip_preprocess=True)
            self.__dimension = feats.shape[0]
        return self.__dimension
