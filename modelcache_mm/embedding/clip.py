# -*- coding: utf-8 -*-
import torch
from modelcache.embedding.base import BaseEmbedding
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from modelscope.preprocessors.image import load_image


class ClipAudio(BaseEmbedding):
    def __init__(self, model: str = 'damo/multi-modal_clip-vit-base-patch16_zh'):
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip_pipeline = pipeline(task=Tasks.multi_modal_embedding,
                                      model=model, model_revision='v1.0.1')
        self.__dimension = 1024

    def to_embeddings(self, data_dict, **_):
        text_list = data_dict['text']
        image_data = data_dict['image']

        img_data = None
        txt_data = None

        if image_data:
            input_img = load_image(image_data)
            img_embedding = self.clip_pipeline.forward({'img': input_img})['img_embedding'].tolist()[0] if input_img else []
        else:
            raise ValueError('image_data is None, please check!')

        if text_list and len(text_list) > 0:
            text_embedding = self.clip_pipeline.forward({'text': text_list})['text_embedding'].tolist()[0] if text_list else []
        else:
            raise ValueError('text_list is None, please check!')

        return {'image_embedding': img_embedding, 'text_embeddings': text_embedding}

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
