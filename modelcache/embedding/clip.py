# -*- coding: utf-8 -*-
import os
import torch
from modelcache.embedding.base import BaseEmbedding
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from modelscope.preprocessors.image import load_image


# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class ClipAudio(BaseEmbedding):
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # parent_dir = os.path.dirname(current_dir)
        # model_dir = os.path.dirname(parent_dir)
        # model = os.path.join(model_dir, 'model/text2vec-base-chinese/')

        self.clip_pipeline = pipeline(task=Tasks.multi_modal_embedding,
                                      model='damo/multi-modal_clip-vit-base-patch16_zh', model_revision='v1.0.1')

        self.__dimension = 1024

    def to_embeddings(self, data_dict, **_):
        text_list = data_dict['text']
        image_data = data_dict['image']

        img_data = None
        txt_data = None

        if image_data:
            input_img = load_image(image_data)
            # 2D Tensor, [图片数, 特征维度]
            img_embedding = self.clip_pipeline.forward({'img': input_img})['img_embedding'].tolist()[0] if input_img else []
            print('img_embedding: {}'.format(img_embedding))
        else:
            raise ValueError('image_data is None, please check!')

        if text_list and len(text_list) > 0:
            # 2D Tensor, [文本数, 特征维度]
            text_embedding = self.clip_pipeline.forward({'text': text_list})['text_embedding'].tolist()[0] if text_list else []
            print('text_embedding: {}'.format(text_embedding))
        else:
            raise ValueError('text_list is None, please check!')

        return {'image_embedding': img_embedding, 'text_embeddings': text_embedding}

        # return {'image_embedding': img_feats, 'text_embeddings': txt_feats}
        # input_texts = ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]
        # input_img = load_image(
        #     'https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg')

        # img_embedding = self.clip_pipeline.forward({'img': input_img})['img_embedding']  # 2D Tensor, [图片数, 特征维度]
        # print('img_embedding: {}'.format(img_embedding))
        # text_embedding = self.clip_pipeline.forward({'text': input_texts})['text_embedding']  # 2D Tensor, [文本数, 特征维度]


        # return embedding_array

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


# if __name__ == '__main__':
#     clip_vec = ClipAudio()
#     text_list = ['hello', '你好']
#     text = ['###'.join(text_list)]
#     image = 'https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg'
#     data_dict = {'text': text, 'image': image}
#     resp = clip_vec.to_embeddings(data_dict)
#     print('resp: {}'.format(resp))
