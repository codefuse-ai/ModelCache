# -*- coding: utf-8 -*-
"""
   Alipay.com Inc.
   Copyright (c) 2004-2023 All Rights Reserved.
   ------------------------------------------------------
   File Name : clip_demo.py
   Author : fuhui.phe
   Create Time : 2024/5/7 11:58
   Description : description what the main function of this file
   Change Activity: 
        version0 : 2024/5/7 11:58 by fuhui.phe  init
"""
import torch
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from modelscope.preprocessors.image import load_image


pipeline = pipeline(task=Tasks.multi_modal_embedding,
                    model='damo/multi-modal_clip-vit-base-patch16_zh', model_revision='v1.0.1')

# pipeline = pipeline(task=Tasks.multi_modal_embedding,
#     model='/Users/penghongen/PycharmProjects/CodeFuse-ModelCache/model/clip_zh', model_revision='v1.0.1')

# pipeline = pipeline(task=Tasks.multi_modal_embedding, model='/Users/penghongen/PycharmProjects/CodeFuse-ModelCache/model/clip_zh')


input_img = load_image('https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg') # 支持皮卡丘示例图片路径/本地图片 返回PIL.Image


input_texts = ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]

# 支持一张图片(PIL.Image)或多张图片(List[PIL.Image])输入，输出归一化特征向量
img_embedding = pipeline.forward({'img': input_img})['img_embedding'] # 2D Tensor, [图片数, 特征维度]
print('img_embedding: {}'.format(img_embedding))

# 支持一条文本(str)或多条文本(List[str])输入，输出归一化特征向量
text_embedding = pipeline.forward({'text': input_texts})['text_embedding'] # 2D Tensor, [文本数, 特征维度]

# 计算图文相似度
with torch.no_grad():
    # 计算内积得到logit，考虑模型temperature
    logits_per_image = (img_embedding / pipeline.model.temperature) @ text_embedding.t()
    # 根据logit计算概率分布
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("图文匹配概率:", probs)


