# -*- coding: utf-8 -*-
import time
import requests
import base64
import numpy as np
from modelcache import cache
from modelcache.utils.error import NotInitError
from modelcache.utils.time import time_cal


def adapt_insert(*args, **kwargs):
    chat_cache = kwargs.pop("cache_obj", cache)
    model = kwargs.pop("model", None)
    require_object_store = kwargs.pop("require_object_store", False)
    if require_object_store:
        assert chat_cache.data_manager.o, "Object store is required for adapter."
    if not chat_cache.has_init:
        raise NotInitError()
    cache_enable = chat_cache.cache_enable_func(*args, **kwargs)
    context = kwargs.pop("cache_context", {})
    embedding_data = None
    pre_embedding_data_dict = chat_cache.mm_insert_pre_embedding_func(
        kwargs,
        extra_param=context.get("pre_embedding_func", None),
        prompts=chat_cache.config.prompts,
    )

    print('pre_embedding_data_dict: {}'.format(pre_embedding_data_dict))
    chat_info = kwargs.pop("chat_info", [])
    llm_data = chat_info[-1]['answer']

    pre_embedding_text = '###'.join(pre_embedding_data_dict['text'])
    pre_embedding_image_url = pre_embedding_data_dict['imageUrl']
    pre_embedding_image_raw = pre_embedding_data_dict['imageRaw']
    pre_embedding_image_id = pre_embedding_data_dict.get('imageId', None)

    if pre_embedding_image_url and pre_embedding_image_raw:
        raise ValueError("Both pre_embedding_image_url and pre_embedding_image_raw cannot be non-empty at the same time.")

    if pre_embedding_image_url:
        url_start_time = time.time()
        response = requests.get(pre_embedding_image_url)
        image_data = response.content
        pre_embedding_image = base64.b64encode(image_data).decode('utf-8')
        get_image_time = '{}s'.format(round(time.time() - url_start_time, 2))
        print('get_image_time: {}'.format(get_image_time))
    elif pre_embedding_image_raw:
        pre_embedding_image = pre_embedding_image_raw
    else:
        pre_embedding_image = None
        if not pre_embedding_text:
            raise ValueError(
                "Both pre_embedding_image_url and pre_embedding_image_raw are empty. Please provide at least one.")

    data_dict = {'text': [pre_embedding_text], 'image': pre_embedding_image}
    embedding_data = None
    mm_type = None

    if cache_enable:
        embedding_data_resp = time_cal(
            chat_cache.embedding_func,
            func_name="image_embedding",
            report_func=chat_cache.report.embedding,
        )(data_dict)

        image_embeddings = embedding_data_resp['image_embedding']
        text_embeddings = embedding_data_resp['text_embeddings']

        print('image_embeddings: {}'.format(image_embeddings))
        print('text_embeddings: {}'.format(text_embeddings))

        if len(image_embeddings) > 0 and len(image_embeddings) > 0:
            image_embedding = np.array(image_embeddings[0])
            text_embedding = text_embeddings[0]
            embedding_data = np.concatenate((image_embedding, text_embedding))
            mm_type = 'mm'
        elif len(image_embeddings) > 0:
            image_embedding = np.array(image_embeddings[0])
            embedding_data = image_embedding
            mm_type = 'image'
        elif len(text_embeddings) > 0:
            text_embedding = np.array(text_embeddings[0])
            embedding_data = text_embedding
            mm_type = 'text'
        else:
            raise ValueError('maya embedding service return both empty list, please check!')

    print('embedding_data: {}'.format(embedding_data))
    chat_cache.data_manager.save(
        pre_embedding_text,
        pre_embedding_image_url,
        pre_embedding_image_id,
        llm_data,
        embedding_data,
        model=model,
        mm_type=mm_type,
        extra_param=context.get("mm_save_func", None)
    )
    return 'success'