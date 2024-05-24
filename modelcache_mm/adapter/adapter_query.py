# -*- coding: utf-8 -*-
import logging
import numpy as np
from modelcache_mm import cache
from modelcache_mm.utils.error import NotInitError
from modelcache_mm.utils.error import MultiTypeError
from modelcache_mm.utils.time import time_cal


def adapt_query(cache_data_convert, *args, **kwargs):
    chat_cache = kwargs.pop("cache_obj", cache)
    scope = kwargs.pop("scope", None)
    model = scope['model']
    if not chat_cache.has_init:
        raise NotInitError()

    cache_enable = chat_cache.cache_enable_func(*args, **kwargs)
    context = kwargs.pop("cache_context", {})
    cache_factor = kwargs.pop("cache_factor", 1.0)

    pre_embedding_data_dict = chat_cache.query_pre_embedding_func(
        kwargs,
        extra_param=context.get("pre_embedding_func", None),
        prompts=chat_cache.config.prompts,
    )

    pre_embedding_text = '###'.join(pre_embedding_data_dict['text'])
    pre_embedding_image_raw = pre_embedding_data_dict['imageRaw']
    pre_embedding_image_url = pre_embedding_data_dict['imageUrl']
    pre_multi_type = pre_embedding_data_dict['multiType']

    # 判断逻辑
    if pre_multi_type == 'IMG_TEXT':
        if pre_embedding_image_raw and pre_embedding_image_url:
            raise ValueError(
                "Both pre_embedding_imageUrl and pre_embedding_imageRaw cannot be non-empty at the same time.")
        if pre_embedding_image_url:
            pre_embedding_image = pre_embedding_image_url
        elif pre_embedding_image_raw:
            pre_embedding_image = pre_embedding_image_raw
        else:
            raise ValueError(
                "Both pre_embedding_imageUrl and pre_embedding_imageRaw are empty. Please provide at least one.")
        data_dict = {'text': [pre_embedding_text], 'image': pre_embedding_image}
    elif pre_multi_type == 'TEXT':
        data_dict = {'text': [pre_embedding_text], 'image': None}
    else:
        raise MultiTypeError

    # embedding_data = None
    # mm_type = None
    if pre_multi_type == 'IMG_TEXT':
        embedding_data_resp = time_cal(
            chat_cache.embedding_func,
            func_name="mm_embedding",
            report_func=chat_cache.report.embedding,
        )(data_dict)
    else:
        embedding_data_resp = time_cal(
            chat_cache.embedding_func,
            func_name="mm_embedding",
            report_func=chat_cache.report.embedding,
        )(data_dict)
    image_embeddings = embedding_data_resp['image_embedding']
    text_embeddings = embedding_data_resp['text_embeddings']

    if len(image_embeddings) > 0 and len(image_embeddings) > 0:
        embedding_data = np.concatenate((image_embeddings, text_embeddings))
        # mm_type = 'mm'
    elif len(image_embeddings) > 0:
        image_embedding = np.array(image_embeddings[0])
        embedding_data = image_embedding
        # mm_type = 'image'
    elif len(text_embeddings) > 0:
        text_embedding = np.array(text_embeddings[0])
        embedding_data = text_embedding
        # mm_type = 'text'
    else:
        raise ValueError('maya embedding service return both empty list, please check!')

    if cache_enable:
        cache_data_list = time_cal(
            chat_cache.data_manager.search,
            func_name="vector_search",
            report_func=chat_cache.report.search,
        )(
            embedding_data,
            extra_param=context.get("search_func", None),
            top_k=kwargs.pop("top_k", -1),
            model=model,
            mm_type=pre_multi_type,
        )

        cache_answers = []
        cache_questions = []
        cache_image_urls = []
        cache_image_ids = []
        cache_ids = []
        similarity_threshold = chat_cache.config.similarity_threshold
        similarity_threshold_long = chat_cache.config.similarity_threshold_long

        min_rank, max_rank = chat_cache.similarity_evaluation.range()
        rank_threshold = (max_rank - min_rank) * similarity_threshold * cache_factor
        rank_threshold_long = (max_rank - min_rank) * similarity_threshold_long * cache_factor
        rank_threshold = (
            max_rank
            if rank_threshold > max_rank
            else min_rank
            if rank_threshold < min_rank
            else rank_threshold
        )
        rank_threshold_long = (
            max_rank
            if rank_threshold_long > max_rank
            else min_rank
            if rank_threshold_long < min_rank
            else rank_threshold_long
        )

        if cache_data_list is None or len(cache_data_list) == 0:
            rank_pre = -1.0
        else:
            cache_data_dict = {'search_result': cache_data_list[0]}
            rank_pre = chat_cache.similarity_evaluation.evaluation(
                None,
                cache_data_dict,
                extra_param=context.get("evaluation_func", None),
            )
        if rank_pre < rank_threshold:
            return

        for cache_data in cache_data_list:
            primary_id = cache_data[1]
            ret = chat_cache.data_manager.get_scalar_data(
                cache_data, extra_param=context.get("get_scalar_data", None)
            )
            if ret is None:
                continue

            if "deps" in context and hasattr(ret.question, "deps"):
                eval_query_data = {
                    "question": context["deps"][0]["data"],
                    "embedding": None
                }
                eval_cache_data = {
                    "question": ret.question.deps[0].data,
                    "answer": ret.answers[0].answer,
                    "search_result": cache_data,
                    "embedding": None,
                }
            else:
                eval_query_data = {
                    "question": pre_embedding_text,
                    "embedding": embedding_data,
                }

                eval_cache_data = {
                    "question": ret[0],
                    "image_url": ret[1],
                    "image_raw": ret[2],
                    "answer": ret[3],
                    "search_result": cache_data,
                    "embedding": None
                }
            rank = chat_cache.similarity_evaluation.evaluation(
                eval_query_data,
                eval_cache_data,
                extra_param=context.get("evaluation_func", None),
            )

            if len(pre_embedding_text) <= 50:
                if rank_threshold <= rank:
                    cache_answers.append((rank, ret[3]))
                    cache_image_urls.append((rank, ret[1]))
                    cache_image_ids.append((rank, ret[2]))
                    cache_questions.append((rank, ret[0]))
                    cache_ids.append((rank, primary_id))
            else:
                if rank_threshold_long <= rank:
                    cache_answers.append((rank, ret[3]))
                    cache_image_urls.append((rank, ret[1]))
                    cache_image_ids.append((rank, ret[2]))
                    cache_questions.append((rank, ret[0]))
                    cache_ids.append((rank, primary_id))

        cache_answers = sorted(cache_answers, key=lambda x: x[0], reverse=True)
        cache_image_urls = sorted(cache_image_urls, key=lambda x: x[0], reverse=True)
        cache_image_ids = sorted(cache_image_ids, key=lambda x: x[0], reverse=True)
        cache_questions = sorted(cache_questions, key=lambda x: x[0], reverse=True)
        cache_ids = sorted(cache_ids, key=lambda x: x[0], reverse=True)

        if len(cache_answers) != 0:
            return_message = chat_cache.post_process_messages_func(
                [t[1] for t in cache_answers]
            )
            return_image_url = chat_cache.post_process_messages_func(
                [t[1] for t in cache_image_urls]
            )
            return_image_id = chat_cache.post_process_messages_func(
                [t[1] for t in cache_image_ids]
            )
            return_query = chat_cache.post_process_messages_func(
                [t[1] for t in cache_questions]
            )
            return_id = chat_cache.post_process_messages_func(
                [t[1] for t in cache_ids]
            )
            # 更新命中次数
            try:
                chat_cache.data_manager.update_hit_count(return_id)
            except Exception:
                logging.warning('update_hit_count except, please check!')

            chat_cache.report.hint_cache()
            return_query_dict = {"image_url": return_image_url, "image_id": return_image_id, "question": return_query}
            return cache_data_convert(return_message, return_query_dict)
