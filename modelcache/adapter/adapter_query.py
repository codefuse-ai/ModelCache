# -*- coding: utf-8 -*-
import asyncio
import logging
from modelcache.embedding import MetricType
from modelcache.utils.time import time_cal
from FlagEmbedding import FlagReranker

USE_RERANKER = False  # 如果为 True 则启用 reranker，否则使用原有逻辑

async def adapt_query(cache_data_convert, *args, **kwargs):
    # Extract query parameters
    chat_cache = kwargs.pop("cache_obj")
    scope = kwargs.pop("scope")
    model = scope['model']
    context = kwargs.pop("cache_context", {})
    cache_factor = kwargs.pop("cache_factor", 1.0)

    # Preprocess query for embedding generation
    pre_embedding_data = chat_cache.query_pre_embedding_func(
        kwargs,
        extra_param=context.get("pre_embedding_func", None),
        prompts=chat_cache.prompts,
    )

    # Generate embedding with performance monitoring
    embedding_data = await time_cal(
        chat_cache.embedding_func,
        func_name="embedding",
        report_func=chat_cache.report.embedding,
        cache_obj=chat_cache
    )(pre_embedding_data)

    search_time_cal = time_cal(
        chat_cache.data_manager.search,
        func_name="vector_search",
        report_func=chat_cache.report.search,
        cache_obj=chat_cache
    )
    cache_data_list = await asyncio.to_thread(
        search_time_cal,
        embedding_data,
        extra_param=context.get("search_func", None),
        top_k=kwargs.pop("top_k", -1),
        model=model
    )

    # Initialize result containers
    cache_answers = []
    cache_questions = []
    cache_ids = []
    cosine_similarity = None

    # Similarity evaluation based on metric type
    if chat_cache.similarity_metric_type == MetricType.COSINE:
        cosine_similarity = cache_data_list[0][0]
        # This code uses the built-in cosine similarity evaluation in milvus
        if cosine_similarity < chat_cache.similarity_threshold:
            return None  # No suitable match found

    elif chat_cache.similarity_metric_type == MetricType.L2:
        # this is the code that uses L2 for similarity evaluation
        similarity_threshold = chat_cache.similarity_threshold
        similarity_threshold_long = chat_cache.similarity_threshold_long

        min_rank, max_rank = chat_cache.similarity_evaluation.range()
        rank_threshold = (max_rank - min_rank) * similarity_threshold * cache_factor
        rank_threshold_long = (max_rank - min_rank) * similarity_threshold_long * cache_factor

        # Clamp thresholds to valid range
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

        # Evaluate similarity score
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
            return None  # Similarity too low
    else:
        raise ValueError(
            f"Unsupported similarity metric type: {chat_cache.similarity_metric_type}"
        )

    # Process search results with optional reranking
    if USE_RERANKER:
        reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=False)
        for cache_data in cache_data_list:
            primary_id = cache_data[1]
            ret = await asyncio.to_thread(
                chat_cache.data_manager.get_scalar_data,
                cache_data, extra_param=context.get("get_scalar_data", None), model=model
            )
            if ret is None:
                continue

            rank = reranker.compute_score([pre_embedding_data, ret[0]], normalize=True)[0]

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
                    "question": pre_embedding_data,
                    "embedding": embedding_data,
                }
                eval_cache_data = {
                    "question": ret[0],
                    "answer": ret[1],
                    "search_result": cache_data,
                    "embedding": None
                }

            if len(pre_embedding_data) <= 256:
                if rank_threshold <= rank:
                    cache_answers.append((rank, ret[0]))
                    cache_questions.append((rank, ret[1]))
                    cache_ids.append((rank, primary_id))
            else:
                if rank_threshold_long <= rank:
                    cache_answers.append((rank, ret[0]))
                    cache_questions.append((rank, ret[1]))
                    cache_ids.append((rank, primary_id))
    else:
        # Original logic without reranking
        for cache_data in cache_data_list:
            primary_id = cache_data[1]
            # Retrieve full cache entry data
            ret = await asyncio.to_thread(
                chat_cache.data_manager.get_scalar_data,
                cache_data, extra_param=context.get("get_scalar_data", None), model=model
            )
            if ret is None:
                continue

            if chat_cache.similarity_metric_type == MetricType.COSINE:
                assert cosine_similarity is not None, "cosine_similarity should not be None"
                cache_answers.append((cosine_similarity, ret[0]))
                cache_questions.append((cosine_similarity, ret[1]))
                cache_ids.append((cosine_similarity, primary_id))

            elif chat_cache.similarity_metric_type == MetricType.L2:
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
                        "question": pre_embedding_data,
                        "embedding": embedding_data,
                    }
                    eval_cache_data = {
                        "question": ret[0],
                        "answer": ret[1],
                        "search_result": cache_data,
                        "embedding": None
                    }

                # Evaluate similarity for this specific result
                rank = chat_cache.similarity_evaluation.evaluation(
                    eval_query_data,
                    eval_cache_data,
                    extra_param=context.get("evaluation_func", None),
                )

                if len(pre_embedding_data) <= 256:
                    if rank_threshold <= rank:
                        cache_answers.append((rank, ret[0]))
                        cache_questions.append((rank, ret[1]))
                        cache_ids.append((rank, primary_id))
                else:
                    if rank_threshold_long <= rank:
                        cache_answers.append((rank, ret[0]))
                        cache_questions.append((rank, ret[1]))
                        cache_ids.append((rank, primary_id))
            else:
                raise ValueError(
                    f"Unsupported similarity metric type: {chat_cache.similarity_metric_type}"
                )

    # Sort results by similarity score (highest first)
    cache_answers = sorted(cache_answers, key=lambda x: x[0], reverse=True)
    cache_questions = sorted(cache_questions, key=lambda x: x[0], reverse=True)
    cache_ids = sorted(cache_ids, key=lambda x: x[0], reverse=True)
    if len(cache_answers) != 0:
        return_message = chat_cache.post_process_messages_func(
            [t[1] for t in cache_answers]
        )
        return_query = chat_cache.post_process_messages_func(
            [t[1] for t in cache_questions]
        )
        return_id = chat_cache.post_process_messages_func(
            [t[1] for t in cache_ids]
        )

        # Update hit count for analytics (async to avoid blocking)
        try:
            asyncio.create_task(asyncio.to_thread(chat_cache.data_manager.update_hit_count,return_id))
        except Exception:
            logging.info('update_hit_count except, please check!')

        # Record cache hit for reporting
        chat_cache.report.hint_cache()
        return cache_data_convert(return_message, return_query)
    return None