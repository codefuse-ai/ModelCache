# -*- coding: utf-8 -*-
import logging
import time
from modelcache import cache
from modelcache.utils.error import NotInitError
from modelcache.utils.time import time_cal
from modelcache.processor.pre import multi_analysis
from FlagEmbedding import FlagReranker

# USE_RERANKER = True  # 如果为 True 则启用 reranker，否则使用原有逻辑


def adapt_query(cache_data_convert, *args, **kwargs):
    chat_cache = kwargs.pop("cache_obj", cache)
    scope = kwargs.pop("scope", None)
    use_reranker = kwargs.pop("use_reranker", False)
    model = scope['model']
    if not chat_cache.has_init:
        raise NotInitError()
    cache_enable = chat_cache.cache_enable_func(*args, **kwargs)
    context = kwargs.pop("cache_context", {})
    embedding_data = None
    cache_factor = kwargs.pop("cache_factor", 1.0)
    pre_embedding_data = chat_cache.query_pre_embedding_func(
        kwargs,
        extra_param=context.get("pre_embedding_func", None),
        prompts=chat_cache.config.prompts,
    )
    if cache_enable:
        embedding_data = time_cal(
            chat_cache.embedding_func,
            func_name="embedding",
            report_func=chat_cache.report.embedding,
        )(pre_embedding_data)
    if cache_enable:
        cache_data_list = time_cal(
            chat_cache.data_manager.search,
            func_name="vector_search",
            report_func=chat_cache.report.search,
        )(
            embedding_data,
            extra_param=context.get("search_func", None),
            top_k=kwargs.pop("top_k", -1),
            model=model
        )
        cache_answers = []
        cache_questions = []
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

        if use_reranker:
            reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=False)
            for cache_data in cache_data_list:
                primary_id = cache_data[1]
                ret = chat_cache.data_manager.get_scalar_data(
                    cache_data, extra_param=context.get("get_scalar_data", None)
                )
                if ret is None:
                    continue

                rank = reranker.compute_score([pre_embedding_data, ret[0]], normalize=True)

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
                        cache_answers.append((rank, ret[1]))
                        cache_questions.append((rank, ret[0]))
                        cache_ids.append((rank, primary_id))
                else:
                    if rank_threshold_long <= rank:
                        cache_answers.append((rank, ret[1]))
                        cache_questions.append((rank, ret[0]))
                        cache_ids.append((rank, primary_id))
        else:
            # 不使用 reranker 时，走原来的逻辑
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
                        "question": pre_embedding_data,
                        "embedding": embedding_data,
                    }

                    eval_cache_data = {
                        "question": ret[0],
                        "answer": ret[1],
                        "search_result": cache_data,
                        "embedding": None
                    }
                rank = chat_cache.similarity_evaluation.evaluation(
                    eval_query_data,
                    eval_cache_data,
                    extra_param=context.get("evaluation_func", None),
                )

                if len(pre_embedding_data) <= 256:
                    if rank_threshold <= rank:
                        cache_answers.append((rank, ret[1]))
                        cache_questions.append((rank, ret[0]))
                        cache_ids.append((rank, primary_id))
                else:
                    if rank_threshold_long <= rank:
                        cache_answers.append((rank, ret[1]))
                        cache_questions.append((rank, ret[0]))
                        cache_ids.append((rank, primary_id))

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
            # 更新命中次数
            try:
                chat_cache.data_manager.update_hit_count(return_id)
            except Exception:
                logging.info('update_hit_count except, please check!')

            chat_cache.report.hint_cache()
            return cache_data_convert(return_message, return_query)