# -*- coding: utf-8 -*-
import asyncio

from modelcache.utils.error import NotInitError
from modelcache.utils.time import time_cal


async def adapt_insert(*args, **kwargs):
    chat_cache = kwargs.pop("cache_obj")
    model = kwargs.pop("model", None)
    require_object_store = kwargs.pop("require_object_store", False)

    # Validate object store availability if required
    if require_object_store:
        assert chat_cache.data_manager.o, "Object store is required for adapter."

    context = kwargs.pop("cache_context", {})
    chat_info = kwargs.pop("chat_info", [])

    # Initialize collections for parallel processing
    pre_embedding_data_list = []      # Preprocessed data ready for embedding
    embedding_futures_list = []       # Async embedding generation tasks
    llm_data_list = []                # Extracted LLM response data

    # Process each chat entry and prepare for parallel embedding generation
    for row in chat_info:
        # Preprocess chat data using configured preprocessing function
        pre_embedding_data = chat_cache.insert_pre_embedding_func(
            row,
            extra_param=context.get("pre_embedding_func", None),
            prompts=chat_cache.prompts,
        )
        pre_embedding_data_list.append(pre_embedding_data)
        llm_data_list.append(row['answer'])  # Extract answer text for storage

        # Create async embedding generation task with performance monitoring
        embedding_future = time_cal(
            chat_cache.embedding_func,
            func_name="embedding",
            report_func=chat_cache.report.embedding,
            cache_obj=chat_cache
        )(pre_embedding_data)
        embedding_futures_list.append(embedding_future)

    # Wait for all embedding generation tasks to complete in parallel
    embedding_data_list = await asyncio.gather(*embedding_futures_list)

    # Save all processed data to the data manager asynchronously
    await asyncio.to_thread(
        chat_cache.data_manager.save,
        pre_embedding_data_list,
        llm_data_list,
        embedding_data_list,
        model=model,
        extra_param=context.get("save_func", None)
    )
    return 'success'
