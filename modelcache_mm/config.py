# -*- coding: utf-8 -*-
from typing import Optional, Callable, List
from modelcache.utils.error import CacheError


class Config:

    def __init__(
            self,
            log_time_func: Optional[Callable[[str, float], None]] = None,
            similarity_threshold: float = 0.95,
            similarity_threshold_long: float = 0.95,
            prompts: Optional[List[str]] = None
    ):
        if similarity_threshold < 0 or similarity_threshold > 1:
            raise CacheError(
                "Invalid the similarity threshold param, reasonable range: 0-1"
            )
        self.log_time_func = log_time_func
        self.similarity_threshold = similarity_threshold
        self.similarity_threshold_long = similarity_threshold_long
        self.prompts = prompts
