# -*- coding: utf-8 -*-
from modelcache.utils.lazy_import import LazyImport
clip = LazyImport("clip", globals(), "modelcache_mm.embedding.clip")


def Clip2Vec(model="damo/multi-modal_clip-vit-base-patch16_zh"):
    return clip.ClipAudio(model)
