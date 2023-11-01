# -*- coding: utf-8 -*-
from modelcache.utils.lazy_import import LazyImport
huggingface = LazyImport("huggingface", globals(), "modelcache.embedding.huggingface")
data2vec = LazyImport("data2vec", globals(), "modelcache.embedding.data2vec")


def Huggingface(model="sentence-transformers/all-mpnet-base-v2"):
    return huggingface.Huggingface(model)


def Data2VecAudio(model="facebook/data2vec-audio-base-960h"):
    return data2vec.Data2VecAudio(model)
