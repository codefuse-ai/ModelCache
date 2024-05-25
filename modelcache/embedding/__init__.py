# -*- coding: utf-8 -*-
from modelcache.utils.lazy_import import LazyImport
huggingface = LazyImport("huggingface", globals(), "modelcache.embedding.huggingface")
data2vec = LazyImport("data2vec", globals(), "modelcache.embedding.data2vec")
llmEmb = LazyImport("llmEmb", globals(), "modelcache.embedding.llmEmb")
fasttext = LazyImport("fasttext", globals(), "modelcache.embedding.fasttext")
paddlenlp = LazyImport("paddlenlp", globals(), "modelcache.embedding.paddlenlp")
timm = LazyImport("timm", globals(), "modelcache.embedding.timm")
huggingface_tei = LazyImport("huggingface_tei", globals(), "modelcache.embedding.huggingface_tei")


def Huggingface(model="sentence-transformers/all-mpnet-base-v2"):
    return huggingface.Huggingface(model)


def Data2VecAudio(model="model/text2vec-base-chinese/"):
    return data2vec.Data2VecAudio(model)


def LlmEmb2vecAudio():
    return llmEmb.LlmEmb2Vec()


def FastText(model="en", dim=None):
    return fasttext.FastText(model, dim)


def PaddleNLP(model="ernie-3.0-medium-zh"):
    return paddlenlp.PaddleNLP(model)


def Timm(model="resnet50", device="default"):
    return timm.Timm(model, device)

def TextEmbeddingsInference(base_url, model):
    return huggingface_tei.HuggingfaceTEI(base_url, model)