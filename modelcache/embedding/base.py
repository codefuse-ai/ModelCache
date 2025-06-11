# -*- coding: utf-8 -*-
from abc import abstractmethod, ABCMeta

from modelcache.utils.lazy_import import LazyImport
from enum import Enum
huggingface = LazyImport("huggingface", globals(), "modelcache.embedding.huggingface")
data2vec = LazyImport("data2vec", globals(), "modelcache.embedding.data2vec")
llmEmb = LazyImport("llmEmb", globals(), "modelcache.embedding.llmEmb")
fasttext = LazyImport("fasttext", globals(), "modelcache.embedding.fasttext")
paddlenlp = LazyImport("paddlenlp", globals(), "modelcache.embedding.paddlenlp")
timm = LazyImport("timm", globals(), "modelcache.embedding.timm")
huggingface_tei = LazyImport("huggingface_tei", globals(), "modelcache.embedding.huggingface_tei")
bge_m3 = LazyImport("bge_m3", globals(), "modelcache.embedding.bge_m3")

# define the embedding model enum
class EmbeddingModel(Enum):
    """
    Enum for different embedding models.
    """
    # todo: fill in the dimension and model_path for each embedding model as needed
    HUGGINGFACE_ALL_MPNET_BASE_V2 = {"dimension":768, "model_path":"sentence-transformers/all-mpnet-base-v2"}
    HUGGINGFACE_ALL_MINILM_L6_V2 = {"dimension":384, "model_path":"sentence-transformers/all-MiniLM-L6-v2"}
    HUGGINGFACE_ALL_MINILM_L12_V2 = {"dimension":384, "model_path":"sentence-transformers/all-MiniLM-L12-v2"}
    DATA2VEC_AUDIO = {"dimension":None, "model_path":"model/text2vec-base-chinese/"}
    LLM_EMB2VEC_AUDIO = {"dimension":None, "model_path":None}
    FASTTEXT = {"dimension":None, "model_path":None}
    PADDLE_NLP = {"dimension":None, "model_path":None}
    TIMM = {"dimension":None, "model_path":None}
    HUGGINGFACE_TEI = {"dimension":None, "model_path":None}
    BGE_M3 = {"dimension":None, "model_path":None}


class MetricType(Enum):
    """
    Enum for different metric types used in similarity evaluation.
    Different models may require different metrics for optimal performance.
    """
    COSINE = "COSINE"
    L2 = "L2"


class BaseEmbedding(metaclass=ABCMeta):
    """
    _Embedding base.
    """

    @abstractmethod
    def to_embeddings(self, data, **kwargs):
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        return 0

    @staticmethod
    def get(model:EmbeddingModel, **kwargs):
        """
        Get the embedding model instance based on the specified model type.
        :param model: The embedding model type.
        :type model: EmbeddingModel
        :param kwargs: Additional parameters for the model.
        :return: An instance of the specified embedding model.
        :rtype: BaseEmbedding
        :raises ValueError: If the specified model type is not supported.
        """
        if model == EmbeddingModel.HUGGINGFACE_ALL_MPNET_BASE_V2:
            model_path = kwargs.pop("model_path","sentence-transformers/all-mpnet-base-v2")
            return huggingface.Huggingface(model_path)

        elif model == EmbeddingModel.DATA2VEC_AUDIO:
            model_path = kwargs.pop("model_path","model/text2vec-base-chinese/")
            return data2vec.Data2VecAudio(model_path)

        elif model == EmbeddingModel.LLM_EMB2VEC_AUDIO:
            return llmEmb.LlmEmb2Vec()

        elif model == EmbeddingModel.FASTTEXT:
            model_path = kwargs.pop("model_path","en")
            dim = kwargs.pop("dim", None)
            return fasttext.FastText(model_path, dim)

        elif model == EmbeddingModel.PADDLE_NLP:
            model_path = kwargs.pop("model_path", "ernie-3.0-medium-zh")
            return paddlenlp.PaddleNLP(model_path)

        elif model == EmbeddingModel.TIMM:
            model_path = kwargs.pop("model_path", "resnet50")
            device = kwargs.pop("device", "default")
            return timm.Timm(model_path, device)

        elif model == EmbeddingModel.HUGGINGFACE_TEI:
            base_url = kwargs.pop("base_url")
            model_path = kwargs.pop("model_path")
            return huggingface_tei.HuggingfaceTEI(base_url, model_path)

        elif model == EmbeddingModel.BGE_M3:
            model_path = kwargs.pop("model_path","model/bge-m3")
            return bge_m3.BgeM3Embedding(model_path)

        else:
            raise ValueError(f"Unsupported embedding model: {model}")

