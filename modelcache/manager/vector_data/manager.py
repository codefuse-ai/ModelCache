# -*- coding: utf-8 -*-
from modelcache.utils.error import NotFoundError, ParamError

TOP_K = 1
FAISS_INDEX_PATH = "faiss.index"
DIMENSION = 0
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
MILVUS_USER = ""
MILVUS_PSW = ""
MILVUS_SECURE = False
MILVUS_INDEX_PARAMS = {
    "metric_type": "L2",
    "index_type": "HNSW",
    "params": {"M": 8, "efConstruction": 64},
}

COLLECTION_NAME = "modelcache"


class VectorBase:
    """
    VectorBase to manager the vector base.
    """

    def __init__(self):
        raise EnvironmentError(
            "VectorBase is designed to be instantiated, please using the `VectorBase.get(name)`."
        )

    @staticmethod
    def check_dimension(dimension):
        if dimension <= 0:
            raise ParamError(
                f"the dimension should be greater than zero, current value: {dimension}."
            )

    @staticmethod
    def get(name, **kwargs):
        top_k = kwargs.get("top_k", TOP_K)
        if name == "milvus":
            from modelcache.manager.vector_data.milvus import Milvus
            dimension = kwargs.get("dimension", DIMENSION)
            milvus_config = kwargs.get("milvus_config")
            VectorBase.check_dimension(dimension)
            host = milvus_config.get('milvus', 'host')
            port = milvus_config.get('milvus', 'port')
            user = milvus_config.get('milvus', 'user')
            password = milvus_config.get('milvus', 'password')

            secure = kwargs.get("secure", MILVUS_SECURE)
            collection_name = kwargs.get("collection_name", COLLECTION_NAME)
            index_params = kwargs.get("index_params", MILVUS_INDEX_PARAMS)
            search_params = kwargs.get("search_params", None)
            local_mode = kwargs.get("local_mode", False)
            local_data = kwargs.get("local_data", "./milvus_data")
            vector_base = Milvus(
                host=host,
                port=port,
                user=user,
                password=password,
                secure=secure,
                collection_name=collection_name,
                dimension=dimension,
                top_k=top_k,
                index_params=index_params,
                search_params=search_params,
                local_mode=local_mode,
                local_data=local_data
            )
        elif name == "redis":
            from modelcache.manager.vector_data.redis import RedisVectorStore
            dimension = kwargs.get("dimension", DIMENSION)
            VectorBase.check_dimension(dimension)

            redis_config = kwargs.get("redis_config")
            host = redis_config.get('redis', 'host')
            port = redis_config.get('redis', 'port')
            user = redis_config.get('redis', 'user')
            password = redis_config.get('redis', 'password')
            namespace = kwargs.get("namespace", "")
            # collection_name = kwargs.get("collection_name", COLLECTION_NAME)

            vector_base = RedisVectorStore(
                host=host,
                port=port,
                username=user,
                password=password,
                namespace=namespace,
                top_k=top_k,
                dimension=dimension,
            )
        elif name == "faiss":
            from modelcache.manager.vector_data.faiss import Faiss

            dimension = kwargs.get("dimension", DIMENSION)
            index_path = kwargs.pop("index_path", FAISS_INDEX_PATH)
            VectorBase.check_dimension(dimension)
            vector_base = Faiss(
                index_file_path=index_path, dimension=dimension, top_k=top_k
            )
        elif name == "chromadb":
            from modelcache.manager.vector_data.chroma import Chromadb

            chromadb_config = kwargs.get("chromadb_config", None)
            persist_directory = chromadb_config.get('chromadb','persist_directory')

            vector_base = Chromadb(
                persist_directory=persist_directory,
                top_k=top_k,
            )
        elif name == "hnswlib":
            from modelcache.manager.vector_data.hnswlib_store import Hnswlib

            dimension = kwargs.get("dimension", DIMENSION)
            index_path = kwargs.pop("index_path", "./hnswlib_index.bin")
            max_elements = kwargs.pop("max_elements", 100000)
            VectorBase.check_dimension(dimension)
            vector_base = Hnswlib(
                index_file_path=index_path, dimension=dimension,
                top_k=top_k, max_elements=max_elements
            )
        else:
            raise NotFoundError("vector store", name)
        return vector_base
