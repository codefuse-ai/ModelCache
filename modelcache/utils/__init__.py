# -*- coding: utf-8 -*-
import importlib.util
from typing import Optional
from modelcache.utils.dependency_control import prompt_install


def _check_library(libname: str, prompt: bool = True, package: Optional[str] = None):
    is_avail = False
    if importlib.util.find_spec(libname):
        is_avail = True
    if not is_avail and prompt:
        prompt_install(package if package else libname)
    return is_avail


def import_onnxruntime():
    _check_library("onnxruntime")


def import_huggingface():
    _check_library("transformers")


def import_huggingface_hub():
    _check_library("huggingface_hub", package="huggingface-hub")


def import_pymysql():
    _check_library("pymysql")


def import_sql_client(db_name):
    if db_name in ["mysql"]:
        import_pymysql()


def import_pymilvus():
    _check_library("pymilvus")


def import_milvus_lite():
    _check_library("milvus")


def import_faiss():
    _check_library("faiss", package="faiss-cpu")


def import_torch():
    _check_library("torch")


def import_fasttext():
    _check_library("fasttext")


def import_paddle():
    prompt_install("protobuf==3.20.0")
    _check_library("paddlepaddle")


def import_paddlenlp():
    _check_library("paddlenlp")
