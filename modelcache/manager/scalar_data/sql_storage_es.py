# -*- coding: utf-8 -*-
import json
from typing import List
from elasticsearch import Elasticsearch, helpers
from modelcache.manager.scalar_data.base import CacheStorage, CacheData
import time
from snowflake import SnowflakeGenerator


class SQLStorage(CacheStorage):
    def __init__(
            self,
            db_type: str = "elasticsearch",
            config=None
    ):
        self.host = config.get('elasticsearch', 'host')
        self.port = int(config.get('elasticsearch', 'port'))
        self.client = Elasticsearch(
            hosts=[{"host": self.host, "port": self.port}],
            timeout=30,
            http_auth=('esuser', 'password')
        )

        self.log_index = "modelcache_query_log"
        self.ans_index = "modelcache_llm_answer"
        self.create()
        self.instance_id = 1  # 雪花算法使用的机器id 使用同一套数据库的分布式系统需要配置不同id
        # 生成雪花id
        self.snowflake_id = SnowflakeGenerator(self.instance_id)

    def create(self):
        answer_index_body = {
            "mappings": {
                "properties": {
                    "gmt_create": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                    "gmt_modified": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                    "question": {"type": "text"},
                    "answer": {"type": "text"},
                    "answer_type": {"type": "integer"},
                    "hit_count": {"type": "integer"},
                    "model": {"type": "keyword"},
                    "embedding_data": {"type": "binary"},
                    "is_deleted": {"type": "integer"},
                }
            }
        }

        log_index_body = {
            "mappings": {
                "properties": {
                    "gmt_create": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                    "gmt_modified": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                    "error_code": {"type": "integer"},
                    "error_desc": {"type": "text"},
                    "cache_hit": {"type": "keyword"},
                    "delta_time": {"type": "float"},
                    "model": {"type": "keyword"},
                    "query": {"type": "text"},
                    "hit_query": {"type": "text"},
                    "answer": {"type": "text"}
                }
            }
        }

        if not self.client.indices.exists(index="modelcache_llm_answer"):
            self.client.indices.create(index="modelcache_llm_answer", body=answer_index_body)

        if not self.client.indices.exists(index="modelcache_query_log"):
            self.client.indices.create(index="modelcache_query_log", body=log_index_body)

    def _insert(self, data: List) -> str or None:
        doc = {
            "answer": data[0],
            "question": data[1],
            "embedding_data": data[2].tolist() if hasattr(data[2], "tolist") else data[2],
            "model": data[3],
            "answer_type": 0,
            "hit_count": 0,
            "is_deleted": 0
        }

        try:

            response = self.client.index(
                index=self.ans_index,
                id=next(self.snowflake_id),
                body=doc,
            )
            return int(response['_id'])
        except Exception as e:

            print(f"Failed to insert document: {e}")
            return None

    def batch_insert(self, all_data: List[List]) -> List[str]:
        successful_ids = []
        for data in all_data:
            _id = self._insert(data)
            if _id is not None:
                successful_ids.append(_id)
        self.client.indices.refresh(index=self.ans_index)  # 批量插入后手动刷新

        return successful_ids

    def insert_query_resp(self, query_resp, **kwargs):
        doc = {
            "error_code": query_resp.get('errorCode'),
            "error_desc": query_resp.get('errorDesc'),
            "cache_hit": query_resp.get('cacheHit'),
            "model": kwargs.get('model'),
            "query": kwargs.get('query'),
            "delta_time": kwargs.get('delta_time'),
            "hit_query": json.dumps(query_resp.get('hit_query'), ensure_ascii=False) if isinstance(
                query_resp.get('hit_query'), list) else query_resp.get('hit_query'),
            "answer": query_resp.get('answer'),
            "hit_count": 0,
            "is_deleted": 0

        }
        self.client.index(index=self.log_index, body=doc)

    def get_data_by_id(self, key: int):
        try:
            response = self.client.get(index=self.ans_index, id=key, _source=['question', 'answer', 'embedding_data', 'model'])
            source = response["_source"]
            result = [
                source.get('question'),
                source.get('answer'),
                source.get('embedding_data'),
                source.get('model')
            ]
            return result
        except Exception as e:
            print(e)

    def update_hit_count_by_id(self, primary_id: int):
        self.client.update(
            index=self.ans_index,
            id=primary_id,
            body={"script": {"source": "ctx._source.hit_count += 1"}}
        )

    def get_ids(self, deleted=True):
        query = {
            "query": {
                "term": {"is_deleted": 1 if deleted else 0}
            }
        }
        response = self.client.search(index=self.ans_index, body=query)
        return [hit["_id"] for hit in response["hits"]["hits"]]

    def mark_deleted(self, keys):
        actions = [
            {
                "_op_type": "update",
                "_index": self.ans_index,
                "_id": key,
                "doc": {"is_deleted": 1}
            }
            for key in keys
        ]
        responses = helpers.bulk(self.client, actions)
        return responses[0]  # 返回更新的文档数

    def model_deleted(self, model_name):
        query = {
            "query": {
                "term": {"model": model_name}
            }
        }

        response = self.client.delete_by_query(index=self.ans_index, body=query)
        return response["deleted"]

    def clear_deleted_data(self):
        query = {
            "query": {
                "term": {"is_deleted": 1}
            }
        }
        response = self.client.delete_by_query(index=self.ans_index, body=query)
        return response["deleted"]

    def count(self, state: int = 0, is_all: bool = False):
        query = {"query": {"match_all": {}}} if is_all else {"query": {"term": {"is_deleted": state}}}
        response = self.client.count(index=self.ans_index, body=query)
        return response["count"]

    def close(self):
        self.client.close()

    def count_answers(self):
        query = {"query": {"match_all": {}}}
        response = self.client.count(index=self.ans_index, body=query)
        return response["count"]
