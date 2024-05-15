# -*- coding: utf-8 -*-
import os
import time

import pymysql
import json
import base64
from typing import List
from modelcache_mm.manager.scalar_data.base import CacheStorage, CacheData
from DBUtils.PooledDB import PooledDB


class SQLStorage(CacheStorage):
    def __init__(
        self,
        db_type: str = "mysql",
        config=None
    ):

        self.host = config.get('mysql', 'host')
        self.port = int(config.get('mysql', 'port'))
        self.username = config.get('mysql', 'username')
        self.password = config.get('mysql', 'password')
        self.database = config.get('mysql', 'database')
        self.pool = PooledDB(
            creator=pymysql,
            host=self.host,
            user=self.username,
            password=self.password,
            port=self.port,
            database=self.database
        )

    def create(self):
        pass

    # def _insert(self, data: List):
    #     answer = data[0]
    #     text = data[1]
    #     image_url = data[2]
    #     image_id = data[3]
    #     model = data[4]
    #     answer_type = 0
    #
    #     table_name = "multimodal_answer"
    #     insert_sql = "INSERT INTO {} (question_text, image_url, image_id, answer, answer_type, model) VALUES (%s, %s, %s, %s, %s, %s)".format(table_name)
    #     conn = self.pool.connection()
    #     try:
    #         with conn.cursor() as cursor:
    #             # data insert operation
    #             values = (text, image_url, image_id, answer, answer_type, model)
    #             cursor.execute(insert_sql, values)
    #             conn.commit()
    #             id = cursor.lastrowid
    #     finally:
    #         # Close the connection and return it back to the connection pool
    #         conn.close()
    #     return id

    def _insert(self, data: List):
        answer = data[0]
        text = data[1]
        image_url = data[2]
        image_id = data[3]
        model = data[4]
        answer_type = 0

        table_name = "open_cache_mm_answer"
        insert_sql = "INSERT INTO {} (question_text, image_url, image_id, answer, answer_type, model) VALUES (%s, %s, %s, %s, %s, %s)".format(table_name)

        conn = self.pool.connection()
        try:
            with conn.cursor() as cursor:
                #  insert data operation
                values = (text, image_url, image_id, answer, answer_type, model)
                cursor.execute(insert_sql, values)
                conn.commit()
                id = cursor.lastrowid
        finally:
            # Close the connection and return it to the connection pool.
            conn.close()
        return id

    def batch_insert(self, all_data: List[CacheData]):
        ids = []
        for data in all_data:
            ids.append(self._insert(data))
        return ids

    def insert_query_resp(self, query_resp, **kwargs):
        error_code = query_resp.get('errorCode')
        error_desc = query_resp.get('errorDesc')
        cache_hit = query_resp.get('cacheHit')
        model = kwargs.get('model')
        query = kwargs.get('query')
        delta_time = kwargs.get('delta_time')
        hit_query = query_resp.get('hit_query')
        answer = query_resp.get('answer')

        if isinstance(hit_query, list):
            hit_query = json.dumps(hit_query, ensure_ascii=False)

        table_name = "open_cache_mm_query_log"
        insert_sql = "INSERT INTO {} (error_code, error_desc, cache_hit, model, query, delta_time, hit_query, answer) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)".format(table_name)
        conn = self.pool.connection()
        try:
            with conn.cursor() as cursor:
                # 执行插入数据操作
                values = (error_code, error_desc, cache_hit, model, query, delta_time, hit_query, answer)
                cursor.execute(insert_sql, values)
                conn.commit()
        finally:
            # 关闭连接，将连接返回给连接池
            conn.close()

    # def get_data_by_id(self, key: int):
    #     table_name = "cache_codegpt_answer"
    #     query_sql = "select question, answer, embedding_data, model from {} where id={}".format(table_name, key)
    #     conn_start = time.time()
    #     conn = self.pool.connection()
    #
    #     search_start = time.time()
    #     try:
    #         with conn.cursor() as cursor:
    #             # 执行数据库操作
    #             cursor.execute(query_sql)
    #             resp = cursor.fetchone()
    #     finally:
    #         # 关闭连接，将连接返回给连接池
    #         conn.close()
    #
    #     if resp is not None and len(resp) == 4:
    #         return resp
    #     else:
    #         return None

    def get_data_by_id(self, key: int):
        table_name = "open_cache_mm_answer"
        query_sql = "select question_text, image_url, image_id, answer, model from {} where id={}".format(table_name, key)
        conn = self.pool.connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query_sql)
                resp = cursor.fetchone()
        finally:
            conn.close()

        if resp is not None and len(resp) == 5:
            return resp
        else:
            return None

    def update_hit_count_by_id(self, primary_id: int):
        table_name = "open_cache_mm_answer"
        update_sql = "UPDATE {} SET hit_count = hit_count+1 WHERE id={}".format(table_name, primary_id)
        conn = self.pool.connection()

        try:
            with conn.cursor() as cursor:
                cursor.execute(update_sql)
                conn.commit()
        finally:
            conn.close()

    def get_ids(self, deleted=True):
        pass

    def mark_deleted(self, keys):
        table_name = "open_cache_mm_answer"
        delete_sql = "Delete from {} WHERE id in ({})".format(table_name, ",".join([str(i) for i in keys]))

        conn = self.pool.connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(delete_sql)
                delete_count = cursor.rowcount
                conn.commit()
        finally:
            conn.close()
        return delete_count

    def model_deleted(self, model_name):
        table_name = "open_cache_mm_answer"
        delete_sql = "Delete from {} WHERE model='{}'".format(table_name, model_name)
        conn = self.pool.connection()
        try:
            with conn.cursor() as cursor:
                resp = cursor.execute(delete_sql)
                conn.commit()
        finally:
            conn.close()
        return resp

    def clear_deleted_data(self):
        pass

    def count(self, state: int = 0, is_all: bool = False):
        pass

    def close(self):
        pass

    def count_answers(self):
        pass
