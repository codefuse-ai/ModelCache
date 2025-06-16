# -*- coding: utf-8 -*-
import uuid

import pymysql
import json
import numpy as np
from typing import List
from modelcache.manager.scalar_data.base import CacheStorage, CacheData
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

    def _insert(self, data: List):
        answer = data[0]
        question = data[1]
        embedding_data = data[2]
        model = data[3]
        answer_type = 0
        embedding_data = embedding_data.tobytes()
        is_deleted = 0
        _id = str(uuid.uuid4())

        table_name = "modelcache_llm_answer"
        insert_sql = f"""
            INSERT INTO {table_name} 
            (id, question, answer, answer_type, model, embedding_data, is_deleted)
            VALUES (%s, %s, %s, %s, %s, _binary%s, %s)
        """
        conn = self.pool.connection()
        try:
            with conn.cursor() as cursor:
                # 执行插入数据操作
                values = (_id, question, answer, answer_type, model, embedding_data, is_deleted)
                cursor.execute(insert_sql, values)
                conn.commit()
        finally:
            # 关闭连接，将连接返回给连接池
            conn.close()
        return _id

    def batch_insert(self, all_data: List[List]):
        table_name = "modelcache_llm_answer"
        insert_sql = f"""
            INSERT INTO {table_name}
            (id, question, answer, answer_type, model, embedding_data, is_deleted)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        values_list = []
        ids = []

        for data in all_data:
            answer = data[0]
            question = data[1]
            embedding_data = data[2].tobytes()
            model = data[3]
            answer_type = 0
            is_deleted = 0
            _id = str(uuid.uuid4())
            ids.append(_id)

            values_list.append((
                _id, question, answer, answer_type, model, embedding_data, is_deleted
            ))

        conn = self.pool.connection()
        try:
            with conn.cursor() as cursor:
                cursor.executemany(insert_sql, values_list)
                conn.commit()
        finally:
            conn.close()

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

        table_name = "modelcache_query_log"
        insert_sql = f"""
            INSERT INTO {table_name} 
            (error_code, error_desc, cache_hit, model, query, delta_time, hit_query, answer) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
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

    def get_data_by_id(self, key: int):
        table_name = "modelcache_llm_answer"
        query_sql = f"""
            SELECT answer, question, embedding_data, model
            FROM {table_name}
            WHERE id = %s
        """
        conn = self.pool.connection()
        try:
            with conn.cursor() as cursor:
                # 执行数据库操作
                cursor.execute(query_sql, (key,))
                resp = cursor.fetchone()
        finally:
            # 关闭连接，将连接返回给连接池
            conn.close()

        if resp is not None and len(resp) == 4:
            return resp[0], resp[1], np.frombuffer(resp[2], dtype=np.float32), resp[3]
        else:
            return None

    def update_hit_count_by_id(self, primary_id: int):
        table_name = "modelcache_llm_answer"
        update_sql = f"""
            UPDATE {table_name} 
            SET hit_count = hit_count+1 
            WHERE id = %s
        """
        conn = self.pool.connection()

        # 使用连接执行更新数据操作
        try:
            with conn.cursor() as cursor:
                # 执行更新数据操作
                cursor.execute(update_sql,(primary_id,))
                conn.commit()
        finally:
            # 关闭连接，将连接返回给连接池
            conn.close()

    def get_ids(self, deleted=True):
        table_name = "modelcache_llm_answer"
        state = 1 if deleted else 0
        query_sql = f"""
            SELECT id 
            FROM {table_name} 
            WHERE is_deleted = %s
        """
        
        conn = self.pool.connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query_sql, (state,))
                ids = [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()
        
        return ids

    def mark_deleted(self, keys):
        table_name = "modelcache_llm_answer"
        placeholders = ",".join(["%s"] * len(keys))
        mark_sql = f"""
            UPDATE {table_name}
            SET is_deleted=1 
            WHERE id in ({placeholders})
        """

        conn = self.pool.connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(mark_sql, keys)
                delete_count = cursor.rowcount
                conn.commit()
        finally:
            conn.close()
        return delete_count

    def model_deleted(self, model_name):
        table_name = "modelcache_llm_answer"
        delete_sql = f"""
            Delete from {table_name}
            WHERE model = %s
        """

        table_log_name = "modelcache_query_log"
        delete_log_sql = f"""
            Delete from {table_log_name} 
            WHERE model = %s
        """

        conn = self.pool.connection()
        # 使用连接执行删除数据操作
        try:
            with conn.cursor() as cursor:
                # 执行删除数据操作
                resp = cursor.execute(delete_sql, (model_name,))
                conn.commit()
                # 执行删除该模型对应日志操作 resp_log行数不返回
                resp_log = cursor.execute(delete_log_sql, (model_name,))
                conn.commit()  # 分别提交事务
        finally:
            # 关闭连接，将连接返回给连接池
            conn.close()
        return resp

    def clear_deleted_data(self):
        table_name = "modelcache_llm_answer"
        delete_sql = f"""
            DELETE FROM {table_name} 
            WHERE is_deleted = 1
        """
        
        conn = self.pool.connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(delete_sql)
                delete_count = cursor.rowcount
                conn.commit()
        finally:
            conn.close()
        
        return delete_count

    def count(self, state: int = 0, is_all: bool = False):
        table_name = "modelcache_llm_answer"

        # we're not using prepared statements here, so we need to ensure state is an integer
        if not isinstance(state, int):
            raise ValueError("'state' must be an integer.")

        if is_all:
            count_sql = f"SELECT COUNT(*) FROM {table_name}"
        else:
            count_sql = f"SELECT COUNT(*) FROM {table_name} WHERE is_deleted = {state}"
        
        conn = self.pool.connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(count_sql)
                num = cursor.fetchone()[0]
        finally:
            conn.close()
        
        return num

    def close(self):
        pass

    def count_answers(self):
        pass

    def flush(self):
        pass