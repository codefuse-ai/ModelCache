# -*- coding: utf-8 -*-
import re
from typing import Dict, Any


def insert_last_content(data: Dict[str, Any], **_: Dict[str, Any]) -> Any:
    return data.get("query")[-1]["content"]


def query_last_content(data: Dict[str, Any], **_: Dict[str, Any]) -> Any:
    return data.get("query")[-1]["content"]


def last_content_without_prompt(data: Dict[str, Any], **params: Dict[str, Any]) -> Any:
    last_content_str = data.get("messages")[-1]["content"]
    prompts = params.get("prompts", [])
    if prompts is None:
        return last_content_str
    pattern = "|".join(prompts)
    new_content_str = re.sub(pattern, "", last_content_str)
    return new_content_str


def all_content(data: Dict[str, Any], **_: Dict[str, Any]) -> Any:
    s = ""
    messages = data.get("messages")
    for i, message in enumerate(messages):
        if i == len(messages) - 1:
            s += message["content"]
        else:
            s += message["content"] + "\n"
    return s


def nop(data: Dict[str, Any], **_: Dict[str, Any]) -> Any:
    return data


def get_prompt(data: Dict[str, Any], **_: Dict[str, Any]) -> Any:
    return data.get("prompt")


def get_file_name(data: Dict[str, Any], **_: Dict[str, Any]) -> str:
    return data.get("file").name


def get_file_bytes(data: Dict[str, Any], **_: Dict[str, Any]) -> bytes:
    return data.get("file").peek()


def get_input_str(data: Dict[str, Any], **_: Dict[str, Any]) -> str:
    input_data = data.get("input")
    return str(input_data["image"].peek()) + input_data["question"]


def get_input_image_file_name(data: Dict[str, Any], **_: Dict[str, Any]) -> str:
    input_data = data.get("input")
    return input_data["image"].name


def query_multi_splicing(data: Dict[str, Any], **_: Dict[str, Any]) -> Any:
    query_list = data.get("query")
    return multi_splicing(query_list)


def insert_multi_splicing(data: Dict[str, Any], **_: Dict[str, Any]) -> Any:
    insert_query_list = data['query']
    return multi_splicing(insert_query_list)

def query_with_role(data: Dict[str, Any], **_: Dict[str, Any]) -> Any:
    query = data["query"][-1]
    content = query["content"]
    role = query["role"]
    return role+": "+content

def multi_splicing(data_list) -> Any:
    result_str = ""
    for d in data_list:
        role = d.get('role', '')
        content = d.get('content', '')
        result_str += role + "###" + content + "|||"

    # 去掉最后一个"|||"
    result_str = result_str[:-3]

    return result_str


def multi_analysis(dialog_str):
    sub_strings = dialog_str.split('|||')

    dict_list = []
    for s in sub_strings:
        parts = s.split('###')

        if len(parts) == 2:
            role = parts[0]
            content = parts[1]
        elif len(parts) > 2:
            role = parts[0]
            content = '###'.join(parts[1:])
        else:
            content = 'exception'

        if content == '':
            d = {"role": role}
        else:
            d = {"role": role, "content": content}
        dict_list.append(d)

    # 3. 将每个字典添加到一个列表中，得到最终的列表
    result_list = dict_list

    # 输出结果
    return result_list
