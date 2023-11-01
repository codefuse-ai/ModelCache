# Codefuse-ModelCache LLMs Semantic Cache
## Contents
- [news](#news)
- [Introduction](#Introduction)
- [Quick-Deployment](#Quick-Deployment)
- [Service-Access](#Service-Access)
- [Articles](#Articles)
- [Modules](#Modules)
- [Core-Features](#Core-Features)
- [Acknowledgements](#Acknowledgements)
## news
[2023.08.26] codefuse-ModelCache...
### Introduction
Codefuse-ModelCache is a semantic cache for large language models (LLMs). By caching pre-generated model results, it reduces response time for similar requests and improves user experience. <br />This project aims to optimize services by introducing a caching mechanism. It helps businesses and research institutions reduce the cost of inference deployment, improve model performance and efficiency, and provide scalable services for large models.  Through open-source, we aim to share and exchange technologies related to large model semantic cache.
## Quick Deployment
### Dependencies

- Python version: 3.8 and above
- Package Installation
```shell
pip install requirements.txt 
```
### Environment Configuration
Before starting the service, the following environment configurations should be performed:

1. Install the relational database MySQL and import the SQL file to create the data tables. The SQL file can be found at: reference_doc/create_table.sql
2. Install the vector database Milvus.
3. Add the database access information to the configuration files: 
   1. modelcache/config/milvus_config.ini 
   2. modelcache/config/mysql_config.ini
4. Download the embedding model bin file from the following address: [https://huggingface.co/shibing624/text2vec-base-chinese/tree/main](https://huggingface.co/shibing624/text2vec-base-chinese/tree/main). Place the downloaded bin file in the model/text2vec-base-chinese folder.
5. Start the backend service using the flask4modelcache.py script.
## Service-Access
The current service provides three core functionalities through RESTful API.: Cache-Writing, Cache-Querying, and Cache-Clearing. Demos:
### Cache-Writing
```python
import json
import requests
url = 'http://127.0.0.1:5000/modelcache'
type = 'insert'
scope = {"model": "CODEGPT-1008"}
chat_info = [{"query": [{"role": "system", "content": "You are an AI code assistant and you must provide neutral and harmless answers to help users solve code-related problems."}, {"role": "user", "content": "你是谁?"}],
                  "answer": "Hello, I am an intelligent assistant. How can I assist you?"}]
data = {'type': type, 'scope': scope, 'chat_info': chat_info}
headers = {"Content-Type": "application/json"}
res = requests.post(url, headers=headers, json=json.dumps(data))
```
### Cache-Querying
```python
import json
import requests
url = 'http://127.0.0.1:5000/modelcache'
type = 'query'
scope = {"model": "CODEGPT-1008"}
query = [{"role": "system", "content": "You are an AI code assistant and you must provide neutral and harmless answers to help users solve code-related problems."}, {"role": "user", "content": "Who are you?"}]
data = {'type': type, 'scope': scope, 'query': query}

headers = {"Content-Type": "application/json"}
res = requests.post(url, headers=headers, json=json.dumps(data))
```
### Cache-Clearing
```python
import json
import requests
url = 'http://127.0.0.1:5000/modelcache'
type = 'remove'
scope = {"model": "CODEGPT-1008"}
remove_type = 'truncate_by_model'
data = {'type': type, 'scope': scope, 'remove_type': remove_type}

headers = {"Content-Type": "application/json"}
res = requests.post(url, headers=headers, json=json.dumps(data))
```
## Articles
Coming soon...
## modules
![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2023/png/275821/1698031968643-35914fc7-bb62-455e-9431-69bca8ba3368.png#clientId=uf441e764-1311-4&from=paste&height=408&id=h5p1L&originHeight=1152&originWidth=1796&originalType=binary&ratio=2&rotation=0&showTitle=false&size=465700&status=done&style=none&taskId=u6f53deb1-7821-47e0-af8a-87d899e3f7a&title=&width=636)
## Core-Features
In ModelCache, we adopted the main idea of GPTCache,  includes core modules: adapter, embedding, similarity, and data_manager. The adapter module is responsible for handling the business logic of various tasks and can connect the embedding, similarity, and data_manager modules. The embedding module is mainly responsible for converting text into semantic vector representations, it transforms user queries into vector form.The rank module is used for sorting and evaluating the similarity of the recalled vectors. The data_manager module is primarily used for managing the database. In order to better facilitate industrial applications, we have made architectural and functional upgrades as follows:

- [x] We have modified it similar to Redis and embedded it into the LLMs product, providing semantic caching capabilities. This ensures that it does not interfere with LLM calls, security audits, and other functionalities,  achieving compatibility with all large-scale model services.
- [x] Multiple Model Loading Schemes: 
   - Support loading local embedding models to address Hugging Face network connectivity issues. 
   - Support loading various pretrained model embedding layers.
- [x] Data Isolation Capability 
   - Environment Isolation: Can pull different database configurations based on the environment to achieve environment isolation (dev, prepub, prod). 
   - Multi-tenant Data Isolation: Dynamically create collections based on the model for data isolation, addressing data isolation issues in multi-model/services scenarios in LLMs products.
- [x] Support for System Commands: Adopting a concatenation approach to address the issue of system commands in the prompt format.
- [x] Differentiation of Long and Short Texts: Long texts pose more challenges for similarity evaluation. To address this, we have added differentiation between long and short texts, allowing for separate configuration of threshold values for determining similarity.
- [x] Milvus Performance Optimization: The consistency_level of Milvus has been adjusted to "Session" level, which can result in better performance.
- [x] Data Management Capability: 
   - Ability to clear the cache, used for data management after model upgrades.
   - Hitquery recall for subsequent data analysis and model iteration reference. 
   - Asynchronous log write-back capability for data analysis and statistics. 
   - Added model field and data statistics field for feature expansion.

Future Features Under Development: 

- [ ] Data isolation based on hyperparameters. 
- [ ] System prompt partitioning storage capability to enhance accuracy and efficiency of similarity matching.
- [ ] More versatile embedding models and similarity evaluation algorithms.
## Acknowledgements
This project has referenced the following open-source projects. We would like to express our gratitude to the projects and their developers for their contributions and research.<br />[GPTCache](https://github.com/zilliztech/GPTCache)
