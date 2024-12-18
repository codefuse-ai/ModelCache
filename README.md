<div align="center">
<h1>
ModelCache
</h1>
</div>

<p align="center">
<div align="center">
<h4 align="center">
    <p>
        <a href="https://github.com/codefuse-ai/CodeFuse-ModelCache/blob/main/README_CN.md">中文</a> |
     <b>English</b>
    </p>
</h4>
</div>

## Contents

- [Contents](#contents)
- [News](#news)
  - [Introduction](#introduction)
- [Architecture](#architecture)
- [Quick start](#quick-start)
  - [Dependencies](#dependencies)
  - [Start service](#start-service)
    - [Start Demo](#start-demo)
    - [Start normal service](#start-normal-service)
- [Access the service](#access-the-service)
  - [Write cache](#write-cache)
  - [Query cache](#query-cache)
  - [Clear cache](#clear-cache)
- [Function comparison](#function-comparison)
- [Core-Features](#core-features)
- [Todo List](#todo-list)
  - [Adapter](#adapter)
  - [Embedding model\&inference](#embedding-modelinference)
  - [Scalar Storage](#scalar-storage)
  - [Vector Storage](#vector-storage)
  - [Ranking](#ranking)
  - [Service](#service)
- [Acknowledgements](#acknowledgements)
- [Contributing](#contributing)

## News

- 🔥🔥[2024.04.09] Added Redis Search to store and retrieve embeddings in multi-tenant. This can reduce the interaction time between Cache and vector databases to 10ms.
- 🔥🔥[2023.12.10] Integrated LLM embedding frameworks such as 'llmEmb', 'ONNX', 'PaddleNLP', 'FastText', and the image embedding framework 'timm' to bolster embedding functionality.
- 🔥🔥[2023.11.20] Integrated local storage, such as sqlite and faiss. This enables you to initiate quick and convenient tests.
- [2023.08.26] codefuse-ModelCache...

### Introduction

Codefuse-ModelCache is a semantic cache for large language models (LLMs). By caching pre-generated model results, it reduces response time for similar requests and improves user experience. <br />This project aims to optimize services by introducing a caching mechanism. It helps businesses and research institutions reduce the cost of inference deployment, improve model performance and efficiency, and provide scalable services for large models.  Through open-source, we aim to share and exchange technologies related to large model semantic cache.

## Architecture

![modelcache modules](docs/modelcache_modules_20240409.png)

## Quick start

You can find the start script in `flask4modelcache.py` and `flask4modelcache_demo.py`.

- `flask4modelcache_demo.py` is a quick test service that embeds sqlite and faiss. You do not need to be concerned about database-related matters.
- `flask4modelcache.py` is the normal service that requires configuration of MySQL and Milvus.

### Dependencies

- Python: V3.8 and above
- Package installation

  ```shell
  pip install -r requirements.txt 
  ```

### Start service

#### Start Demo

1. Download the embedding model bin file on [Hugging Face](https://huggingface.co/shibing624/text2vec-base-chinese/tree/main). Place the downloaded bin file in the model/text2vec-base-chinese folder.
2. Start the backend service by using `flask4modelcache_dome.py`.

  ```shell
  cd CodeFuse-ModelCache
  ```

  ```shell
  python flask4modelcache_demo.py
  ```

#### Start normal service

Before you start normal service, make sure that you have completed these steps:

1. Install the relational database MySQL and import the SQL file to create the data tables. You can find the SQL file in `reference_doc/create_table.sql`.
2. Install vector database Milvus.
3. Add the database access information to the configuration files:
   1. `modelcache/config/milvus_config.ini`
   2. `modelcache/config/mysql_config.ini`
4. Download the embedding model bin file from [Hugging Face](https://huggingface.co/shibing624/text2vec-base-chinese/tree/main). Put the bin file in the `model/text2vec-base-chinese` directory.
5. Start the backend service by using the `flask4modelcache.py` script.

## Access the service

The current service provides three core functionalities through RESTful API.: Cache-Writing, Cache-Querying, and Cache-Clearing. Demos:

### Write cache

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

### Query cache

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

### Clear cache

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

## Function comparison

We've implemented several key updates to our repository. We've resolved network issues with Hugging Face and improved inference speed by introducing local embedding capabilities. Due to limitations in SqlAlchemy, we've redesigned our relational database interaction module for more flexible operations. We've added multi-tenancy support to ModelCache, recognizing the need for multiple users and models in LLM products. Lastly, we've made initial adjustments for better compatibility with system commands and multi-turn dialogues.

<table>
  <tr>
    <th rowspan="2">Module</th>
    <th rowspan="2">Function</th>

  </tr>
  <tr>
    <th>ModelCache</th>
    <th>GPTCache</th>
  </tr>
  <tr>
    <td rowspan="2">Basic Interface</td>
    <td>Data query interface</td>
    <td class="checkmark">&#9745; </td>
    <td class="checkmark">&#9745; </td>
  </tr>
  <tr>
    <td>Data writing interface</td>
    <td class="checkmark">&#9745; </td>
    <td class="checkmark">&#9745; </td>
  </tr>
  <tr>
    <td rowspan="3">Embedding</td>
    <td>Embedding model configuration</td>
    <td class="checkmark">&#9745; </td>
    <td class="checkmark">&#9745; </td>
  </tr>
  <tr>
    <td>Large model embedding layer</td>
    <td class="checkmark">&#9745; </td>
    <td></td>
  </tr>
  <tr>
    <td>BERT model long text processing</td>
    <td class="checkmark">&#9745; </td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="2">Large model invocation</td>
    <td>Decoupling from large models</td>
    <td class="checkmark">&#9745; </td>
    <td></td>
  </tr>
  <tr>
    <td>Local loading of embedding model</td>
    <td class="checkmark">&#9745; </td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="2">Data isolation</td>
    <td>Model data isolation</td>
    <td class="checkmark">&#9745; </td>
    <td class="checkmark">&#9745; </td>
  </tr>
  <tr>
    <td>Hyperparameter isolation</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="3">Databases</td>
    <td>MySQL</td>
    <td class="checkmark">&#9745; </td>
    <td class="checkmark">&#9745; </td>
  </tr>
  <tr>
    <td>Milvus</td>
    <td class="checkmark">&#9745; </td>
    <td class="checkmark">&#9745; </td>
  </tr>
  <tr>
    <td>OceanBase</td>
    <td class="checkmark">&#9745; </td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="3">Session management</td>
    <td>Single-turn dialogue</td>
    <td class="checkmark">&#9745; </td>
    <td class="checkmark">&#9745; </td>
  </tr>
  <tr>
    <td>System commands</td>
    <td class="checkmark">&#9745; </td>
    <td></td>
  </tr>
  <tr>
    <td>Multi-turn dialogue</td>
    <td class="checkmark">&#9745; </td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="2">Data management</td>
    <td>Data persistence</td>
    <td class="checkmark">&#9745; </td>
    <td class="checkmark">&#9745; </td>
  </tr>
  <tr>
    <td>One-click cache clearance</td>
    <td class="checkmark">&#9745; </td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="2">Tenant management</td>
    <td>Support for multi-tenancy</td>
    <td class="checkmark">&#9745; </td>
    <td></td>
  </tr>
  <tr>
    <td>Milvus multi-collection capability</td>
    <td class="checkmark">&#9745; </td>
    <td></td>
  </tr>
  <tr>
    <td>Other</td>
    <td>Long-short dialogue distinction</td>
    <td class="checkmark">&#9745; </td>
    <td></td>
  </tr>
</table>

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

## Todo List

### Adapter

- [ ] Register adapter for Milvus：Based on the "model" parameter in the scope, initialize the corresponding Collection and perform the load operation.

### Embedding model&inference

- [ ] Inference Optimization: Optimizing the speed of embedding inference, compatible with inference engines such as FasterTransformer, TurboTransformers, and ByteTransformer.
- [ ] Compatibility with Hugging Face models and ModelScope models, offering more methods for model loading.

### Scalar Storage

- [ ] Support MongoDB
- [ ] Support ElasticSearch

### Vector Storage

- [ ] Adapts Faiss storage in multimodal scenarios.

### Ranking

- [ ] Add ranking model to refine the order of data after embedding recall.

### Service

- [ ] Supports FastAPI.
- [ ] Add visual interface to offer a more direct user experience.

## Acknowledgements

This project has referenced the following open-source projects. We would like to express our gratitude to the projects and their developers for their contributions and research.<br />[GPTCache](https://github.com/zilliztech/GPTCache)

## Contributing

ModelCache is a captivating and invaluable project, whether you are an experienced developer or a novice just starting out, your contributions to this project are warmly welcomed. Your involvement in this project, be it through raising issues, providing suggestions, writing code, or documenting and creating examples, will enhance the project's quality and make a significant contribution to the open-source community.
