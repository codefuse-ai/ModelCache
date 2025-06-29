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
  - [Running the service](#running-the-service)
    - [Demo service](#demo-service)
    - [Standard service](#standard-service)
- [Using the service](#using-the-service)
  - [Write cache](#write-cache)
  - [Query cache](#query-cache)
  - [Clear cache](#clear-cache)
- [Function comparison](#function-comparison)
- [Features](#features)
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
- 🔥🔥[2025.06.28] Added a Websocket-based API, memory cache, multiprocessing-based embedding with configurable amount of workers, bulk-insert support in the backend, python 12 support and massive performance improvements
- 🔥🔥[2024.10.22] Added tasks for 1024 developer day.
- 🔥🔥[2024.04.09] Added Redis Search to store and retrieve embeddings in multi-tenant. This can reduce the interaction time between Cache and vector databases to 10ms.
- 🔥🔥[2023.12.10] Integrated LLM embedding frameworks such as 'llmEmb', 'ONNX', 'PaddleNLP', 'FastText', and the image embedding framework 'timm' to bolster embedding functionality.
- 🔥🔥[2023.11.20] Integrated local storage, such as sqlite and faiss. This enables you to initiate quick and convenient tests.
- [2023.08.26] codefuse-ModelCache...

### Introduction

Codefuse-ModelCache is a standalone semantic cache for large language models (LLMs).\
By caching pre-generated model results, it reduces response time for similar requests and improves user experience. <br />This project aims to optimize services by introducing a caching mechanism. It helps businesses and research institutions reduce the cost of inference deployment, improve model performance and efficiency, and provide scalable services for large models.  Through open-source, we aim to share and exchange technologies related to large model semantic cache.

## Architecture

![modelcache modules](docs/modelcache_modules_20240409.png)

# Quick start

You can find the start scripts at the root of the repository.\
There are standard services that require MySQL and Milvus configuration, and there are quick test services that use SQLite and FAISS (No database configuration required).\
The quick test services have `_demo` at the end of the file name

### Dependencies

- Python: V3.8 or above
- Package installation

  ```shell
  pip install -r requirements.txt 
  ```

## Running the service

### Demo service
Navigate to the root of the repository and run one of the following:
- `python flask4modelcache_demo.py`
- `python fastapi4modelcache_demo.py`
- `python websocket4modelcache_demo.py`

### Standard service

You can choose to run the databases via docker-compose or installing them manually onto your machine

#### Starting databases using docker-compose
Navigate to the root of the repository and run
```shell
docker-compose up -d
```
#### Manual databases insall
1. Install MySQL and import the SQL file from `reference_doc/create_table.sql`.
2. Install vector database Milvus.
3. Configure database access in:
    - `modelcache/config/milvus_config.ini`
    - `modelcache/config/mysql_config.ini`

\-\-\-\-\-\-\-\-\-\-\-\-

After installing and running the databases, start a backend service of your choice
- `python flask4modelcache_demo.py`
-  `python fastapi4modelcache_demo.py`
- `python websocket4modelcache_demo.py`

## Using the service

The service provides three core functionalities: Cache-Writing, Cache-Querying, and Cache-Clearing.\
The service supports both a RESTful API and Websocket API

RESTful API - `flask4modelcache.py` and `fastapi4modelcache.py`\
Websocket API - `websocket4modelcache.py`

### RESTful API
#### Write cache

```json
{
  "type": "insert",
  "scope": {
    "model": "CODEGPT-1008"
  },
  "chat_info": [
    {
      "query": [
        {
          "role": "user",
          "content": "Who are you?"
        },
        {
          "role": "system",
          "content": "You are an AI code assistant and you must provide neutral and harmless answers to help users solve code-related problems."
        }
      ],
      "answer": "Hello, I am an intelligent assistant. How can I assist you?"
    }
  ]
}
```
Code example
```python
import json
import requests
url = 'http://127.0.0.1:5000/modelcache'
type = 'insert'
scope = {"model": "CODEGPT-1008"}
chat_info = [{"query": [{"role": "system", "content": "You are an AI code assistant and you must provide neutral and harmless answers to help users solve code-related problems."}, {"role": "user", "content": "Who are you?"}],"answer": "Hello, I am an intelligent assistant. How can I assist you?"}]
data = {'type': type, 'scope': scope, 'chat_info': chat_info}

headers = {"Content-Type": "application/json"}
res = requests.post(url, headers=headers, json=json.dumps(data))
```

\-\-\-\-\-\-\-\-\-\-\-\-

#### Query cache
```json
{
  "type": "query",
  "scope": {
    "model": "CODEGPT-1008"
  },
  "query": [
    {
      "role": "user",
      "content": "Who are you?"
    },
    {
      "role": "system",
      "content": "You are an AI code assistant and you must provide neutral and harmless answers to help users solve code-related problems."
    }
  ]
}
```
Code example
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

\-\-\-\-\-\-\-\-\-\-\-\-

#### Clear cache
```json
{
  "type": "remove",
  "scope": {
    "model": "CODEGPT-1008"
  },
  "remove_type": "truncate_by_model"
}
```
Code example
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

### Websocket API

The websocket API is inherently asynchronous, so we need to wrap the request with a request id in order to be able to track it.\
The service will return a response with the appropriate request id that was given for the request

#### Write cache
```json
{
  "requestId": "943e9450-3467-4d73-9b32-68a337691f6d",
  "payload": {
    "type": "insert",
    "scope": {
      "model": "CODEGPT-1008"
    },
    "chat_info": [
      {
        "query": [
          {
            "role": "user",
            "content": "Who are you?"
          },
          {
            "role": "system",
            "content": "You are an AI code assistant and you must provide neutral and harmless answers to help users solve code-related problems."
          }
        ],
        "answer": "Hello, I am an intelligent assistant. How can I assist you?"
      }
    ]
  }
}
```

#### Query cache
```json
{
  "requestId": "51f00484-acc9-406f-807d-29fba672473e",
  "payload": {
    "type": "query",
    "scope": {
      "model": "CODEGPT-1008"
    },
    "query": [
      {
        "role": "user",
        "content": "Who are you?"
      },
      {
        "role": "system",
        "content": "You are an AI code assistant and you must provide neutral and harmless answers to help users solve code-related problems."
      }
    ]
  }
}
```

#### Clear cache
```json
{
  "requestId": "f96bbc87-5ef9-4161-9e96-3076ca97b4b9",
  "payload": {
    "type": "remove",
    "scope": {
      "model": "CODEGPT-1008"
    },
    "remove_type": "truncate_by_model"
  }
}
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

## Features

In ModelCache, we incorporated the core principles of GPTCache.\
ModelCache has four modules: adapter, embedding, similarity, and data_manager.

- The adapter module orchestrates the business logic for various tasks, integrate the embedding, similarity, and data_manager modules.
- The embedding module converts text into semantic vector representations, and transforms user queries into vectors.
- The rank module ranks and evaluate the similarity of recalled vectors.
- The data_manager module manages the databases.

To make ModelCache more suitable for industrial use, we made several improvements to its architecture and functionality:

- [x] Architectural adjustment (lightweight integration): 
  - Embedded into LLM products using a Redis-like caching mode
  - Provided semantic caching without interfering with LLM calls, security audits, and other functions
  - Compatible with all LLM services
- [x] Multiprocessing-based embedding:
    - True parallel embedding, serving multiple requests at once
    - Highly scalable, supports configuring the amount of embedding worker.
    - Enables efficient use of available computing resources
- [x] Multiple model loading:
  - Supported local embedding model loading, and resolved Hugging Face network connectivity issues
  - Supported loading embedding layers from various pre-trained models
- [x] Data isolation
  - Environment isolation: Read different database configurations based on the environment. Isolate  development, staging, and production environments.
  - Multi-tenant data isolation: Dynamically create collections based on models for data isolation, addressing data separation issues in multi-model/service scenarios within large language model products
- [x] Supported system instruction: Adopted a concatenation approach to resolve issues with system instructions in the prompt paradigm.
- [x] Long and short text differentiation: Long texts bring more challenges for similarity assessment. Added differentiation between long and short texts, allowing for separate threshold configurations.
- [x] Milvus performance optimization: Adjusted Milvus consistency level to "Session" level for better performance.
- [x] Data management:
  - One-click cache clearing to enable easy data management after model upgrades.
  - Recall of hit queries for subsequent data analysis and model iteration reference.
  - Asynchronous log write-back for data analysis and statistics
  - Added model field and data statistics field to enhance features

## Todo List

### Adapter

- [x] Register adapter for Milvus：Based on the "model" parameter in the scope, initialize the corresponding Collection and perform the load operation.

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

- [x] Supports FastAPI.
- [ ] Add visual interface to offer a more direct user experience.

## Acknowledgements

This project has referenced the following open-source projects. We would like to express our gratitude to the projects and their developers for their contributions and research.<br />[GPTCache](https://github.com/zilliztech/GPTCache)

## Contributing

ModelCache is a captivating and invaluable project, whether you are an experienced developer or a novice just starting out, your contributions to this project are warmly welcomed. Your involvement in this project, be it through raising issues, providing suggestions, writing code, or documenting and creating examples, will enhance the project's quality and make a significant contribution to the open-source community.
