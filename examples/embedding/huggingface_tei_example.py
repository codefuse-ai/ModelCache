# -*- coding: utf-8 -*-
import sys
sys.path.append(".")
from modelcache.embedding.huggingface_tei import HuggingfaceTEI

'''
run tei server:
text-embeddings-router --model-id BAAI/bge-large-zh-v1.5 --port 8080
'''

def run():
    tei_instance = HuggingfaceTEI('http://127.0.0.1:8080/v1/embeddings', 'BAAI/bge-large-zh-v1.5')
    print('dimenson', tei_instance.dimension)
    print('embedding', tei_instance.to_embeddings('hello'))

if __name__ == '__main__':
    run()