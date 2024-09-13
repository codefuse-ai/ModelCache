# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from modelcache.embedding.base import BaseEmbedding


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class Data2VecAudio(BaseEmbedding):
    def __init__(self, model):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        model_dir = os.path.dirname(parent_dir)
        model_path = os.path.join(model_dir, model)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = BertModel.from_pretrained(model_path, local_files_only=True)

        try:
            self.__dimension = self.model.config.hidden_size
        except Exception:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model)
            self.__dimension = config.hidden_size

    def to_embeddings(self, data, **_):
        encoded_input = self.tokenizer(data, padding=True, truncation=True, return_tensors='pt')
        num_tokens = sum(map(len, encoded_input['input_ids']))

        if num_tokens <= 512:
            with torch.no_grad():
                encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                model_output = self.model(**encoded_input)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = sentence_embeddings.squeeze(0).detach().cpu().numpy()
            embedding_array = np.array(sentence_embeddings).astype("float32")
            return embedding_array
        else:
            window_size = 510
            start = 0
            input_ids = encoded_input['input_ids']
            input_ids = input_ids[:, 1:-1]
            start_token = self.tokenizer.cls_token
            end_token = self.tokenizer.sep_token
            start_token_id = self.tokenizer.convert_tokens_to_ids(start_token)
            end_token_id = self.tokenizer.convert_tokens_to_ids(end_token)
            begin_element = torch.tensor([[start_token_id]])
            end_element = torch.tensor([[end_token_id]])

            embedding_array_list = list()
            while start < num_tokens:
                # Calculate the ending position of the sliding window.
                end = start + window_size
                # If the ending position exceeds the length, adjust it to the length.
                if end > num_tokens:
                    end = num_tokens
                # Retrieve the data within the sliding window.
                input_ids_window = input_ids[:, start:end]
                # Insert a new element at position 0.
                input_ids_window = torch.cat([begin_element, input_ids_window[:, 0:]], dim=1)
                # Insert a new element at the last position.
                input_ids_window = torch.cat([input_ids_window, end_element], dim=1)
                input_ids_window_length = sum(map(len, input_ids_window))
                token_type_ids = torch.tensor([[0] * input_ids_window_length])
                attention_mask = torch.tensor([[1] * input_ids_window_length])

                # Concatenate new input_ids
                encoded_input_window = {'input_ids': input_ids_window, 'token_type_ids': token_type_ids,
                                        'attention_mask': attention_mask}
                with torch.no_grad():
                    encoded_input_window = {k: v.to(self.device) for k, v in encoded_input_window.items()}
                    model_output_window = self.model(**encoded_input_window)

                sentence_embeddings_window = mean_pooling(model_output_window, encoded_input_window['attention_mask'])
                sentence_embeddings_window = sentence_embeddings_window.squeeze(0).detach().cpu().numpy()
                embedding_array_window = np.array(sentence_embeddings_window).astype("float32")
                embedding_array_list.append(embedding_array_window)
                start = end

            embedding_array = np.mean(embedding_array_list, axis=0)
            return embedding_array

    def post_proc(self, token_embeddings, inputs):
        attention_mask = inputs["attention_mask"]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sentence_embs = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sentence_embs

    @property
    def dimension(self):
        """Embedding dimension.

        :return: embedding dimension
        """
        return self.__dimension
