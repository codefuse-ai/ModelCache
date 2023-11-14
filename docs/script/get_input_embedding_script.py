# -*- coding: utf-8 -*-
import torch
import numpy as np
from transformers import AutoModelForCausalLM


model_path = ''
device = torch.device('cuda')
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to(device)
embedding_weights = model.get_input_embeddings().weight.to('cpu').detach().numpy()
np.save('gpt-neox-embedding.npy', embedding_weights)
