import math
import torch
import os
import sys
sys.path.append("/home/grads/h/han/workspace/flash-attention-x")

import warnings
warnings.filterwarnings("ignore")

import torch 
import json
import time
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache
import torch.nn as nn


# model_name = 'meta-llama/Llama-2-7b-chat-hf'
model_name = '/data/hf_models/Llama-2-7b-hf'
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16, use_flash_attention_2=False).cuda()
model.eval()


input_text = "What is the capital of France?"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
input_ids = input_ids.cuda()

# Run the model
import time
start_time = time.time()
with torch.no_grad():
    tokens = model.generate(input_ids, max_new_tokens=200, do_sample=False)
    output_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
runing_time = time.time() - start_time

print(output_text)
print("Running time: ", runing_time)