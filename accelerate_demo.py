import os
import platform
from transformers import AutoTokenizer, AutoModel
import torch
from fastllm_pytools import llm
import readline
import requests  
from lxml import etree  

from config.config import MODEL_PATH, TOKENIZER_PATH, DEVICE


# 寻找以 .flm 结尾的文件
model_flm = [f for f in os.listdir(MODEL_PATH) if f.endswith(".flm")]
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)a