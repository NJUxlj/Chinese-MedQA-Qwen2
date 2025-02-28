import os
import platform
from transformers import AutoTokenizer, AutoModel
import torch
from fastllm_pytools import llm
import readline
import requests  
from lxml import etree  

from config.config import MODEL_PATH, TOKENIZER_PATH, DEVICE


# find model weight files that ends with .flm
model_flm = [f for f in os.listdir(MODEL_PATH) if f.endswith(".flm")]
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)


if len(model_flm) == 0:
    print("No model weight files found, please check the model path")
else:
    if "cuda" in DEVICE:
        # doing 4-bit quantization to model
        # model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE).quantize(4)
        model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE)
    else:
        # CPU  or Intel GPU that can use Float 16
        model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).float().to(DEVICE)

    model = llm.from_hf(model, tokenizer, dtyped = "float16") # float16 or int8 or int4
    # model.save(os.path.join(MODEL_PATH, "model.flm"))
    


os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

welcome_prompt = "欢迎使用中医聊天机器人，使用 clear 命令可清除聊天历史，使用 exit 命令可退出应用程序。"




def main():
    pass


if __name__ == "__main__":
    main()