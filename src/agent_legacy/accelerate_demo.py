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
    model = llm.model("model.flm") # 导入fastllm模型
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
'''
在一个实时交互的聊天系统中，用户可能希望随时停止当前的回答输出，而无需等待整个回答完成。通过设置 stop_stream，可以实现这一需求。

stop_stream 是通过 global 声明的全局变量，因此可以在其他函数或线程中修改它的值，从而实现跨函数或线程的控制。
'''

welcome_prompt = "欢迎使用中医聊天机器人，使用 clear 命令可清除聊天历史，使用 exit 命令可退出应用程序。"




def main():
    past_key_values, history = None, []
    history = [
            (
            '现在你是一名专业的中医医生，请用你的专业知识提供详尽而清晰的关于中医问题的回答。', 
            '当然，我将尽力为您提供关于中医的详细而清晰的回答。请问您有特定的中医问题或主题感兴趣吗？您可以提出您想了解的中医相关问题，比如中医理论、诊断方法、治疗技术、中药等方面的问题。我将根据您的需求提供相应的解答。'
            )
        ]

    global stop_stream
    print(welcome_prompt)
    while True:
        query = input("\n患者：")
        if query.strip() == "exit":
            break
        
        if query.strip() == 'clear':
            past_key_values = None
            history = []
            os.system(clear_command)
            print(welcome_prompt)
            continue
        
        print("\n医师：", end="")
        current_length = 0
        
        for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history, top_p=1,
                                        temperature=0.01, past_key_values=past_key_values,
                                            return_past_key_values=True):
            if stop_stream:
                stop_stream = False
                break
            else:
                print(str(response[current_length:]), end = "", flush=True)
                current_length = len(response)
        print("")

if __name__ == "__main__":
    main()