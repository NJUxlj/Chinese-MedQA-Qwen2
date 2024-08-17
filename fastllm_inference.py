import os


import platform

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

import torch


# 获取当前脚本的上级目录
parent_path = os.path.abspath(os.path.dirname(__file__))

parent_path = os.path.join(parent_path, "..")

fastllm_path = os.path.join(parent_path, "fastllm/tools")

import sys

sys.path.append(fastllm_path)
from fastllm_pytools import llm



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 这两个路径是同一个值
MODEL_PATH = "/mnt/data/models/qwen2-7b-instruct"

TOKENIZER_PATH = "/mnt/data/models/qwen2-7b-instruct"

model_flm = []


if 'cuda' in DEVICE:
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(device).quantize(4)

else:

    # 非英伟达GPU
    model = AutoModel.from_pretrained(Model_PATH, trust_remote_code=True).float().to(device)


# 转为flm格式


os_name = platform.system()

clear_command = None
if os.name == 'Windows':
    clear_command = 'cls'
else:
    clear_command = "clear"


welcome_prompt = "欢迎使用医疗问答机器人；"\
    "\n 使用 clear 命令可清除聊天历史；"\
        "\n 使用 exit 命令可退出应用程序。"


stop_stream = False


def main():
    past_key_values, history = None, []

    history = [{
        "role": "user",
        "content": '现在你是一名专业的中医医生，请用你的专业知识提供详尽而清晰的关于中医问题的回答。'
    },{
        "role":"assistant",
        "content":'当然，我将尽力为您提供关于中医的详细而清晰的回答。请问您有特定的中医问题或主题感兴趣吗？您可以提出您想了解的中医相关问题，比如中医理论、诊断方法、治疗技术、中药等方面的问题。我将根据您的需求提供相应的解答。'
    }
    ]


    global stop_stream
    print(welcome_prompt)

    while True:
        query = input("\n患者: ")

        if query.strip() == "clear":
            past_key_values, history = None, []
            os.system(clear_command)
            print(welcome_prompt)
            continue
        
        if query.strip() == "exit":
            break


        print()
        print()
        print("Cyber Doctor:", end = "")

        resp_last_turn_len = 0

        for response, history, past_key_values in model.stream_chat(tokenizer, query, history, 
                                        top_p = 1, temperature = 0.5, 
                                        past_key_values = past_key_values,
                                        return_past_key_values=True):
                
            if stop_stream:
                stop_stream = False
                break

            else:
                print(str(response[resp_last_turn_len:]))
                resp_last_turn_len += len(response)
        









if __name__ == '__main__':
    main()