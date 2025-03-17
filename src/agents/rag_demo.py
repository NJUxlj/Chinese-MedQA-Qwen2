from llmtuner import ChatModel
from llmtuner.extras.misc import torch_gc
import requests
from lxml import etree 

from model.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from config.config import MODEL_PATH

from prompt_template import PromptTemplate


try:
    import platform
    if platform.system()!="Windows":
        import readline
except ImportError:
    print("Install `readline` for a better experience.")
    
    



class RAGFlow():
    
    def __init__(self, use_hf_model = True, use_prompt_template = True):
        self.use_hf_model = use_hf_model
        self.use_prompt_template = use_prompt_template
        
        if use_hf_model:
            self.chat_model = Qwen2ForCausalLM(MODEL_PATH)
        else:
            self.chat_model = ChatModel()
        
        if self.use_prompt_template:
            self.prompt_template = PromptTemplate()
            
            
        self.db = None # database is temporarily not implemented
    
    
    
    def build_rag_prompt(self):
        # 生成工具调用提示  
        prompt = prompt_template.generate_prompt(query, history)  
        
        # 获取模型输出（假设model为LLM实例）  
        tool_call_str = model.generate(prompt)  
        
        # 执行工具调用  
        raw_result = dispatcher.execute(tool_call_str)  
        
        # 结果处理  
        summary = summarize_results(raw_result)  
        
        # 生成最终回答  
        final_prompt = f"搜索结果：\n{summary}\n\n基于以上信息回答问题：{query}"  
        return model.generate(final_prompt) 
            
    def rag_chat(self, query:str):
        
        history = []
        print("欢迎使用中医聊天机器人，使用 clear 命令可清除聊天历史，使用 exit 命令可退出应用程序。")
        
        
        while True:
            try:
                query = input("\n患者：")
            except UnicodeDecodeError:
                print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
                continue
            except Exception:
                raise
            
            if query.strip() == "exit":
                break

            if query.strip() == "clear":
                history = []
                torch_gc()
                print("History has been removed.")
                continue
            
            print("医师: ", end="", flush=True)
            query = build_rag_prompt(query)
            response = ""
            
            
            if self.use_hf_model:
                for new_token in self.chat_model.generate(query, history):
                    pass
            else:
                for new_token in self.chat_model.stream_chat(query, history):
                    print(new_token, end="", flush=True)
                    response+=new_token
            print()
            
            history += [(query, response)]

    
    
    
    
def build_prompt(query):
    try:
        # search api
        url = f"https://www.baidu.com/s?tn=15007414_5_dg&ie=utf-8&wd={query}"
        
        
        # send get request and get response
        response = requests.get(url)
        content = response.text
        
        # 使用lxml解析HTML  
        html = etree.HTML(content)  
        
        answers = html.xpath("/html/body/div[2]/div/div/b-superframe-body/div/div[2]/div/div/article/section/section/div/div/a/div[2]/text()")[:3]
        
        if len(answers)==0:
            answer_texts = "无相似回答"
        else:
            answer_texts = {f"相似回答 {i+1}": answer for i, answer in enumerate(answers)}
    
    
    
    except Exception as e:
        print(e)
        answer_texts = "无相似回答"
        
        
    # construct final prompt
    prompt = prompt = f'现在你是一名专业的中医医生，请回答以下患者的问诊问题：“{query}"。 \
                            这里有一些相似的回答可能会帮助到你，需要注意的是，在你提供的答案中，请以你的中医知识为主，相似回答仅作为参考。相似回答：{answer_texts}'
    
    return prompt
        






def main():
    chat_model = ChatModel()
    history = []
    print("欢迎使用中医聊天机器人，使用 clear 命令可清除聊天历史，使用 exit 命令可退出应用程序。")
    
    
    while True:
        try:
            query = input("\n患者：")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise
        
        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            history = []
            torch_gc()
            print("History has been removed.")
            continue
        
        print("医师: ", end="", flush=True)
        query = build_prompt(query)
        response = ""
        
        for new_token in chat_model.stream_chat(query, history):
            print(new_token, end="", flush=True)
            response+=new_token
        print()
        
        history += [(query, response)]










if __name__ == "__main__":
    main()