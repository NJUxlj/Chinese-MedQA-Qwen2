from llmtuner import ChatModel
from llmtuner.extras.misc import torch_gc
import requests
from lxml import etree 


try:
    import platform
    if platform.system()!="Windows":
        import readline
except ImportError:
    print("Install `readline` for a better experience.")
    
    



class RAGFlow():
    pass
    
    
    
    
def build_prompt(query):
    try:
        # search api
        url = f"https://www.baidu.com/s?tn=15007414_5_dg&ie=utf-8&wd={query}"
        
        
        # send get request and get response
        response = requests.get(url)
        content = response.text
        
        # 使用lxml解析HTML  
        
        
    
    
    
    except Exception as e:
        print(e)
        answer_texts = "无相似回答"
        
        
        
    prompt = ""
    
    return prompt
        






def main():
    pass










if __name__ == "__main__":
    main()