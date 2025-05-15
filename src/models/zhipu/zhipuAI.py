
import os
from zhipu import ZhipuAI



class ZhipuAPIModel():
    def __init__(self, system_prompt:str=None):
        self.client = ZhipuAI(api_key=os.environ.get("zhipuApiKey")) # 填写您自己的APIKey
        
        self.system_prompt = system_prompt
        
        
    def generate(self, prompt)->str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model="glm-4-flash",  # 填写需要调用的模型名称
            messages=messages,
        )
        print(response.choices[0].message.content)
        
        return response.choices[0].message.content