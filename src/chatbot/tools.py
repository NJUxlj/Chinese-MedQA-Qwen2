import requests  
import json  
import re  
from typing import Dict, Any  
import os

'''


API配置准备

1. 访问Google Cloud Console创建项目 ： https://console.cloud.google.com/welcome?inv=1&invt=AbqxIg&project=perfect-obelisk-432607-u7
2. 启用Custom Search API : https://console.cloud.google.com/apis/library/customsearch.googleapis.com?inv=1&invt=AbqxIg&project=perfect-obelisk-432607-u7
3. 获取API密钥和搜索引擎ID（需预先配置）

执行流程：
模型输出工具调用 → 解析工具字符串 → 匹配执行器 → 调用API → 格式化结果 → 返回给LLM  
响应示例结构

json
{  
  "items": [  
    {  
      "title": "华为Mate60和iPhone15全面对比：参数/价格/影像 - 中关村在线",  
      "link": "https://detail.zol.com.cn/...",  
      "snippet": "华为Mate60搭载麒麟9000S芯片，支持卫星通信...iPhone15采用A16仿生芯片..."  
    },  
    {  
      "title": "iPhone 15 vs Huawei Mate 60: 旗舰手机对比 - GSMArena",  
      "link": "https://www.gsmarena.com/...",  
      "snippet": "屏幕尺寸：6.1\" vs 6.7\" • 摄像头：48MP vs 50MP..."  
    }  
  ]  
}  



'''

# CUSTOM_SEARCH_API_KEY = os.environ.get("CUSTOM_SEARCH_API_KEY")

CUSTOM_SEARCH_API_KEY = ""

CUSTOM_SEARCH_ENGINE_ID = ""

os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['all_proxy'] = 'socks5://127.0.0.1:7890'


class GoogleSearchExecutor:  
    def __init__(self, api_key: str, search_engine_id: str):  
        self.base_url = "https://www.googleapis.com/customsearch/v1"  
        self.api_key = api_key  
        self.search_engine_id = search_engine_id  

    def execute(self, query: str, max_results: int = 5) -> Dict[str, Any]:  
        params = {  
            "key": self.api_key,  
            "cx": self.search_engine_id,  
            "q": query,  
            "num": max_results  
        }  
        
        try:  
            response = requests.get(self.base_url, params=params)  
            response.raise_for_status()  
            return self._parse_results(response.json())  
        except Exception as e:  
            return {"error": str(e)}  

    def _parse_results(self, data: Dict) -> Dict:  
        """解析Google API响应"""  
        return {  
            "items": [{  
                "title": item.get("title"),  
                "link": item.get("link"),  
                "snippet": item.get("snippet")  
            } for item in data.get("items", [])]  
        }  
        
        





class ToolDispatcher:  
    def __init__(self):  
        self.executors = {  
            "google_search": GoogleSearchExecutor(  
                api_key=CUSTOM_SEARCH_API_KEY,  
                search_engine_id=CUSTOM_SEARCH_ENGINE_ID  
            )  
        }  
    
    def parse_tool_call(self, tool_str: str) -> Dict:  
        """解析工具调用字符串"""  
        pattern = r"(\w+)\((.*)\)"  
        match = re.match(pattern, tool_str)  
        if not match:  
            return None  
        
        tool_name = match.group(1)  
        args_str = match.group(2)  
        
        # 解析参数键值对  
        args = {}  
        for pair in re.findall(r"(\w+)=([^,]+)", args_str):  
            key = pair[0]  
            value = pair[1].strip("'")
            if re.match(r'^-?\d+$', value):  # 支持负整数
                value = int(value)
            args[key] = value  
        
        return {"tool": tool_name, "args": args}  

    def execute(self, tool_call: str) -> Dict:  
        """执行工具调用"""  
        parsed = self.parse_tool_call(tool_call)  
        if not parsed:  
            return {"error": "Invalid tool format"}  
        
        executor = self.executors.get(parsed["tool"])  
        if not executor:  
            return {"error": "Tool not registered"}  
        
        print( "parse_args = ", parsed["args"])
        
        # parsed["args"] = {"query":..., "max_results":...}
        return executor.execute(query = parsed["args"]["query"], max_results=parsed["args"]["max_results"]) 
    
    
    

if __name__ == "__main__":
    dispatcher = ToolDispatcher()
    model_output = "google_search(query='iPhone15 vs Huawei Mate60', max_results=3)"
    result = dispatcher.execute(model_output)
    print(json.dumps(result, indent=2, ensure_ascii=False))  

'''
# 使用示例  
dispatcher = ToolDispatcher()  

# 模型输出示例  
model_output = "google_search(query='iPhone15 vs Huawei Mate60', max_results=3)"  

# 执行调用  
result = dispatcher.execute(model_output)  
print(json.dumps(result, indent=2, ensure_ascii=False))  

'''