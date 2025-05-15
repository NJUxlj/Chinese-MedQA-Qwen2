import requests  
import json  
import re  
from typing import Dict, Any, List
import os

from bs4 import BeautifulSoup 
from baidusearch.baidusearch import search
from urllib.parse import quote_plus  
import time
import random

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

# os.environ['https_proxy'] = 'http://127.0.0.1:7890'
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['all_proxy'] = 'socks5://127.0.0.1:7890'


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
        
        
        
        
class BaiduSearchExecutor:  
    """百度网页版搜索执行器（非官方API）"""  
    def __init__(self, max_retries=3, proxies=None):  
        self.base_url = "https://www.baidu.com/s"  
        self.headers = {  
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",  
            "Accept-Language": "zh-CN,zh;q=0.9"  
        }  
        self.max_retries = max_retries  
        self.timeout = 15  # 增加超时时间 
        
        self.proxies = proxies or {  
            "http": "socks5://user:pass@host:port",  
            "https": "socks5://user:pass@host:port"  
        }  

    def execute(self, query: str, region: str = "全国", page: int = 1) -> dict:  
        params = {  
            "wd": quote_plus(query),  
            "pn": (page-1)*10,  # 百度分页逻辑  
            "rn": 10,  
            "cl": self._get_region_code(region),
            "ie": "utf-8"  # 明确指定编码  
        }  

        for retry in range(self.max_retries):  
            try:  
                
                # 添加随机延迟（1-3秒）  
                time.sleep(random.uniform(1, 3))  
                
                resp = requests.get(  
                    self.base_url,  
                    params=params,  
                    headers=self.headers,  
                    timeout=self.timeout,
                    verify = True, # 关闭SSL验证
                    proxies=None
                )  
                
                # 处理百度反爬机制  
                if resp.status_code == 403:  
                    raise Exception("触发百度反爬机制（403 Forbidden）")  
                
                resp.raise_for_status()  
                print(resp.text)
                return self._parse_html(resp.text)  
            
            except Exception as e:  
                print(f"重试: {str(e)}")
                
                print("\n===========具体错误类型==============")
                
                if isinstance(e, requests.Timeout):
                    print("请求超时")
                    
                elif isinstance(e, requests.ConnectionError):
                    print("连接错误")
                elif isinstance(e, requests.exceptions.SSLError):
                    print("SSL链接错误")
                elif isinstance(e, requests.exceptions.RequestException):
                    print(f"Request 请求异常 (重试 {retry+1}/{self.max_retries})")
                continue  
            
        return {"error": f"超出最大重试次数，搜索失败"}  

    def _get_region_code(self, region: str) -> int:  
        """地区编码映射（示例）"""  
        regions = {  
            "全国": 0,  
            "北京": 131,  
            "上海": 289,  
            "广东": 340  
        }  
        return regions.get(region, 0)  

    def _parse_html(self, html: str) -> dict:  
        soup = BeautifulSoup(html, 'html.parser')  
        results = []  
        
        for item in soup.find_all('div', class_='result'):  
            title_elem = item.find('h3')  
            link_elem = item.find('a', href=True)  
            desc_elem = item.find('div', class_='c-abstract')  

            if all([title_elem, link_elem]):  
                result = {  
                    "title": title_elem.get_text(strip=True),  
                    "link": link_elem['href'],  
                    "snippet": desc_elem.get_text(strip=True) if desc_elem else ""  
                }  
                results.append(result)  
        
        return {"items": results}  






class BaiduApiSearchExecutor:
    def __init__(self,   
                 timeout: int = 10,  
                 max_results: int = 5,  
                 proxy: str = None):  
        """  
        Args:  
            timeout: 请求超时时间（秒）  
            max_results: 最大返回结果数  
            proxy: 代理设置，例如："http://user:pass@host:port"  
        """  
        self.timeout = timeout
        self.max_results = max_results
        self.proxy = proxy
    
    def execute(self, query: str) -> List[Dict]:  
        """执行百度搜索  
        
        Args:  
            query: 搜索关键词  
        
        Returns:  
            {  
                "items": [  
                    {  
                        "title": 结果标题,  
                        "url": 链接,  
                        "abstract": 摘要,  
                    }  
                ]  
            }  
        """  

        response:List[Dict] = search(query, num_results=self.max_results)
        
        return self._parse_results(response)

    
    
    def _parse_results(self, data:List[Dict])->Dict:
        result = {"items":[]}
        for d in data:
            result["items"].append(
                {
                    "title":d["title"],
                    "url":d["url"],
                    "abstract":d["abstract"]
                    
                }
            )
        
        return result
        








class SerpApiExecutor:  
    """SerpAPI通用执行器（支持多引擎）"""  
    def __init__(self, api_key: str):  
        self.base_url = "https://serpapi.com/search"  
        self.api_key = api_key  

    def execute(self,   
               engine: str,   
               query: str,   
               location: str = "China",  
               **kwargs) -> dict:  
        params = {  
            "api_key": self.api_key,  
            "engine": engine,  
            "q": query,  
            "location": location,  
            **kwargs  
        }  

        try:  
            resp = requests.get(self.base_url, params=params)  
            resp.raise_for_status()  
            return self._parse_response(engine, resp.json())  
        except Exception as e:  
            return {"error": str(e)}  

    def _parse_response(self, engine: str, data: dict) -> dict:  
        """统一解析不同引擎的响应"""  
        if engine == "baidu":  
            return self._parse_baidu(data)  
        elif engine == "google":  
            return self._parse_google(data)  
        else:  
            return data  # 返回原始数据  

    def _parse_baidu(self, data: dict) -> dict:  
        items = []  
        for result in data.get("organic_results", []):  
            items.append({  
                "title": result.get("title"),  
                "link": result.get("link"),  
                "snippet": result.get("snippet")  
            })  
        return {"items": items}  

    def _parse_google(self, data: dict) -> dict:  
        items = []  
        for result in data.get("organic_results", []):  
            items.append({  
                "title": result.get("title"),  
                "link": result.get("link"),  
                "snippet": result.get("snippet")  
            })  
        return {"items": items}  






class ToolDispatcher:  
    def __init__(self):  
        self.executors = {  
            "google_search": GoogleSearchExecutor(  
                api_key=CUSTOM_SEARCH_API_KEY,  
                search_engine_id=CUSTOM_SEARCH_ENGINE_ID  
            ),
            "baidu_search": BaiduSearchExecutor(),  
            "baidu_api_search": BaiduApiSearchExecutor(),
            "serpapi": SerpApiExecutor(api_key="YOUR_SERPAPI_KEY")    
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
        return executor.execute(**parsed["args"]) 
    
    
    
    
    def register_serpapi(self):  
        """集成SerpApi（支持多搜索引擎）"""  
        self.executors["web_search"] = SerpApiExecutor(  
            api_key="serpapi_key",  
            engine="google"  # 可切换bing/baidu等  
        )  
    




def summarize_results(results: Dict) -> str:  
    """将原始结果转换为自然语言摘要"""  
    summaries = []  
    for item in results.get("items", []):  
        summaries.append(f"标题：{item['title']}\n摘要：{item['snippet']}")  
    return "\n\n".join(summaries) 





# 在RAG流程中的典型使用  
def rag_cycle(query: str, history: list):  
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
    
    

if __name__ == "__main__":
    dispatcher = ToolDispatcher()
    # # model_output = "google_search(query='iPhone15 vs Huawei Mate60', max_results=3)"
    # # result = dispatcher.execute(model_output)
    result =  dispatcher.execute("baidu_api_search(query='深度学习框架对比')")  
    
    print(result)
    # print(json.dumps(result, indent=2, ensure_ascii=False))  
    
    # executor = BaiduApiSearchExecutor()
    # executor.execute("iPhone16")

'''
# 使用示例  
dispatcher = ToolDispatcher()  

# 模型输出示例  
model_output = "google_search(query='iPhone15 vs Huawei Mate60', max_results=3)"  

# 执行调用  
result = dispatcher.execute(model_output)  
print(json.dumps(result, indent=2, ensure_ascii=False))  


# 百度搜索调用  
baidu_result = dispatcher.execute("baidu_search(query='深度学习框架对比', region='北京')")  

# SerpAPI多引擎调用  
serp_result = dispatcher.execute(  
    "serpapi(engine='baidu', query='华为Mate60', location='Shanghai')"  
)  

'''