from typing import Dict, List, Optional  
from pydantic import BaseModel  

class ToolParameter(BaseModel):  
    """工具参数规范"""  
    name: str  
    type: str  
    description: str  
    required: bool = True  

class ToolTemplate:  
    def __init__(self,   
                 name: str,  
                 description: str,  
                 parameters: List[ToolParameter],  
                 call_template: str):  
        """  
        Args:  
            name: 工具名称（英文标识）  
            description: 自然语言描述  
            parameters: 参数列表  
            call_template: 调用模板，如：google_search(query)  
        """  
        self.name = name  
        self.description = description  
        self.parameters = parameters  
        self.call_template = call_template  

class PromptTemplate:  
    def __init__(self):  
        self.tools: Dict[str, ToolTemplate] = {
            
        }  
        self.base_prompt = """你是一个专业助手，可以使用以下工具："""  
        
        # 预置常用工具  
        self.register_tool(self._build_google_template())  
        self.register_tool(self._build_baidu_template())  
        self.register_tool(self._build_wolfram_template())  
        self.register_tool(self._build_baidu_api_template())
    
    def register_tool(self, tool: ToolTemplate):  
        """注册新工具"""  
        self.tools[tool.name] = tool  
    
    def generate_prompt(self, query: str, history: Optional[List] = None) -> str:  
        """生成完整提示"""  
        tools_desc = "\n".join(  
            [f"{tool.name}: {tool.description}\n参数: {[p.name for p in tool.parameters]}"  
             for tool in self.tools.values()]  
        )  
        
        return f"""{self.base_prompt}  
        
                可用工具列表：  
                {tools_desc}  

                当前对话历史：  
                {history if history else "无"}  

                用户问题：{query}  

                请按照以下格式响应：  
                <思考>分析问题并选择工具</思考>  
                <工具调用>{'{工具名称}'}(参数1=值1, 参数2=值2)</工具调用>"""  

    @classmethod  
    def _build_google_template(cls) -> ToolTemplate:  
        return ToolTemplate(  
            name="google_search",  
            description="Google搜索引擎（实时网络信息）",  
            parameters=[  
                ToolParameter(name="query", type="str",   
                            description="搜索关键词，用英文逗号分隔"),  
                ToolParameter(name="max_results", type="int",  
                            description="返回结果数量", required=False)  
            ],  
            call_template="google_search(query='{query}', max_results={max_results})"  
        )  

    @classmethod  
    def _build_baidu_template(cls) -> ToolTemplate:  
        return ToolTemplate(  
            name="baidu_search",  
            description="百度搜索引擎",  
            parameters=[  
                ToolParameter(name="query", type="str",   
                            description="中文搜索关键词"),  
                ToolParameter(name="region", type="str",  
                            description="地区限定", required=False)  
            ],  
            call_template="baidu_search(query='{query}', region='{region}')"  
        )  
        
    @classmethod  
    def _build_baidu_api_template(cls) -> ToolTemplate:  
        return ToolTemplate(  
            name="baidu_api_search",  
            description="百度搜索引擎（API版）",  
            parameters=[  
                ToolParameter(name="query", type="str",   
                            description="中文搜索关键词"),  
            ],  
            call_template="baidu_api_search(query='{query}')"  
        ) 

    @classmethod  
    def _build_wolfram_template(cls) -> ToolTemplate:  
        return ToolTemplate(  
            name="wolfram_alpha",  
            description="数学计算和事实查询",  
            parameters=[  
                ToolParameter(name="expression", type="str",  
                            description="数学表达式或事实查询")  
            ],  
            call_template="wolfram_alpha(expression='{expression}')"  
        )