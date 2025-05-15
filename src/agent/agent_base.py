# 基础agent类
from typing import Dict, List, Optional, Union, Any, Tuple
from models.base_model import BaseModel
import json
import re
from utils.logger import setup_logger

logger = setup_logger(__name__)

class AgentBase:
    """基础Agent类"""
    
    def __init__(
        self,
        model: BaseModel,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_iterations: int = 5,
        verbose: bool = False,
    ):
        self.model = model
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # 设置系统提示
        self.system_prompt = system_prompt or (
            "你是一个专业的医疗助手，可以回答医疗相关问题并能使用工具来获取更准确的信息。"
            "在回答问题时，请遵循以下步骤：\n"
            "1. 思考问题是否需要使用工具来获取信息\n"
            "2. 如果需要，选择合适的工具并提供准确的参数\n"
            "3. 分析工具返回的结果\n"
            "4. 提供最终回答\n"
            "请确保你的回答是准确的、有帮助的，并基于可靠的医疗知识。"
        )
    
    def register_tool(self, tool: Dict[str, Any]):
        """注册工具"""
        self.tools.append(tool)
    
    def format_tools(self) -> str:
        """格式化工具描述"""
        if not self.tools:
            return "你没有可用的工具。"
        
        tools_desc = "你可以使用以下工具：\n\n"
        
        for i, tool in enumerate(self.tools):
            tools_desc += f"工具 {i+1}: {tool['name']}\n"
            tools_desc += f"描述: {tool['description']}\n"
            
            if tool.get("parameters"):
                tools_desc += "参数:\n"
                for param_name, param_info in tool["parameters"].items():
                    tools_desc += f"  - {param_name}: {param_info['description']}"
                    if param_info.get("required", False):
                        tools_desc += " (必需)"
                    tools_desc += "\n"
            
            tools_desc += "\n"
        
        tools_desc += "当你需要使用工具时，请使用以下格式：\n"
        tools_desc += "```\n工具: 工具名称\n参数: {\"参数1\": \"值1\", \"参数2\": \"值2\", ...}\n```\n"
        
        return tools_desc
    
    def format_prompt(self, user_query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """格式化提示"""
        # 构建系统提示
        full_system_prompt = self.system_prompt
        
        # 添加工具描述
        if self.tools:
            full_system_prompt += "\n\n" + self.format_tools()
        
        # 构建完整提示
        prompt = f"<|im_start|>system\n{full_system_prompt}<|im_end|>\n"
        
        # 添加聊天历史
        if chat_history:
            for message in chat_history:
                if message["role"] == "user":
                    prompt += f"<|im_start|>user\n{message['content']}<|im_end|>\n"
                elif message["role"] == "assistant":
                    prompt += f"<|im_start|>assistant\n{message['content']}<|im_end|>\n"
        
        # 添加用户查询
        prompt += f"<|im_start|>user\n{user_query}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        
        return prompt
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具调用"""
        # 查找匹配的工具
        tool = None
        for t in self.tools:
            if t["name"] == tool_name:
                tool = t
                break
        
        if not tool:
            return {
                "error": f"工具 '{tool_name}' 不存在。",
                "result": f"错误: 工具 '{tool_name}' 不存在。"
            }
        
        # 检查工具是否有执行函数
        if "function" not in tool:
            return {
                "error": f"工具 '{tool_name}' 没有执行函数。",
                "result": f"错误: 工具 '{tool_name}' 配置不正确。"
            }
        
        # 验证参数
        if "parameters" in tool:
            for param_name, param_info in tool["parameters"].items():
                if param_info.get("required", False) and param_name not in parameters:
                    return {
                        "error": f"缺少必需参数 '{param_name}'。",
                        "result": f"错误: 工具 '{tool_name}' 需要参数 '{param_name}'。"
                    }
        
        # 执行工具函数
        try:
            result = tool["function"](**parameters)
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {
                "error": str(e),
                "result": f"执行工具时出错: {str(e)}"
            }
    
    def extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """从文本中提取工具调用"""
        tool_calls = []
        
        # 正则表达式匹配工具调用
        pattern = r"工具:\s*(.*?)\n参数:\s*(.*?)(?=\n```|\Z)"
        
        # 查找所有匹配项
        matches = re.finditer(pattern, text, re.DOTALL)
        
        for match in matches:
            tool_name = match.group(1).strip()
            params_str = match.group(2).strip()
            
            try:
                # 尝试解析参数JSON
                parameters = json.loads(params_str)
                
                tool_calls.append({
                    "name": tool_name,
                    "parameters": parameters
                })
            except json.JSONDecodeError:
                logger.error(f"Failed to parse tool parameters: {params_str}")
                # 尝试更宽松的解析
                params_dict = {}
                param_pairs = params_str.split(",")
                for pair in param_pairs:
                    if ":" in pair:
                        key, value = pair.split(":", 1)
                        params_dict[key.strip().strip('"').strip("'")] = value.strip().strip('"').strip("'")
                
                if params_dict:
                    tool_calls.append({
                        "name": tool_name,
                        "parameters": params_dict
                    })
        
        return tool_calls
    
    def run(self, user_query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """运行Agent"""
        iteration = 0
        current_chat_history = chat_history.copy() if chat_history else []
        final_response = ""
        
        while iteration < self.max_iterations:
            # 生成提示
            prompt = self.format_prompt(user_query, current_chat_history)
            
            if self.verbose:
                logger.debug(f"Prompt:\n{prompt}")
            
            # 获取模型响应
            response = self.model.generate(prompt)
            
            if self.verbose:
                logger.debug(f"Response:\n{response}")
            
            # 提取工具调用
            tool_calls = self.extract_tool_calls(response)
            
            # 如果没有工具调用，完成
            if not tool_calls:
                final_response = response
                break
            
            # 执行工具调用并将结果添加到聊天历史
            for tool_call in tool_calls:
                result = self.execute_tool(tool_call["name"], tool_call["parameters"])
                
                if self.verbose:
                    logger.debug(f"Tool result: {result}")
                
                # 添加工具调用到聊天历史
                current_chat_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                # 添加工具结果到聊天历史
                tool_result_message = (
                    f"工具结果: {tool_call['name']}\n\n"
                    f"{result.get('result', str(result))}"
                )
                
                current_chat_history.append({
                    "role": "user",
                    "content": tool_result_message
                })
                
                # 更新用户查询为工具结果，让模型继续处理
                user_query = tool_result_message
            
            iteration += 1
        
        # 如果达到最大迭代次数但没有最终响应
        if not final_response:
            final_response = "很抱歉，我无法完成这个任务。请尝试重新描述您的问题，或者联系技术支持。"
        
        return {
            "response": final_response,
            "iterations": iteration,
            "chat_history": current_chat_history
        }


# agent/medical_agent.py

from typing import Dict, List, Optional, Union, Any, Tuple
from .agent_base import AgentBase
from models.base_model import BaseModel
from rag.rag_pipeline import RAGPipeline
from utils.logger import setup_logger

logger = setup_logger(__name__)

class MedicalAgent(AgentBase):
    """医疗特定的Agent实现"""
    
    def __init__(
        self,
        model: BaseModel,
        rag_pipeline: Optional[RAGPipeline] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_iterations: int = 5,
        verbose: bool = False,
    ):
        # 设置默认医疗系统提示
        default_system_prompt = (
            "你是一个专业的医疗助手，可以回答医疗相关问题并能使用工具来获取更准确的信息。"
            "在回答医疗问题时，请遵循以下原则：\n"
            "1. 基于可靠的医学知识提供准确的信息\n"
            "2. 不要做出诊断、处方或治疗建议，而是提供一般的医学知识\n"
            "3. 对于紧急情况，建议用户立即就医\n"
            "4. 当你不确定时，坦率承认并鼓励用户咨询专业医生\n"
            "5. 提供实用的一般健康信息和预防措施\n"
            "请用清晰、易懂的语言回答问题，避免使用过多的专业术语。"
        )
        
        super().__init__(
            model=model,
            system_prompt=system_prompt or default_system_prompt,
            tools=tools,
            max_iterations=max_iterations,
            verbose=verbose,
        )
        
        self.rag_pipeline = rag_pipeline
        
        # 注册医疗知识检索工具
        if self.rag_pipeline:
            self.register_medical_search_tool()
    
    def register_medical_search_tool(self):
        """注册医疗知识检索工具"""
        def search_medical_knowledge(query: str, top_k: int = 3):
            """搜索医疗知识库"""
            if not self.rag_pipeline:
                return "医疗知识库未初始化。"
            
            retrieved_docs = self.rag_pipeline.query(query, top_k)
            
            if not retrieved_docs:
                return "未找到相关医疗信息。"
            
            result = "找到以下相关医疗信息：\n\n"
            for i, doc in enumerate(retrieved_docs):
                result += f"[文档 {i+1}] (相关度: {doc.get('score', 0):.2f}):\n"
                result += f"{doc['text']}\n\n"
            
            return result
        
        medical_search_tool = {
            "name": "搜索医疗知识",
            "description": "搜索医疗知识库获取相关信息",
            "parameters": {
                "query": {
                    "description": "搜索查询",
                    "required": True
                },
                "top_k": {
                    "description": "返回的最大结果数",
                    "required": False
                }
            },
            "function": search_medical_knowledge
        }
        
        self.register_tool(medical_search_tool)
    
    def process_with_rag(self, user_query: str) -> Dict[str, Any]:
        """使用RAG处理用户查询"""
        if not self.rag_pipeline:
            logger.warning("RAG pipeline not initialized, falling back to regular processing")
            return self.run(user_query)
        
        # 构建RAG提示
        rag_prompt = self.rag_pipeline.build_rag_prompt(user_query)
        
        # 添加RAG上下文到用户查询
        enhanced_query = (
            f"请回答以下医疗问题。我已经提供了一些相关的医疗信息供你参考。\n\n"
            f"相关医疗信息：\n{rag_prompt['context']}\n\n"
            f"问题：{user_query}"
        )
        
        # 运行Agent
        result = self.run(enhanced_query)
        
        # 添加RAG元数据
        result["rag_context"] = rag_prompt["context"]
        result["original_query"] = user_query
        
        return result

