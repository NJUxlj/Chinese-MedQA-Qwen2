import time
import json
from typing import Dict, List, Optional, Any, Union, Tuple

from .agent_base import AgentBase
from .tool_manager import ToolManager
from ..models.base_model import BaseModel
from ..rag.rag_pipeline import RAGPipeline
from ..utils.logger import get_logger

logger = get_logger(__name__)

class MedicalAgent(AgentBase):
    """
    医疗Agent实现，基于RAG和大型语言模型
    """
    
    def __init__(
        self, 
        model: BaseModel,
        rag_pipeline: Optional[RAGPipeline] = None,
        agent_id: Optional[str] = None,
        name: str = "医疗助手",
        description: str = "一个能够回答医疗问题的AI助手，使用西医知识进行疾病诊疗。",
        system_prompt: Optional[str] = None,
        max_iterations: int = 5,
        temperature: float = 0.7,
        verbose: bool = False,
    ) -> None:
        """
        初始化医疗Agent
        
        Args:
            model: 语言模型实例
            rag_pipeline: RAG流水线实例，用于检索相关医疗知识
            agent_id: Agent的唯一标识
            name: Agent的名称
            description: Agent的描述
            system_prompt: 系统提示词
            max_iterations: 最大迭代次数
            temperature: 采样温度
            verbose: 是否输出详细日志
        """
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            model_name=model.model_name if hasattr(model, 'model_name') else None,
            max_iterations=max_iterations,
            verbose=verbose
        )
        
        self.model = model
        self.rag_pipeline = rag_pipeline
        self.tool_manager = ToolManager()
        self.temperature = temperature
        
        # 设置系统提示词
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.add_to_memory("system", self.system_prompt)
        
    def _get_default_system_prompt(self) -> str:
        """获取默认的系统提示词"""
        return (
            f"你是{self.name}，{self.description}\n"
            "你的回答必须基于科学医学证据，并且要简洁清晰。\n"
            "当你不确定答案时，请明确说明你的不确定性，不要编造信息。\n"
            "对于需要就医的情况，请建议患者咨询专业医生。\n"
            "你可以使用以下工具来帮助回答问题：\n"
            f"{self.tool_manager.get_tools_description()}"
        )
    
    def add_tool(self, tool) -> None:
        """
        添加工具到Agent
        
        Args:
            tool: 工具实例
        """
        self.tool_manager.add_tool(tool)
        # 更新系统提示词以包含新工具
        self.system_prompt = self._get_default_system_prompt()
        # 更新内存中的系统提示词
        if self.memory and self.memory[0]["role"] == "system":
            self.memory[0]["content"] = self.system_prompt
        else:
            self.memory.insert(0, {"role": "system", "content": self.system_prompt})
    
    def _should_use_tools(self, query: str) -> bool:
        """
        判断是否应该使用工具
        
        Args:
            query: 用户查询
            
        Returns:
            是否应该使用工具
        """
        if not self.tool_manager.has_tools():
            return False
            
        # 构建提示词，让模型判断是否需要使用工具
        tool_decision_prompt = (
            f"用户查询: {query}\n\n"
            f"可用工具: {self.tool_manager.get_tools_description()}\n\n"
            "请判断是否需要使用工具来回答这个问题？只需回答'是'或'否'。"
        )
        
        response = self.model.generate(
            [{"role": "user", "content": tool_decision_prompt}], 
            temperature=0.1,
            max_tokens=5
        )
        
        return "是" in response or "yes" in response.lower()
    
    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        从模型响应中解析工具调用
        
        Args:
            response: 模型响应
            
        Returns:
            工具调用列表
        """
        tool_calls = []
        
        # 寻找工具调用的格式，例如：
        # 调用工具: 工具名称
        # 参数: {...}
        import re
        
        # 匹配模式1: 标准格式
        pattern1 = r"调用工具[:：]\s*([^\n]+)\n参数[:：]\s*({[^}]+})"
        matches1 = re.findall(pattern1, response, re.DOTALL)
        
        # 匹配模式2: JSON格式
        pattern2 = r'```json\n({"tool":[^}]+"name":[^}]+"[^"]+",[^}]+"parameters":{[^}]+}})\n```'
        matches2 = re.findall(pattern2, response, re.DOTALL)
        
        for tool_name, parameters_str in matches1:
            try:
                parameters = json.loads(parameters_str.strip())
                tool_calls.append({
                    "name": tool_name.strip(),
                    "parameters": parameters
                })
            except json.JSONDecodeError:
                logger.warning(f"工具参数JSON解析失败: {parameters_str}")
        
        for json_str in matches2:
            try:
                tool_call = json.loads(json_str)
                tool_calls.append({
                    "name": tool_call.get("name", ""),
                    "parameters": tool_call.get("parameters", {})
                })
            except json.JSONDecodeError:
                logger.warning(f"工具调用JSON解析失败: {json_str}")
        
        return tool_calls
    
    def _execute_tool(self, tool_call: Dict[str, Any]) -> Tuple[bool, str]:
        """
        执行工具调用
        
        Args:
            tool_call: 工具调用信息
            
        Returns:
            (成功标志, 工具执行结果)
        """
        tool_name = tool_call.get("name", "")
        parameters = tool_call.get("parameters", {})
        
        tool = self.tool_manager.get_tool(tool_name)
        if not tool:
            return False, f"找不到名为 '{tool_name}' 的工具"
        
        try:
            result = tool.run(**parameters)
            return True, result
        except Exception as e:
            logger.error(f"工具 '{tool_name}' 执行失败: {str(e)}")
            return False, f"工具 '{tool_name}' 执行失败: {str(e)}"
    
    def _retrieve_medical_knowledge(self, query: str) -> str:
        """
        使用RAG检索相关医疗知识
        
        Args:
            query: 用户查询
            
        Returns:
            检索到的相关医疗知识文本
        """
        if not self.rag_pipeline:
            return ""
        
        try:
            retrieval_results = self.rag_pipeline.retrieve(query)
            if not retrieval_results:
                return ""
            
            formatted_context = "\n\n".join([
                f"文档 {i+1}:\n{doc.page_content}\n来源: {doc.metadata.get('source', '未知')}"
                for i, doc in enumerate(retrieval_results)
            ])
            
            return formatted_context
        except Exception as e:
            logger.error(f"知识检索失败: {str(e)}")
            return ""
    
    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        运行Agent，处理用户查询
        
        Args:
            query: 用户查询
            kwargs: 其他参数
            
        Returns:
            包含响应和元数据的字典
        """
        start_time = time.time()
        self.add_to_memory("user", query)
        
        iteration = 0
        response = None
        metadata = {
            "iterations": 0,
            "tool_calls": [],
            "rag_used": False,
            "retrieved_documents": [],
            "timing": {}
        }
        
        # 检查是否应使用工具
        should_use_tools = self._should_use_tools(query)
        metadata["tool_usage_decision"] = should_use_tools
        
        # 检索相关医疗知识
        rag_start_time = time.time()
        context = ""
        if self.rag_pipeline:
            context = self._retrieve_medical_knowledge(query)
            metadata["rag_used"] = bool(context)
            if context:
                documents = self.rag_pipeline.get_last_retrieval_documents()
                metadata["retrieved_documents"] = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    } for doc in documents
                ]
        metadata["timing"]["rag_retrieval"] = time.time() - rag_start_time
        
        while iteration < self.max_iterations:
            iteration += 1
            metadata["iterations"] = iteration
            
            # 构建提示词
            if context:
                augmented_query = (
                    f"用户查询: {query}\n\n"
                    f"相关医疗知识:\n{context}\n\n"
                    "请基于上述信息回答用户的问题。如果提供的信息不足以回答问题，可以使用工具或基于你的医学知识回答。"
                )
            else:
                augmented_query = query
            
            # 将augmented_query放入内存中作为最新的用户消息
            if iteration > 1:
                self.memory[-1]["content"] = augmented_query
            
            # 生成响应
            generation_start_time = time.time()
            response_text = self.model.generate(
                self.memory,
                temperature=self.temperature,
                **kwargs
            )
            metadata["timing"]["model_generation"] = time.time() - generation_start_time
            
            # 如果不使用工具或者已达到最大迭代次数，直接返回响应
            if not should_use_tools or iteration == self.max_iterations:
                response = response_text
                self.add_to_memory("assistant", response)
                break
            
            # 解析工具调用
            tool_calls = self._parse_tool_calls(response_text)
            
            if not tool_calls:
                # 没有工具调用，直接返回响应
                response = response_text
                self.add_to_memory("assistant", response)
                break
            
            # 执行工具调用
            tool_results = []
            for tool_call in tool_calls:
                metadata["tool_calls"].append(tool_call)
                success, result = self._execute_tool(tool_call)
                tool_name = tool_call.get("name", "未知工具")
                
                tool_results.append({
                    "tool": tool_name,
                    "success": success,
                    "result": result
                })
            
            # 组织工具执行结果作为新的用户输入
            tools_response = "工具执行结果:\n"
            for result in tool_results:
                tools_response += f"工具: {result['tool']}\n"
                tools_response += f"执行状态: {'成功' if result['success'] else '失败'}\n"
                tools_response += f"结果: {result['result']}\n\n"
            
            # 将工具执行结果添加到内存
            self.add_to_memory("user", tools_response)
        
        metadata["timing"]["total"] = time.time() - start_time
        
        return {
            "query": query,
            "response": response,
            "metadata": metadata
        }
    
    def reset(self) -> None:
        """重置Agent状态，但保留系统提示词"""
        system_prompt = None
        if self.memory and self.memory[0]["role"] == "system":
            system_prompt = self.memory[0]["content"]
        
        self.clear_memory()
        
        if system_prompt:
            self.add_to_memory("system", system_prompt)