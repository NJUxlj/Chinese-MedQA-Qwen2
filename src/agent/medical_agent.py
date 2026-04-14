import time
import json
import re
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).parent.parent))

from agent.base_agent import BaseAgent
from tools.tool_manager import ToolManager
from models.api_model import ApiModel
from rag.rag_pipeline import RAGPipeline
from utils.logger import setup_logger

logger = setup_logger(__name__, level="INFO")

class MedicalAgent(BaseAgent):
    """
    医疗Agent实现，基于RAG和大型语言模型
    """
    
    def __init__(
        self, 
        model: ApiModel,
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
            model=model,
            system_prompt=system_prompt,
            tools=None,
            max_iterations=max_iterations,
            verbose=verbose
        )
        
        self.model = model
        self.rag_pipeline = rag_pipeline
        self.tool_manager = ToolManager()
        self.temperature = temperature
        self.agent_id = agent_id
        self.name = name
        self.description = description
        
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self._memory = [{"role": "system", "content": self.system_prompt}]
        
    @property
    def memory(self) -> List[Dict[str, str]]:
        """获取对话记忆"""
        return self._memory
    
    @memory.setter
    def memory(self, value: List[Dict[str, str]]):
        """设置对话记忆"""
        self._memory = value
    
    def add_to_memory(self, role: str, content: str) -> None:
        """
        向记忆中添加消息
        
        Args:
            role: 消息角色 (system, user, assistant)
            content: 消息内容
        """
        self._memory.append({"role": role, "content": content})
    
    def clear_memory(self) -> None:
        """清空记忆，保留系统提示词"""
        self._memory = [{"role": "system", "content": self.system_prompt}]
        
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
            retrieval_results = self.rag_pipeline.query(query)
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
    
    def _assess_query_complexity(self, query: str) -> Dict[str, Any]:
        """
        评估查询复杂度，用于决定处理策略
        
        Args:
            query: 用户查询
            
        Returns:
            复杂度评估结果
        """
        complexity_keywords = {
            "CRITICAL": ["急诊", "急救", "生命危险", "胸痛", "呼吸困难", "昏迷", "大出血"],
            "HIGH": ["诊断", "治疗方案", "药物相互作用", "手术", "肿瘤", "慢性病"],
            "MODERATE": ["症状", "检查", "预防", "康复", "生活建议"],
            "LOW": ["常识", "保健", "营养", "运动建议"]
        }
        
        query_lower = query.lower()
        max_severity = "LOW"
        matched_keywords = []
        
        for severity, keywords in complexity_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    if severity in ["CRITICAL", "HIGH"]:
                        return {"complexity": severity, "requires_evidence": True, "matched_keywords": [keyword]}
                    elif severity == "MODERATE" and max_severity != "CRITICAL":
                        max_severity = "MODERATE"
                        matched_keywords.append(keyword)
        
        return {
            "complexity": max_severity,
            "requires_evidence": max_severity in ["HIGH", "MODERATE"],
            "matched_keywords": matched_keywords
        }
    
    def _verify_response_safety(self, response: str, query: str) -> Dict[str, Any]:
        """
        验证响应安全性
        
        Args:
            response: 生成的回答
            query: 原始查询
            
        Returns:
            安全性验证结果
        """
        safety_issues = []
        
        dangerous_patterns = [
            (r"建议.*自行.*治疗", "不建议自行治疗复杂疾病"),
            (r"不用.*去医院", "对于严重症状应建议就医"),
            (r"一定.*治愈", "医学上很少有绝对的治愈保证"),
            (r"没有任何.*副作用", "所有药物都可能有副作用"),
            (r"代替.*医生", "AI不能代替专业医生的诊断")
        ]
        
        for pattern, warning in dangerous_patterns:
            if re.search(pattern, response):
                safety_issues.append(warning)
        
        return {
            "is_safe": len(safety_issues) == 0,
            "issues": safety_issues,
            "warnings": []
        }
    
    def _calculate_confidence_score(self, response: str, context: str) -> float:
        """
        计算回答的置信度
        
        Args:
            response: 生成的回答
            context: 检索到的上下文
            
        Returns:
            置信度分数 (0-1)
        """
        base_confidence = 0.5
        
        if context:
            base_confidence += 0.3
        
        if re.search(r"根据.*研究|根据.*指南|循证医学", response):
            base_confidence += 0.1
        
        if re.search(r"建议.*咨询医生|建议.*就医|建议.*专业医师", response):
            base_confidence += 0.05
        
        if re.search(r"可能|也许|不确定|研究表明", response):
            base_confidence -= 0.05
        
        return min(1.0, max(0.0, base_confidence))
    
    def _extract_medical_evidence(self, response: str) -> List[str]:
        """
        提取回答中的医学证据引用
        
        Args:
            response: 生成的回答
            
        Returns:
            证据列表
        """
        evidence_patterns = [
            r"根据([^\n，]+)",
            r"研究显示([^\n。]+)",
            r"指南建议([^\n。]+)",
            r"循证医学([^\n。]+)"
        ]
        
        evidence = []
        for pattern in evidence_patterns:
            matches = re.findall(pattern, response)
            evidence.extend(matches)
        
        return evidence[:5] if evidence else []
    
    def _enhanced_run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        增强版运行方法，包含安全性和质量保证
        
        Args:
            query: 用户查询
            kwargs: 其他参数
            
        Returns:
            包含响应和完整元数据的字典
        """
        start_time = time.time()
        self.add_to_memory("user", query)
        
        complexity_assessment = self._assess_query_complexity(query)
        
        iteration = 0
        response = None
        metadata = {
            "query": query,
            "complexity_assessment": complexity_assessment,
            "iterations": 0,
            "tool_calls": [],
            "rag_used": False,
            "retrieved_documents": [],
            "safety_verification": None,
            "confidence_score": 0.0,
            "evidence_references": [],
            "timing": {},
            "warnings": []
        }
        
        if complexity_assessment["complexity"] == "CRITICAL":
            metadata["warnings"].append("警告：此查询涉及紧急医疗情况，建议用户立即就医或拨打急救电话")
        
        should_use_tools = self._should_use_tools(query)
        metadata["tool_usage_decision"] = should_use_tools
        
        rag_start_time = time.time()
        context = ""
        if self.rag_pipeline:
            context = self._retrieve_medical_knowledge(query)
            metadata["rag_used"] = bool(context)
            if context:
                documents = self.rag_pipeline.get_last_retrieval_documents()
                metadata["retrieved_documents"] = [
                    {
                        "content": doc.page_content[:500],
                        "metadata": doc.metadata
                    } for doc in documents
                ]
        metadata["timing"]["rag_retrieval"] = time.time() - rag_start_time
        
        while iteration < self.max_iterations:
            iteration += 1
            metadata["iterations"] = iteration
            
            if context:
                augmented_query = (
                    f"用户查询: {query}\n\n"
                    f"相关医疗知识:\n{context}\n\n"
                    "请基于上述信息回答用户的问题。你的回答必须：\n"
                    "1. 基于医学证据，避免编造信息\n"
                    "2. 对于不确定的信息，明确表达不确定性\n"
                    "3. 对于严重症状，建议咨询专业医生\n"
                    "4. 引用具体的医学指南或研究支持你的建议\n"
                    "如果提供的信息不足以回答问题，可以使用工具或基于你的医学知识回答。"
                )
            else:
                augmented_query = query
            
            if iteration > 1:
                self.memory[-1]["content"] = augmented_query
            
            generation_start_time = time.time()
            response_text = self.model.generate(
                self.memory,
                temperature=self.temperature,
                **kwargs
            )
            metadata["timing"][f"generation_{iteration}"] = time.time() - generation_start_time
            
            safety_check = self._verify_response_safety(response_text, query)
            metadata["safety_verification"] = safety_check
            
            if not should_use_tools or iteration == self.max_iterations:
                response = response_text
                self.add_to_memory("assistant", response)
                break
            
            tool_calls = self._parse_tool_calls(response_text)
            
            if not tool_calls:
                response = response_text
                self.add_to_memory("assistant", response)
                break
            
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
            
            tools_response = "工具执行结果:\n"
            for result in tool_results:
                tools_response += f"工具: {result['tool']}\n"
                tools_response += f"执行状态: {'成功' if result['success'] else '失败'}\n"
                tools_response += f"结果: {result['result']}\n\n"
            
            self.add_to_memory("user", tools_response)
        
        metadata["confidence_score"] = self._calculate_confidence_score(response or "", context)
        metadata["evidence_references"] = self._extract_medical_evidence(response or "")
        metadata["timing"]["total"] = time.time() - start_time
        
        if metadata["safety_verification"] and not metadata["safety_verification"]["is_safe"]:
            for issue in metadata["safety_verification"]["issues"]:
                metadata["warnings"].append(f"安全提醒: {issue}")
        
        return {
            "response": response,
            "metadata": metadata
        }