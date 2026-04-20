# 基础agent类
from typing import Dict, List, Optional, Union, Any, Tuple
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).parent.parent))
from providers import LLMProvider
import json
import re
from utils.logger import setup_logger

logger = setup_logger(__name__)

class BaseAgent:
    """基础Agent类"""
    
    def __init__(
        self,
        model: LLMProvider,
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
        
        tools_desc += "当你需要使用工具来获取医疗相关的最新数据、药品信息或专业指南时，请严格使用以下格式调用工具：\n"
        tools_desc += "```\n工具: 工具名称\n参数: {\"参数1\": \"值1\", \"参数2\": \"值2\", ...}\n```\n"
        tools_desc += "注意：请确保工具调用的参数准确无误，所有必需参数都已提供。工具调用完成后，请结合返回结果给出专业的医疗建议。\n"
        
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
                continue_query = tool_result_message
            
            iteration += 1
        
        # 如果达到最大迭代次数但没有最终响应
        if not final_response:
            final_response = "很抱歉，我无法完成这个任务。请尝试重新描述您的问题，或者联系技术支持。"
        
        return {
            "response": final_response,
            "iterations": iteration,
            "chat_history": current_chat_history
        }

    def plan(self, query: str) -> str:
        '''
        Agent 的计划模块
        '''
        plan_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            请你为以下医疗问题制定详细的解决计划：
            问题：{query}
            
            请按照以下格式输出计划：
            1. 分析问题：明确问题的核心需求和关键信息
            2. 确定步骤：列出解决问题的具体步骤
            3. 工具选择：说明需要使用哪些工具（如果有）
            4. 预期结果：描述每个步骤的预期结果
            
            请确保计划具有可操作性和逻辑性。
            """
        )
        
        chain = plan_prompt | self.model | StrOutputParser()
        response = chain.invoke({"query": query})
        return response

    def reflect(self, response):
        '''
        Agent 的反思模块
        '''
        reflect_prompt = PromptTemplate(
            input_variables=["response"],
            template="""
            请你对以下回答进行反思和评估：
            回答：{response}
            
            请按照以下维度进行评估：
            1. 准确性：回答是否准确反映了医疗知识
            2. 完整性：回答是否覆盖了问题的所有方面
            3. 逻辑性：回答是否具有清晰的逻辑结构
            4. 实用性：回答是否对用户有实际帮助
            5. 规范性：回答是否符合医疗术语规范
            
            请提供具体的改进建议。
            """
        )
        
        chain = reflect_prompt | self.model | StrOutputParser()
        reflection = chain.invoke({"response": response})
        return reflection
    def refine(self, query):
        '''
        Agent 的根据反思进行修正的模块
        '''
        refine_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            请你根据之前的反思结果，对以下医疗问题的回答进行修正和优化：
            问题：{query}
            
            修正要求：
            1. 确保回答的准确性，纠正之前可能存在的错误
            2. 补充缺失的信息，使回答更加完整
            3. 优化逻辑结构，使回答更加清晰易懂
            4. 提升实用性，为用户提供更具体的建议
            5. 规范医疗术语的使用
            
            请输出修正后的完整回答。
            """
        )
        
        chain = refine_prompt | self.model | StrOutputParser()
        refined_response = chain.invoke({"query": query})
        return refined_response

    def deep_search(self, query):
        """
        深度搜索模块，参考通义DeepResearch最佳实践
        """
        # 1. 问题分解
        decomposition_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            请将以下医疗问题分解为多个子问题，以便进行深度研究：
            问题：{query}
            
            分解要求：
            1. 子问题之间应具有逻辑性和层次性
            2. 覆盖问题的各个方面
            3. 每个子问题应具有明确的研究方向
            
            请输出分解后的子问题列表。
            """
        )
        
        decomposition_chain = decomposition_prompt | self.model | StrOutputParser()
        sub_questions = decomposition_chain.invoke({"query": query})
        
        # 2. 子问题研究
        research_results = []
        research_prompt = PromptTemplate(
            input_variables=["sub_question"],
            template="""
            请对以下医疗子问题进行深入研究：
            子问题：{sub_question}
            
            研究要求：
            1. 收集相关的医疗知识和数据
            2. 分析问题的关键因素
            3. 提供详细的解释和证据
            4. 引用可靠的医疗资源（如果有）
            
            请输出详细的研究结果。
            """
        )
        
        research_chain = research_prompt | self.model | StrOutputParser()
        
        for sub_question in sub_questions.split('\n'):
            if sub_question.strip():
                result = research_chain.invoke({"sub_question": sub_question})
                research_results.append((sub_question, result))
        
        # 3. 结果整合
        results_str = "\n".join([f"子问题：{q}\n结果：{r}" for q, r in research_results])
        integration_prompt = PromptTemplate(
            input_variables=["query", "results_str"],
            template="""
            请将以下子问题的研究结果整合为一个完整的回答：
            原始问题：{query}
            
            子问题研究结果：
            {results_str}
            
            整合要求：
            1. 保持逻辑连贯
            2. 突出重点信息
            3. 避免重复内容
            4. 提供清晰的结论
            
            请输出整合后的完整回答。
            """
        )
        
        integration_chain = integration_prompt | self.model | StrOutputParser()
        final_answer = integration_chain.invoke({"query": query, "results_str": results_str})
        
        return final_answer




    




