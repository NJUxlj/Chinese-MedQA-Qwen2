from typing import Dict, Any, Optional, List, Union
import json
import re

from .tool_base import ToolBase
from utils.logger import setup_logger as get_logger

logger = get_logger(__name__)

class ReActAgentTool(ToolBase):
    """
    ReAct (Reasoning and Acting) Agent 工具
    用于处理需要多步推理和行动的复杂任务
    """
    
    def __init__(
        self,
        model,  # 语言模型实例
        available_tools: Optional[Dict[str, Any]] = None,
        name: str = "推理代理",
        description: str = "用于处理复杂的医疗推理任务，能够进行多步思考和行动来解决问题"
    ) -> None:
        """
        初始化ReAct Agent工具
        
        Args:
            model: 语言模型实例，用于生成推理和行动
            available_tools: 可供ReAct Agent使用的工具字典
            name: 工具名称
            description: 工具描述
        """
        parameters = {
            "task": {
                "type": "string",
                "description": "需要解决的复杂任务或问题，应提供清晰的目标和相关背景信息",
                "required": True
            },
            "max_steps": {
                "type": "int",
                "description": "允许的最大推理步骤数",
                "required": False,
                "default": 5
            }
        }
        
        super().__init__(name=name, description=description, parameters=parameters)
        
        self.model = model
        self.available_tools = available_tools or {}
    
    def _run(self, task: str, max_steps: int = 5) -> str:
        """
        执行ReAct推理和行动
        
        Args:
            task: 任务描述
            max_steps: 最大步骤数
            
        Returns:
            ReAct执行结果
        """
        # 构建ReAct提示词
        tools_description = ""
        if self.available_tools:
            tools_description = "可用工具:\n"
            for tool_name, tool in self.available_tools.items():
                tools_description += f"- {tool_name}: {tool.description}\n"
        
        react_prompt = (
            f"任务: {task}\n\n"
            f"{tools_description}\n"
            "请使用以下格式解决这个问题:\n"
            "思考: [思考问题并制定计划]\n"
            "行动: [执行具体操作，可调用工具]\n"
            "观察: [记录行动结果]\n"
            "... (可重复思考-行动-观察多轮)\n"
            "结论: [给出最终答案]\n\n"
            "开始解决问题:"
        )
        
        # 追踪执行过程
        execution_trace = [react_prompt]
        current_step = 0
        
        while current_step < max_steps:
            current_step += 1
            
            # 生成下一步思考或行动
            response = self.model.generate(
                [{"role": "user", "content": "\n".join(execution_trace)}],
                temperature=0.7,
                max_tokens=1000
            )
            
            execution_trace.append(response)
            
            # 检查是否已得出结论
            if "结论:" in response:
                break
            
            # 解析行动并执行工具调用
            action_match = re.search(r"行动:\s*(.*?)(?=\n思考:|\n观察:|\n结论:|$)", response, re.DOTALL)
            if action_match:
                action_text = action_match.group(1).strip()
                
                # 解析工具调用
                tool_match = re.search(r"调用工具:\s*([^\n]+)\n参数:\s*({[^}]+})", action_text, re.DOTALL)
                if tool_match and self.available_tools:
                    tool_name = tool_match.group(1).strip()
                    try:
                        tool_params = json.loads(tool_match.group(2).strip())
                        
                        # 执行工具调用
                        if tool_name in self.available_tools:
                            tool = self.available_tools[tool_name]
                            tool_result = tool.run(**tool_params)
                            
                            # 添加观察结果
                            observation = f"观察: 工具 '{tool_name}' 的执行结果:\n{tool_result}"
                            execution_trace.append(observation)
                        else:
                            observation = f"观察: 错误 - 找不到工具 '{tool_name}'"
                            execution_trace.append(observation)
                    except Exception as e:
                        observation = f"观察: 错误 - 工具调用异常: {str(e)}"
                        execution_trace.append(observation)
            
        # 提取最终结论
        final_response = "\n".join(execution_trace)
        conclusion_match = re.search(r"结论:\s*(.*?)($)", final_response, re.DOTALL)
        
        if conclusion_match:
            conclusion = conclusion_match.group(1).strip()
            return (
                f"任务: {task}\n\n"
                f"执行了 {current_step} 步推理与行动\n\n"
                f"最终结论:\n{conclusion}"
            )
        else:
            # 如果没有明确的结论，返回完整的执行过程
            return (
                f"任务: {task}\n\n"
                f"执行了 {current_step} 步推理与行动，但未得出明确结论。\n\n"
                f"执行过程:\n{final_response}"
            )