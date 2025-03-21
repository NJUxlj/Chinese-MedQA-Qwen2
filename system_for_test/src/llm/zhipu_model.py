
import os
from typing import Dict, List, Optional, Union, Any

import zhipuai
from zhipuai import ZhipuAI

class ZhipuLLM:
    """ZhipuAI大语言模型接口"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化ZhipuAI LLM。
        
        Args:
            config: 包含模型配置的字典
        """
        self.config = config
        self.model_name = config.get("model_name", "glm-4")
        self.api_key = config.get("api_key", os.environ.get("ZHIPU_API_KEY"))
        
        if not self.api_key:
            raise ValueError("未提供ZhipuAI API密钥。请在配置中提供或设置ZHIPU_API_KEY环境变量。")
            
        # 初始化ZhipuAI客户端
        self.client = ZhipuAI(api_key=self.api_key)
        
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> str:
        """
        生成文本响应。
        
        Args:
            prompt: 用户输入的提示
            system_prompt: 系统提示（可选）
            temperature: 生成温度，越低越确定性
            max_tokens: 最大生成令牌数
            stream: 是否流式输出
            
        Returns:
            生成的文本响应
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
        
        if stream:
            # 流式输出处理
            collected_chunks = []
            for chunk in response:
                if chunk.choices[0].delta.content:
                    collected_chunks.append(chunk.choices[0].delta.content)
                    # 如果需要实时显示，可以在这里添加打印
            return "".join(collected_chunks)
        else:
            return response.choices[0].message.content
            
    def generate_with_tools(
        self, 
        prompt: str, 
        tools: List[Dict], 
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048
    ) -> Dict:
        """
        使用工具生成响应（支持函数调用）。
        
        Args:
            prompt: 用户输入的提示
            tools: 工具定义列表
            system_prompt: 系统提示（可选）
            temperature: 生成温度
            max_tokens: 最大生成令牌数
            
        Returns:
            包含响应和可能工具调用的字典
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        result = {
            "content": response.choices[0].message.content,
            "tool_calls": None
        }
        
        if hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls:
            result["tool_calls"] = response.choices[0].message.tool_calls
            
        return result
