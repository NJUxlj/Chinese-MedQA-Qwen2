"""
基础智能体类 - 所有医疗智能体的基类
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from ..config import AgentConfig, AgentType
from ..tools.logger import setup_logger
from ..core.api_integration import ApiIntegrationManager, ApiModelWrapper

class BaseAgent(ABC):
    """医疗智能体基类"""
    
    def __init__(self, config: AgentConfig, api_model=None, llm_config=None):
        """
        初始化智能体
        
        Args:
            config: 智能体配置
            api_model: API模型实例，用于调用LLM
            llm_config: LLM配置实例
        """
        self.config = config
        self.agent_type = config.agent_type
        self.name = config.name
        self.logger = setup_logger(f"{self.__class__.__name__}")
        self.api_model = api_model
        self.llm_config = llm_config
        self.conversation_history = []
        self.status = "idle"
        
        # 初始化API集成管理器
        if api_model:
            self.api_integration = None
            self.api_wrapper = api_model
        elif llm_config:
            self.api_integration = ApiIntegrationManager(llm_config)
            self.api_wrapper = ApiModelWrapper(self.api_integration)
        else:
            self.api_integration = None
            self.api_wrapper = None
        
    def log_interaction(self, interaction_type: str, content: str, metadata: Dict[str, Any] = None):
        """记录交互日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_type": self.agent_type.value,
            "agent_name": self.name,
            "interaction_type": interaction_type,
            "content": content,
            "metadata": metadata or {}
        }
        self.conversation_history.append(log_entry)
        
        self.logger.info(f"[{interaction_type}] {self.name}: {content}")
    
    @abstractmethod
    def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        处理查询的抽象方法
        
        Args:
            query: 输入查询
            context: 上下文信息
            
        Returns:
            处理结果字典
        """
        pass
    
    def generate_response(self, prompt: str, messages: List[Dict[str, str]] = None, 
                         additional_args: Dict[str, Any] = None) -> str:
        """
        生成响应
        
        Args:
            prompt: 提示词
            messages: 消息列表
            additional_args: 额外参数
            
        Returns:
            生成的响应文本
        """
        if self.api_model is None:
            raise ValueError("API模型未配置，无法生成响应")
        
        self.status = "processing"
        self.log_interaction("generate_start", f"开始处理提示词: {prompt[:100]}...")
        
        try:
            response = self.api_model.generate(
                prompt=prompt,
                messages=messages,
                additional_args=additional_args or {}
            )
            
            self.log_interaction("generate_success", f"成功生成响应: {response[:100]}...")
            self.status = "idle"
            return response
            
        except Exception as e:
            self.logger.error(f"生成响应失败: {e}")
            self.status = "error"
            self.log_interaction("generate_error", f"生成响应失败: {str(e)}")
            raise
    
    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return self.config.system_prompt
    
    def format_messages(self, user_prompt: str, system_prompt: str = None) -> List[Dict[str, str]]:
        """格式化消息列表"""
        messages = []
        
        if system_prompt or self.config.system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt or self.config.system_prompt
            })
        
        messages.append({
            "role": "user", 
            "content": user_prompt
        })
        
        return messages
    
    def get_capabilities(self) -> List[str]:
        """获取智能体能力列表"""
        return self.config.capabilities.copy()
    
    def get_status(self) -> str:
        """获取当前状态"""
        return self.status
    
    def reset(self):
        """重置智能体状态"""
        self.status = "idle"
        self.conversation_history.clear()
        self.log_interaction("reset", "智能体状态已重置")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        return {
            "name": self.name,
            "agent_type": self.agent_type.value,
            "capabilities": self.get_capabilities(),
            "status": self.status,
            "conversation_length": len(self.conversation_history),
            "config": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "description": self.config.description
            }
        }
    
    def __str__(self) -> str:
        return f"{self.name} ({self.agent_type.value}, 状态: {self.status})"
    
    def __repr__(self) -> str:
        return self.__str__()