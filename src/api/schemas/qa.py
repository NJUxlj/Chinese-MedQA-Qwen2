"""
问答服务的模式定义
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class QuestionRequest(BaseModel):
    """问题请求"""
    question: str = Field(..., description="用户问题")
    model_name: Optional[str] = Field(None, description="模型名称，为空使用默认模型")
    max_tokens: int = Field(1024, description="最大生成token数")
    temperature: float = Field(0.7, description="温度参数，控制随机性")
    top_p: float = Field(0.9, description="Top-p参数，控制采样的词表覆盖范围")
    use_template: bool = Field(True, description="是否使用系统提示词模板")

class QuestionResponse(BaseModel):
    """问题响应"""
    question: str = Field(..., description="原始问题")
    answer: str = Field(..., description="模型回答")
    model: str = Field(..., description="使用的模型")
    process_time: float = Field(..., description="处理时间(秒)")
    tokens_used: int = Field(..., description="使用的token数量")

class StreamQuestionRequest(BaseModel):
    """流式问题请求"""
    question: str = Field(..., description="用户问题")
    model_name: Optional[str] = Field(None, description="模型名称，为空使用默认模型")
    max_tokens: int = Field(1024, description="最大生成token数")
    temperature: float = Field(0.7, description="温度参数，控制随机性")
    top_p: float = Field(0.9, description="Top-p参数，控制采样的词表覆盖范围")
    use_template: bool = Field(True, description="是否使用系统提示词模板")

class ChatMessage(BaseModel):
    """聊天消息"""
    role: str = Field(..., description="消息角色(system/user/assistant)")
    content: str = Field(..., description="消息内容")

class ChatRequest(BaseModel):
    """聊天请求"""
    messages: List[ChatMessage] = Field(..., description="聊天消息历史")
    model_name: Optional[str] = Field(None, description="模型名称，为空使用默认模型")
    max_tokens: int = Field(1024, description="最大生成token数")
    temperature: float = Field(0.7, description="温度参数，控制随机性")
    top_p: float = Field(0.9, description="Top-p参数，控制采样的词表覆盖范围")

class ChatResponse(BaseModel):
    """聊天响应"""
    message: ChatMessage = Field(..., description="模型生成的消息")
    model: str = Field(..., description="使用的模型")
    process_time: float = Field(..., description="处理时间(秒)")
    tokens_used: int = Field(..., description="使用的token数量")