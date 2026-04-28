"""
RAG服务的模式定义
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class Document(BaseModel):
    """文档"""
    id: str = Field(..., description="文档ID")
    title: Optional[str] = Field(None, description="文档标题")
    content: str = Field(..., description="文档内容")
    metadata: Optional[Dict[str, Any]] = Field(None, description="文档元数据")
    score: Optional[float] = Field(None, description="相关性分数")
    source: Optional[str] = Field(None, description="文档来源")



class RagQuestionRequest(BaseModel):
    """RAG问答请求"""
    question: str = Field(..., description="用户问题")
    model_name: Optional[str] = Field(None, description="模型名称，为空使用默认模型")
    top_k: int = Field(5, description="检索文档数量")
    filter: Optional[Dict[str, Any]] = Field(None, description="过滤条件")

class RagQuestionResponse(BaseModel):
    """RAG问答响应"""
    question: str = Field(..., description="原始问题")
    response: str = Field(..., description="模型回答")
    context: str = Field(..., description="使用的上下文片段")
    source_documents: List[Document] = Field(..., description="引用的来源")
    model: str = Field(..., description="使用的模型")
    process_time: float = Field(..., description="处理时间(秒)")

