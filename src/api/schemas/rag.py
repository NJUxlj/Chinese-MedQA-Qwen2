"""
RAG服务的模式定义
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class Document(BaseModel):
    """文档"""
    id: str = Field(..., description="文档ID")
    title: Optional[str] = Field(None, description="文档标题")
    content: str = Field(..., description="文档内容")
    metadata: Optional[Dict[str, Any]] = Field(None, description="文档元数据")
    score: Optional[float] = Field(None, description="相关性分数")
    source: Optional[str] = Field(None, description="文档来源")

class RetrieveRequest(BaseModel):
    """检索请求"""
    query: str = Field(..., description="查询文本")
    kb_name: str = Field(..., description="知识库名称")
    top_k: int = Field(5, description="返回文档数量")
    filter: Optional[Dict[str, Any]] = Field(None, description="过滤条件")

class RetrieveResponse(BaseModel):
    """检索响应"""
    query: str = Field(..., description="查询文本")
    documents: List[Document] = Field(..., description="检索到的文档")
    kb_name: str = Field(..., description="知识库名称")
    process_time: float = Field(..., description="处理时间(秒)")
    total_results: int = Field(..., description="总结果数")

class RagQuestionRequest(BaseModel):
    """RAG问答请求"""
    question: str = Field(..., description="用户问题")
    kb_name: str = Field(..., description="知识库名称")
    model_name: Optional[str] = Field(None, description="模型名称，为空使用默认模型")
    top_k: int = Field(5, description="检索文档数量")
    filter: Optional[Dict[str, Any]] = Field(None, description="过滤条件")

class RagQuestionResponse(BaseModel):
    """RAG问答响应"""
    question: str = Field(..., description="原始问题")
    answer: str = Field(..., description="模型回答")
    contexts: List[str] = Field(..., description="使用的上下文片段")
    sources: List[Dict[str, Any]] = Field(..., description="引用的来源")
    kb_name: str = Field(..., description="知识库名称")
    model: str = Field(..., description="使用的模型")
    process_time: float = Field(..., description="处理时间(秒)")

class KnowledgeBaseRequest(BaseModel):
    """知识库请求"""
    name: str = Field(..., description="知识库名称")
    description: str = Field("", description="知识库描述")

class KnowledgeBaseResponse(BaseModel):
    """知识库响应"""
    name: str = Field(..., description="知识库名称")
    description: str = Field("", description="知识库描述")
    document_count: int = Field(0, description="文档数量")
    created_at: str = Field(..., description="创建时间")
    last_updated: Optional[str] = Field(None, description="最后更新时间")
    size_bytes: Optional[int] = Field(None, description="索引大小(字节)")

class DocumentUploadRequest(BaseModel):
    """文档上传请求"""
    documents: List[Document] = Field(..., description="要上传的文档列表")