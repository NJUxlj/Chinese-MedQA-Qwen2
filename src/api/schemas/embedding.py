"""
嵌入服务的模式定义
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """嵌入请求"""
    text: str = Field(..., description="输入文本")
    model_name: Optional[str] = Field(None, description="模型名称")


class BatchEmbeddingRequest(BaseModel):
    """批量嵌入请求"""
    texts: List[str] = Field(..., description="输入文本列表")
    model_name: Optional[str] = Field(None, description="模型名称")


class SimilarityRequest(BaseModel):
    """相似度计算请求"""
    text1: str = Field(..., description="第一个文本")
    text2: str = Field(..., description="第二个文本")
    model_name: Optional[str] = Field(None, description="模型名称")
