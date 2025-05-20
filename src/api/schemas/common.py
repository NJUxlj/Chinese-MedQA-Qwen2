"""
通用的模式定义
"""

from typing import Dict, Any, List, Optional, Generic, TypeVar, Union
from pydantic import BaseModel, Field

T = TypeVar('T')

class ErrorResponse(BaseModel):
    """错误响应"""
    detail: str = Field(..., description="错误详情")
    message: str = Field(..., description="错误消息")
    code: Optional[int] = Field(None, description="错误代码")

class PaginationRequest(BaseModel):
    """分页请求"""
    page: int = Field(1, description="页码")
    page_size: int = Field(20, description="每页数量")

class PaginatedResponse(BaseModel, Generic[T]):
    """分页响应"""
    items: List[T] = Field(..., description="数据项")
    total: int = Field(..., description="总数")
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页数量")
    total_pages: int = Field(..., description="总页数")

class StatusResponse(BaseModel):
    """状态响应"""
    success: bool = Field(..., description="操作是否成功")
    message: str = Field(..., description="状态消息")
    data: Optional[Any] = Field(None, description="相关数据")
    timestamp: float = Field(..., description="时间戳")