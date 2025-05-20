"""
嵌入服务路由
提供文本嵌入接口
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from services.embedding_service import get_embedding_service, EmbeddingService

router = APIRouter()

class EmbeddingRequest(BaseModel):
    """嵌入请求"""
    text: str
    model_name: Optional[str] = None

class BatchEmbeddingRequest(BaseModel):
    """批量嵌入请求"""
    texts: List[str]
    model_name: Optional[str] = None

class SimilarityRequest(BaseModel):
    """相似度计算请求"""
    text1: str
    text2: str
    model_name: Optional[str] = None

@router.post("/embed", response_model=List[float])
async def get_embedding(
    request: EmbeddingRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """
    获取文本嵌入向量
    
    Args:
        request: 嵌入请求
        embedding_service: 嵌入服务
    
    Returns:
        嵌入向量
    """
    try:
        embedding = embedding_service.get_embedding(
            text=request.text,
            model_name=request.model_name
        )
        return embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embed_batch", response_model=List[List[float]])
async def get_embeddings_batch(
    request: BatchEmbeddingRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """
    批量获取文本嵌入向量
    
    Args:
        request: 批量嵌入请求
        embedding_service: 嵌入服务
    
    Returns:
        嵌入向量列表
    """
    try:
        embeddings = embedding_service.get_embeddings(
            texts=request.texts,
            model_name=request.model_name
        )
        return embeddings
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/similarity", response_model=float)
async def calculate_similarity(
    request: SimilarityRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """
    计算两个文本的相似度
    
    Args:
        request: 相似度计算请求
        embedding_service: 嵌入服务
    
    Returns:
        相似度分数(0-1)
    """
    try:
        similarity = embedding_service.calculate_similarity(
            text1=request.text1,
            text2=request.text2,
            model_name=request.model_name
        )
        return similarity
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models", response_model=List[str])
async def get_available_models(
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """
    获取可用嵌入模型列表
    
    Args:
        embedding_service: 嵌入服务
    
    Returns:
        模型名称列表
    """
    return embedding_service.get_available_models()