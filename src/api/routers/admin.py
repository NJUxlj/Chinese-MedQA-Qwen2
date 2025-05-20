"""
管理接口路由
提供系统管理功能
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Security, Query
from fastapi.security import APIKeyHeader
import time
import os

from services.model_service import get_model_service, ModelService
from services.embedding_service import get_embedding_service, EmbeddingService
from services.rag_service import get_rag_service, RAGService

router = APIRouter()

# API密钥认证
API_KEY = os.environ.get("ADMIN_API_KEY", "change_me_in_production")
api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key: str = Security(api_key_header)):
    """检查API密钥"""
    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="无效的API密钥"
        )
    return api_key

@router.post("/models/load", response_model=Dict[str, Any])
async def load_model(
    model_name: str = Query(..., description="模型名称"),
    model_path: str = Query(..., description="模型路径"),
    model_type: str = Query("qwen2", description="模型类型"),
    model_service: ModelService = Depends(get_model_service),
    api_key: str = Depends(get_api_key)
):
    """
    加载LLM模型
    
    Args:
        model_name: 模型名称
        model_path: 模型路径
        model_type: 模型类型
        model_service: 模型服务
        api_key: API密钥
    
    Returns:
        操作结果
    """
    try:
        model = model_service.load_model(
            model_name=model_name,
            model_path=model_path,
            model_type=model_type
        )
        
        return {
            "success": True,
            "model_name": model_name,
            "model_type": model_type,
            "message": f"模型加载成功: {model_name}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/unload", response_model=Dict[str, Any])
async def unload_model(
    model_name: str = Query(..., description="模型名称"),
    model_service: ModelService = Depends(get_model_service),
    api_key: str = Depends(get_api_key)
):
    """
    卸载LLM模型
    
    Args:
        model_name: 模型名称
        model_service: 模型服务
        api_key: API密钥
    
    Returns:
        操作结果
    """
    try:
        success = model_service.unload_model(model_name)
        
        if not success:
            raise HTTPException(status_code=400, detail=f"卸载模型失败: {model_name}")
        
        return {
            "success": True,
            "model_name": model_name,
            "message": f"模型已卸载: {model_name}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embedding/load", response_model=Dict[str, Any])
async def load_embedding_model(
    model_name: str = Query(..., description="模型名称"),
    dimension: int = Query(384, description="嵌入维度"),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    api_key: str = Depends(get_api_key)
):
    """
    加载嵌入模型
    
    Args:
        model_name: 模型名称
        dimension: 嵌入维度
        embedding_service: 嵌入服务
        api_key: API密钥
    
    Returns:
        操作结果
    """
    try:
        manager = embedding_service.get_embedding_manager(
            model_name=model_name,
            dimension=dimension
        )
        
        return {
            "success": True,
            "model_name": model_name,
            "dimension": dimension,
            "message": f"嵌入模型加载成功: {model_name}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embedding/unload", response_model=Dict[str, Any])
async def unload_embedding_model(
    model_name: str = Query(..., description="模型名称"),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    api_key: str = Depends(get_api_key)
):
    """
    卸载嵌入模型
    
    Args:
        model_name: 模型名称
        embedding_service: 嵌入服务
        api_key: API密钥
    
    Returns:
        操作结果
    """
    try:
        success = embedding_service.unload_model(model_name)
        
        if not success:
            raise HTTPException(status_code=400, detail=f"卸载嵌入模型失败: {model_name}")
        
        return {
            "success": True,
            "model_name": model_name,
            "message": f"嵌入模型已卸载: {model_name}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", response_model=Dict[str, Any])
async def get_admin_status(
    model_service: ModelService = Depends(get_model_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    rag_service: RAGService = Depends(get_rag_service),
    api_key: str = Depends(get_api_key)
):
    """
    获取系统全面状态
    
    Args:
        model_service: 模型服务
        embedding_service: 嵌入服务
        rag_service: RAG服务
        api_key: API密钥
    
    Returns:
        系统状态信息
    """
    try:
        # 收集系统信息
        models_info = model_service.get_loaded_models()
        embedding_models = embedding_service.get_available_models()
        knowledge_bases = rag_service.get_available_knowledge_bases()
        
        return {
            "models": {
                "llm": models_info,
                "default_llm": model_service.default_model_name,
                "embedding": embedding_models,
                "default_embedding": embedding_service.default_model_name
            },
            "knowledge_bases": knowledge_bases,
            "server_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))