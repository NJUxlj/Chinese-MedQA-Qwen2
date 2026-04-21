"""
管理接口路由
提供系统管理功能
"""

from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Security, Query
from fastapi.security import APIKeyHeader
import time

from config.settings import settings
from services.rag_service import get_rag_service, RAGService

router = APIRouter()

API_KEY = settings.api_server.admin_api_key
api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key: str = Security(api_key_header)):
    """检查API密钥"""
    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="无效的API密钥"
        )
    return api_key

@router.get("/status", response_model=Dict[str, Any])
async def get_admin_status(
    rag_service: RAGService = Depends(get_rag_service),
    api_key: str = Depends(get_api_key)
):
    """
    获取系统全面状态

    Args:
        rag_service: RAG服务
        api_key: API密钥

    Returns:
        系统状态信息
    """
    try:
        knowledge_bases = rag_service.get_available_knowledge_bases()

        return {
            "knowledge_bases": knowledge_bases,
            "server_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
