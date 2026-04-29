"""
RAG服务路由
提供知识检索和增强生成接口
"""

from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Path
import time

from api.schemas.rag import (
    RagQuestionRequest,
    RagQuestionResponse
)

from api.services.rag_service import get_rag_service, RAGService

router = APIRouter()


def _get_rag_service() -> RAGService:
    """无参依赖函数，用于获取已初始化的 RAG 服务"""
    return get_rag_service()


@router.post("/ask", response_model=RagQuestionResponse)
async def ask_rag_question(
    request: RagQuestionRequest,
    rag_service: RAGService = Depends(_get_rag_service)
):
    """
    基于知识库回答问题

    Args:
        request: RAG问题请求
        rag_service: RAG服务

    Returns:
        回答结果
    """
    start_time = time.time()

    try:
        response = rag_service.generate_response(
            query=request.question,
            model_name=request.model_name,
            top_k=request.top_k
        )

        process_time = time.time() - start_time

        return RagQuestionResponse(
            question=request.question,
            response=response.get("response", ""),
            context=response.get("context", ""),
            source_documents=response.get("source_documents", []),
            model=request.model_name or "default",
            process_time=process_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


