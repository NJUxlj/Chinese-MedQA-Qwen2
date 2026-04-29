"""RAG服务路由
提供知识检索和增强生成接口（支持高并发）

并发架构：
    - FastAPI 端点：异步接收请求（支持无限并发连接）
    - asyncio.Queue：请求缓冲队列
    - ThreadPoolExecutor：执行 CPU/GPU 密集型 RAG 计算
    - 每个请求通过 asyncio.Future 等待结果
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
import time

from api.schemas.rag import (
    RagQuestionRequest,
    RagQuestionResponse
)
from api.services.rag_service import get_rag_service
from api.services.rag_async_queue import (
    submit_rag_task,
    get_rag_stats,
    set_rag_service_fn,
)

router = APIRouter()

# 注册 RAG 服务获取函数（模块加载时执行一次）
set_rag_service_fn(get_rag_service)


@router.post("/ask", response_model=RagQuestionResponse)
async def ask_rag_question(request: RagQuestionRequest):
    """
    基于知识库回答问题（支持高并发）
    
    并发特性：
        - 支持 50+ 并发连接
        - 请求进入队列，由线程池异步处理
        - 不会阻塞 FastAPI 事件循环
    
    Args:
        request: RAG问题请求
        
    Returns:
        回答结果
    """
    start_time = time.time()
    
    try:
        # 提交任务到异步队列并等待结果
        result = await submit_rag_task(
            query=request.question,
            model_name=request.model_name,
            top_k=request.top_k,
            timeout=300.0  # 5分钟超时
        )
        
        if not result.get("success", False):
            error_msg = result.get("error", "未知错误")
            raise HTTPException(status_code=500, detail=error_msg)
        
        total_time = time.time() - start_time
        
        return RagQuestionResponse(
            question=request.question,
            response=result.get("response", ""),
            context=result.get("context", ""),
            source_documents=result.get("source_documents", []),
            model=result.get("model", "default"),
            process_time=total_time
        )
        
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=f"请求超时: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=Dict[str, Any])
async def get_rag_queue_stats():
    """
    获取 RAG 服务统计信息
    
    Returns:
        包含队列大小、处理统计等信息的字典
    """
    return get_rag_stats()


@router.get("/health", response_model=Dict[str, str])
async def rag_health_check():
    """
    RAG 服务健康检查
    
    Returns:
        服务状态
    """
    stats = get_rag_stats()
    return {
        "status": "healthy" if stats["worker_running"] else "degraded",
        "queue_size": str(stats["current_queue_size"]),
        "total_requests": str(stats["total_requests"]),
        "completed_requests": str(stats["completed_requests"]),
    }
