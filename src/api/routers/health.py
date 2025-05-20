"""
健康检查路由
提供系统状态监控接口
"""

from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
import time
import psutil
import os
import platform
import torch

from services.model_service import get_model_service, ModelService
from services.embedding_service import get_embedding_service, EmbeddingService

router = APIRouter()

@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """
    系统健康检查
    
    Returns:
        系统状态信息
    """
    try:
        # 收集系统信息
        system_info = {
            "status": "ok",
            "timestamp": time.time(),
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "memory_available": psutil.virtual_memory().available / (1024 * 1024 * 1024),  # GB
            "disk_usage": psutil.disk_usage('/').percent,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        }
        
        # CUDA信息
        if torch.cuda.is_available():
            system_info["cuda_available"] = True
            system_info["cuda_version"] = torch.version.cuda
            system_info["cuda_devices"] = torch.cuda.device_count()
            system_info["gpu_info"] = []
            
            for i in range(torch.cuda.device_count()):
                gpu_info = {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory / (1024 * 1024 * 1024),  # GB
                    "memory_used": torch.cuda.memory_allocated(i) / (1024 * 1024 * 1024)  # GB
                }
                system_info["gpu_info"].append(gpu_info)
        else:
            system_info["cuda_available"] = False
        
        return system_info
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": time.time()
        }

@router.get("/ping")
async def ping():
    """
    简单的可用性检查
    
    Returns:
        简单响应
    """
    return {"status": "pong", "timestamp": time.time()}

@router.get("/models/status", response_model=Dict[str, Any])
async def get_models_status(
    model_service: ModelService = Depends(get_model_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """
    获取模型加载状态
    
    Args:
        model_service: 模型服务
        embedding_service: 嵌入服务
    
    Returns:
        模型状态信息
    """
    try:
        return {
            "llm_models": model_service.get_loaded_models(),
            "embedding_models": embedding_service.get_available_models(),
            "default_llm": model_service.default_model_name,
            "default_embedding": embedding_service.default_model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))