"""
健康检查路由
提供系统状态监控接口
"""

from typing import Dict, Any
from fastapi import APIRouter
import time
import psutil
import platform
import torch

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

        # GPU信息 (CUDA 或 Apple Silicon MPS)
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
        elif torch.backends.mps.is_available():
            system_info["cuda_available"] = False
            system_info["mps_available"] = True
            system_info["gpu_info"] = [{
                "index": 0,
                "name": "Apple Silicon GPU (MPS)",
                "memory_total": None,
                "memory_used": None
            }]
        else:
            system_info["cuda_available"] = False
            system_info["mps_available"] = False

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
