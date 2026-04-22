#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Chinese-MedQA-Qwen2 API 服务主入口
提供医疗问答和知识检索的HTTP接口
"""

import os
import logging
from typing import Dict, Any, List
import time
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import settings

from .routers import qa, admin, rag, health, evaluation, mdagents
from .utils.async_eval_queue import start_evaluation_worker, stop_evaluation_worker

from .services.rag_service import get_rag_service, RAGService
from .services.mdagents_service import get_mdagents_service, MDAgentsService

from .ui.medagents_ui import MDAgentsUI

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api")

# 全局模型加载状态
models_loaded = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用程序生命周期管理
    在应用启动时加载模型，应用关闭时释放资源
    """
    logger.info("API服务启动，开始加载模型...")
    
    # 加载服务
    rag_service = get_rag_service()
    mdagents_service = get_mdagents_service()

    if bool(settings.api_server.preload_models):
        try:
            global models_loaded
            models_loaded = True
            logger.info("模型加载完成")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
    else:
        logger.info("跳过模型预加载")

    await start_evaluation_worker()
    
    yield  # 应用运行
    
    # 清理资源
    logger.info("API服务关闭，释放资源...")
    await stop_evaluation_worker()

# 创建应用
app = FastAPI(
    title="Chinese-MedQA-Qwen2 API",
    description="中文医疗问答系统API，基于Qwen2模型",
    version="1.0.0",
    lifespan=lifespan
)

# 添加中间件
_cors_origins = settings.api_server.cors_allowed_origins
if isinstance(_cors_origins, str):
    _cors_origins = [o.strip() for o in _cors_origins.split(",") if o.strip()]
else:
    _cors_origins = list(_cors_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(health.router, tags=["健康检查"])
app.include_router(qa.router, prefix="/api/qa", tags=["问答服务"])
app.include_router(rag.router, prefix="/api/rag", tags=["知识检索"])
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["评估服务"])
app.include_router(admin.router, prefix="/api/admin", tags=["管理接口"])
app.include_router(mdagents.router, prefix="/api/mdagents", tags=["MDAgents医疗多智能体系统"])

# 挂载Gradio UI
try:
    logger.info("正在初始化Gradio UI...")
    ui = MDAgentsUI(api_base_url="http://localhost:8000/api/mdagents")
    gradio_app = ui.create_interface()
    app.mount("/ui", gradio_app)
    logger.info("Gradio UI已挂载到 /ui 路径")
except Exception as e:
    logger.warning(f"Gradio UI挂载失败: {e}")

# 请求计数中间件
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# 错误处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"全局异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "message": "服务器内部错误"}
    )

# 静态文件
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    logger.warning("未找到static目录，跳过静态文件挂载")

# 根路由
@app.get("/")
async def root():
    return {
        "name": "Chinese-MedQA-Qwen2 API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": models_loaded
    }

if __name__ == "__main__":
    import uvicorn

    host = str(settings.api_server.host)
    port = int(settings.api_server.port)

    uvicorn.run("main:app", host=host, port=port, reload=True)