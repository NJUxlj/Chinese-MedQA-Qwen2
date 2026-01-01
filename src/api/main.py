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

# 导入路由
from routers import qa, admin, rag, embedding, health, evaluation, mdagents

# 导入模型服务
from services.model_service import get_model_service, ModelService
from services.rag_service import get_rag_service, RAGService
from services.embedding_service import get_embedding_service, EmbeddingService
from services.mdagents_service import get_mdagents_service, MDAgentsService

# 导入Gradio UI
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ui.medagents_ui import MDAgentsUI

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
    
    # 加载模型
    model_service = get_model_service()
    rag_service = get_rag_service()
    embedding_service = get_embedding_service()
    mdagents_service = get_mdagents_service()
    
    if os.environ.get("PRELOAD_MODELS", "true").lower() == "true":
        try:
            # 预加载模型
            model_service.load_default_model()
            embedding_service.load_default_model()
            global models_loaded
            models_loaded = True
            logger.info("模型加载完成")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
    else:
        logger.info("跳过模型预加载")
    
    yield  # 应用运行
    
    # 清理资源
    logger.info("API服务关闭，释放资源...")
    model_service.unload_all_models()

# 创建应用
app = FastAPI(
    title="Chinese-MedQA-Qwen2 API",
    description="中文医疗问答系统API，基于Qwen2模型",
    version="1.0.0",
    lifespan=lifespan
)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(health.router, tags=["健康检查"])
app.include_router(qa.router, prefix="/api/qa", tags=["问答服务"])
app.include_router(rag.router, prefix="/api/rag", tags=["知识检索"])
app.include_router(embedding.router, prefix="/api/embedding", tags=["嵌入服务"])
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
except:
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
    
    # 获取配置
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", 8000))
    
    # 启动服务
    uvicorn.run("main:app", host=host, port=port, reload=True)