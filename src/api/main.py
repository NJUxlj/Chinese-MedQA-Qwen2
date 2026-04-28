#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chinese-MedQA-Qwen2 API 服务主入口
提供医疗问答和知识检索的HTTP接口
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import settings
from .routers import qa, admin, rag, health, evaluation
from .evaluation.async_eval_queue import start_evaluation_worker, stop_evaluation_worker
from .services.rag_service import get_rag_service
from .providers.llm_provider import LLMProvider
from .providers.embedding_provider import EmbeddingProvider
from .providers.reranker_provider import RerankerProvider

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用程序生命周期管理：启动时初始化服务，关闭时释放资源"""
    logger.info("API服务启动，开始初始化...")

    # 初始化 providers 并注册到 rag_service
    llm_cfg = settings.llm
    llm_provider_name = str(llm_cfg.provider)
    llm_kwargs = dict(
        provider=llm_provider_name,
        model_name=str(llm_cfg.model_name),
        base_url=str(llm_cfg.base_url) if hasattr(llm_cfg, 'base_url') and llm_cfg.base_url else None,
        api_key=str(llm_cfg.api_key) if hasattr(llm_cfg, 'api_key') and llm_cfg.api_key else None,
        max_tokens=int(llm_cfg.max_tokens) if hasattr(llm_cfg, 'max_tokens') and llm_cfg.max_tokens else 2048,
        temperature=float(llm_cfg.temperature) if hasattr(llm_cfg, 'temperature') and llm_cfg.temperature else 0.7,
        top_p=float(llm_cfg.top_p) if hasattr(llm_cfg, 'top_p') and llm_cfg.top_p else 0.9,
        timeout=int(llm_cfg.timeout) if hasattr(llm_cfg, 'timeout') and llm_cfg.timeout else 60,
    )
    if llm_provider_name == "local":
        llm_kwargs.update(
            model_path=str(llm_cfg.model_path) if hasattr(llm_cfg, 'model_path') and llm_cfg.model_path else "",
            local_backend=str(llm_cfg.local_backend) if hasattr(llm_cfg, 'local_backend') and llm_cfg.local_backend else "transformers"
        )

    embedding_cfg = settings.embedding
    embedding_kwargs = dict(
        provider=str(embedding_cfg.provider),
        model_name=str(embedding_cfg.model_name),
        model_path=str(embedding_cfg.model_path) if hasattr(embedding_cfg, 'model_path') and embedding_cfg.model_path else None,
        device=str(embedding_cfg.device) if hasattr(embedding_cfg, 'device') and embedding_cfg.device else None,
        base_url=str(embedding_cfg.base_url) if hasattr(embedding_cfg, 'base_url') and embedding_cfg.base_url else None,
        api_key=str(embedding_cfg.api_key) if hasattr(embedding_cfg, 'api_key') and embedding_cfg.api_key else None,
    )

    reranker_cfg = settings.reranker
    reranker_kwargs = dict(config=reranker_cfg)

    llm_provider = LLMProvider(**llm_kwargs)
    embedding_provider = EmbeddingProvider(**embedding_kwargs)
    reranker_provider = RerankerProvider(**reranker_kwargs)
    get_rag_service(llm_provider, embedding_provider, reranker_provider)

    await start_evaluation_worker()
    logger.info("API服务初始化完成")

    yield

    # 关闭时释放资源
    logger.info("API服务关闭，释放资源...")
    await stop_evaluation_worker()


# 创建 FastAPI 应用
app = FastAPI(
    title="Chinese-MedQA-Qwen2 API",
    description="中文医疗问答系统API，基于Qwen2模型",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 中间件
_cors_origins = settings.api_server.cors_allowed_origins
if isinstance(_cors_origins, str):
    _cors_origins = [o.strip() for o in _cors_origins.split(",") if o.strip()]
else:
    _cors_origins = list(_cors_origins) if _cors_origins else []

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

# 注意: MDAgents 路由已暂时禁用（服务未修复）
# app.include_router(mdagents.router, prefix="/api/mdagents", tags=["MDAgents医疗多智能体系统"])


# 请求计时中间件
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(time.time() - start_time)
    return response


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"全局异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "message": "服务器内部错误"}
    )


# 静态文件（可选）
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
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn
    host = str(settings.api_server.host)
    port = int(settings.api_server.port)
    uvicorn.run("main:app", host=host, port=port, reload=True)
