#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""仅挂载RAG路由的轻量入口，用于在无完整依赖时本地验证 rag HTTP API。"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextlib import asynccontextmanager

from fastapi import FastAPI

from routers import rag
from services.rag_service import get_rag_service, RAGService
from providers.llm_provider import LLMProvider
from config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化 rag_service
    cfg = settings.llm
    provider = str(cfg.model_provider)
    kwargs = dict(
        provider=provider,
        model_name=str(cfg.model_name),
        base_url=str(cfg.base_url) if hasattr(cfg, 'base_url') and cfg.base_url else None,
        api_key=str(cfg.api_key) if hasattr(cfg, 'api_key') and cfg.api_key else None,
        max_tokens=int(cfg.max_tokens) if hasattr(cfg, 'max_tokens') and cfg.max_tokens else 2048,
        temperature=float(cfg.temperature) if hasattr(cfg, 'temperature') and cfg.temperature else 0.7,
        top_p=float(cfg.top_p) if hasattr(cfg, 'top_p') and cfg.top_p else 0.9,
        timeout=int(cfg.timeout) if hasattr(cfg, 'timeout') and cfg.timeout else 60,
    )
    if provider == "local":
        model_path = str(cfg.model_path) if hasattr(cfg, 'model_path') and cfg.model_path else ""
        local_backend = str(cfg.local_backend) if hasattr(cfg, 'local_backend') and cfg.local_backend else "transformers"
        kwargs.update(model_path=model_path, local_backend=local_backend)

    llm_provider = LLMProvider(**kwargs)
    get_rag_service(llm_provider=llm_provider)
    yield


app = FastAPI(title="Chinese-MedQA-Qwen2 RAG-Only API", version="1.0.0", lifespan=lifespan)
app.include_router(rag.router, prefix="/api/rag", tags=["RAG服务"])


@app.get("/health")
async def health():
    return {"status": "ok", "mode": "rag_only"}
