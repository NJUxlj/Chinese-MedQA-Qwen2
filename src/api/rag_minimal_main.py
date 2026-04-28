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
from providers.embedding_provider import EmbeddingProvider
from providers.reranker_provider import RerankerProvider
from config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化 rag_service
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
        model_path = str(llm_cfg.model_path) if hasattr(llm_cfg, 'model_path') and llm_cfg.model_path else ""
        local_backend = str(llm_cfg.local_backend) if hasattr(llm_cfg, 'local_backend') and llm_cfg.local_backend else "transformers"
        llm_kwargs.update(model_path=model_path, local_backend=local_backend)

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
    reranker_kwargs = dict(
        config=reranker_cfg,
    )

    llm_provider = LLMProvider(**llm_kwargs)
    embedding_provider = EmbeddingProvider(**embedding_kwargs)
    reranker_provider = RerankerProvider(**reranker_kwargs)
    
    get_rag_service(llm_provider, embedding_provider, reranker_provider)
    yield


app = FastAPI(title="Chinese-MedQA-Qwen2 RAG-Only API", version="1.0.0", lifespan=lifespan)
app.include_router(rag.router, prefix="/api/rag", tags=["RAG服务"])


@app.get("/health")
async def health():
    return {"status": "ok", "mode": "rag_only"}
