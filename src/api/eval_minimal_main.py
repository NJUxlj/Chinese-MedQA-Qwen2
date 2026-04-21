#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""仅挂载评估路由的轻量入口，用于在无完整依赖时本地验证 evaluation HTTP API。"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextlib import asynccontextmanager

from fastapi import FastAPI

from utils.async_eval_queue import start_evaluation_worker, stop_evaluation_worker
from routers import evaluation


@asynccontextmanager
async def lifespan(app: FastAPI):
    await start_evaluation_worker()
    yield
    await stop_evaluation_worker()


app = FastAPI(title="Chinese-MedQA-Qwen2 Eval-Only API", version="1.0.0", lifespan=lifespan)
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["评估服务"])


@app.get("/health")
async def health():
    return {"status": "ok", "mode": "eval_only"}
