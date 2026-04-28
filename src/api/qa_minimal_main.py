#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""仅挂载问答路由的轻量入口，用于本地验证 QA HTTP API（含流式接口）。"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI

from routers import qa

app = FastAPI(title="Chinese-MedQA-Qwen2 QA-Only API", version="1.0.0")
app.include_router(qa.router, prefix="/api/qa", tags=["问答服务"])


@app.get("/health")
async def health():
    return {"status": "ok", "mode": "qa_only"}
