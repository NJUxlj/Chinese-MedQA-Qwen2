"""
直接测试 qa router，不经过 main.py（避免 faiss 依赖链加载）
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi.testclient import TestClient
from api.routers.qa import router as qa_router
from pydantic import BaseModel

# Minimal app just for testing qa endpoints
from fastapi import FastAPI
app = FastAPI()
app.include_router(qa_router, prefix="/api/qa")

client = TestClient(app)

print("=" * 60)
print("Testing /api/qa/ask")
print("=" * 60)
resp = client.post("/api/qa/ask", json={
    "question": "你好，什么是高血压？",
    "max_tokens": 100,
    "temperature": 0.7,
    "use_template": False
})
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    print(f"Answer: {data.get('answer', '')[:200]}")
    print(f"Model: {data.get('model')}")
    print(f"Process time: {data.get('process_time'):.3f}s")
else:
    print(f"Error: {resp.text}")

print()
print("=" * 60)
print("Testing /api/qa/ask (with template)")
print("=" * 60)
resp = client.post("/api/qa/ask", json={
    "question": "高血压患者应该注意什么？",
    "max_tokens": 100,
    "use_template": True
})
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    print(f"Answer: {data.get('answer', '')[:200]}")
else:
    print(f"Error: {resp.text}")

print()
print("=" * 60)
print("Testing /api/qa/models")
print("=" * 60)
resp = client.get("/api/qa/models")
print(f"Status: {resp.status_code}")
print(f"Response: {resp.json()}")
