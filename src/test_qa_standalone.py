"""
Standalone minimal FastAPI app for testing qa endpoints.
Bypasses rag_pipeline/faiss chain by not importing main.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.routers.qa import router as qa_router

app = FastAPI()
app.include_router(qa_router, prefix="/api/qa")
client = TestClient(app)

BASE = "http://localhost:8000"

def test_ask():
    print("=" * 60)
    print("POST /api/qa/ask")
    print("=" * 60)
    resp = client.post("/api/qa/ask", json={
        "question": "你好，什么是高血压？",
        "max_tokens": 50,
        "temperature": 0.7,
        "use_template": False
    })
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Answer: {data.get('answer', '')[:150]}")
        print(f"Model: {data.get('model')}")
        print(f"Process time: {data.get('process_time', 0):.3f}s")
    else:
        print(f"Error: {resp.text}")

def test_ask_with_template():
    print()
    print("=" * 60)
    print("POST /api/qa/ask (with template)")
    print("=" * 60)
    resp = client.post("/api/qa/ask", json={
        "question": "高血压患者应该注意什么？",
        "max_tokens": 50,
        "use_template": True
    })
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Answer: {data.get('answer', '')[:150]}")
    else:
        print(f"Error: {resp.text}")

def test_models():
    print()
    print("=" * 60)
    print("GET /api/qa/models")
    print("=" * 60)
    resp = client.get("/api/qa/models")
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}")

def test_root():
    print()
    print("=" * 60)
    print("GET /")
    print("=" * 60)
    resp = client.get("/")
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}")

if __name__ == "__main__":
    test_root()
    test_models()
    test_ask()
    test_ask_with_template()
