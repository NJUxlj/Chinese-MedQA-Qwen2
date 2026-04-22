"""
测试 evaluation router 的两个端点：
1. POST /api/evaluation/evaluate_model - 提交评估任务
2. GET /api/evaluation/get_evaluation_result/{task_id} - 查询评估结果

使用 FastAPI TestClient 直接测试，不经过 main.py（避免不必要的依赖加载）
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi.testclient import TestClient
from api.routers.evaluation import router as evaluation_router
from pydantic import BaseModel
from fastapi import FastAPI

# Minimal app just for testing evaluation endpoints
app = FastAPI()
app.include_router(evaluation_router, prefix="/api/evaluation")

client = TestClient(app)

# 测试数据
test_dataset_path = str(Path(__file__).parent / "src" / "training" / "data" / "sft_data" / "sft_valid_data.json")

print("=" * 60)
print("测试 1: POST /api/evaluation/evaluate_model (提交评估任务)")
print("=" * 60)

resp = client.post("/api/evaluation/evaluate_model", json={
    "dataset_path": test_dataset_path,
    "question_key": "messages",  # 使用 messages 作为问题键
    "model_answer_key": "output",
    "ground_true_answer_key": "answer",
    "model_name": "MiniMax-M2.7",
    "max_workers": 1,
})
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    print(f"Response: {data}")
    task_id = data.get("evaluation_task_id")
    print(f"task_id: {task_id}")
else:
    print(f"Error: {resp.text}")
    task_id = None

print()

# 如果获取到了 task_id，继续测试查询端点
if task_id:
    print("=" * 60)
    print(f"测试 2: GET /api/evaluation/get_evaluation_result/{task_id}")
    print("=" * 60)

    # 轮询结果，最多 5 次
    for i in range(5):
        resp = client.get(f"/api/evaluation/get_evaluation_result/{task_id}")
        print(f"轮询 #{i+1} - Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"Response: {data}")
            status = data.get("status", "unknown")
            print(f"Status: {status}")
            if status == "stop":
                print("评估完成!")
                break
            elif status == "pending":
                print("任务仍在处理中，等待...")
                import time
                time.sleep(2)
        else:
            print(f"Error: {resp.text}")
            break

print()
print("=" * 60)
print("测试 3: GET /api/evaluation/get_evaluation_result/{invalid_id} (404 测试)")
print("=" * 60)
resp = client.get("/api/evaluation/get_evaluation_result/invalid-task-id-12345")
print(f"Status: {resp.status_code}")
print(f"Response: {resp.json() if resp.status_code == 404 else resp.text}")