"""
评估服务路由：将评估请求入队，异步在线程池中运行，轮询结果。

流程：
  POST /evaluate_model
    → register_pending_task()
    → queue.put(payload)          # 立即返回 pending
  async_eval_queue worker:
    → asyncio.to_thread(run_evaluation_task, payload)   # 在线程中运行
      → EvaluatorFactory.create()
      → evaluator.evaluate()  （内部可再开 ThreadPoolExecutor）
    → EVALUATION_TASKS[task_id] 更新为 stop + metrics
  GET /get_evaluation_result/{task_id}
    → 返回当前快照
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from evaluation.async_eval_queue import (
    get_task_snapshot,
    evaluation_task_queue,
    register_pending_task,
    set_evaluation_task_fn,
    start_evaluation_worker,
)
from evaluation.evaluation_runner import run_evaluation_task
from evaluation.task_types import EvaluationTaskType
from config.settings import settings

router = APIRouter()

# 注册同步评估函数（模块加载时执行一次即可）
set_evaluation_task_fn(run_evaluation_task)


# ---------------------------------------------------------------------------
# Pydantic 模型
# ---------------------------------------------------------------------------

class EvaluationRequest(BaseModel):
    evaluation_task_id: Optional[str] = Field(default=None)
    task_type: EvaluationTaskType = Field(default=EvaluationTaskType.MEDQA)
    dataset_path: str = Field(..., description="JSON 数据集路径")
    question_key: str = Field(default="question")
    model_answer_key: str = Field(default="output")
    ground_true_answer_key: str = Field(default="answer")

    # LLM 连接参数，默认读取 settings.llm
    model_name: str = Field(default=settings.llm.model_name)
    base_url: str = Field(default=settings.llm.base_url)
    api_key: str = Field(default=settings.llm.api_key)
    temperature: float = Field(default=settings.llm.temperature)
    max_tokens: int = Field(default=settings.llm.max_tokens)
    max_workers: int = Field(default=settings.llm.max_workers)


class EvaluationResponse(BaseModel):
    evaluation_task_id: str
    metrics: Dict[str, float] = Field(default_factory=dict)
    model_name: str = ""
    status: str = "pending"
    process_time: float = 0.0
    additional_info: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------

@router.post("/evaluate_model", response_model=EvaluationResponse)
async def evaluate_model(request: EvaluationRequest) -> EvaluationResponse:
    """提交一个评估任务，立即返回 pending 状态；通过轮询接口查询结果。"""
    await start_evaluation_worker()

    task_id = (request.evaluation_task_id or "").strip() or str(uuid.uuid4())
    register_pending_task(task_id, request.model_name)

    await evaluation_task_queue.put({
        "evaluation_task_id":      task_id,
        "task_type":               request.task_type.value,
        "dataset_path":            request.dataset_path,
        "question_key":            request.question_key,
        "model_answer_key":        request.model_answer_key,
        "ground_true_answer_key":  request.ground_true_answer_key,
        "model_name":              request.model_name,
        "base_url":                request.base_url,
        "api_key":                 request.api_key,
        "temperature":             request.temperature,
        "max_tokens":              request.max_tokens,
        "max_workers":             request.max_workers,
    })

    snap = get_task_snapshot(task_id) or {}
    return EvaluationResponse(
        evaluation_task_id=task_id,
        metrics=snap.get("metrics") or {},
        model_name=str(snap.get("model_name") or request.model_name),
        status=snap.get("status", "pending"),
        process_time=float(snap.get("process_time") or 0.0),
        additional_info=snap.get("additional_info"),
    )


@router.get("/get_evaluation_result/{evaluation_task_id}", response_model=EvaluationResponse)
async def get_evaluation_result(evaluation_task_id: str) -> EvaluationResponse:
    """轮询评估任务结果。status 为 'stop' 时表示完成（成功或失败）。"""
    snap = get_task_snapshot(evaluation_task_id)
    if snap is None:
        raise HTTPException(status_code=404, detail="未知的 evaluation_task_id")
    return EvaluationResponse(
        evaluation_task_id=evaluation_task_id,
        metrics=snap.get("metrics") or {},
        model_name=str(snap.get("model_name") or ""),
        status=snap["status"],
        process_time=float(snap.get("process_time") or 0.0),
        additional_info=snap.get("additional_info"),
    )
