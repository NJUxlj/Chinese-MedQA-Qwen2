"""
评估服务路由：直接实例化 evaluator，在线程池中运行，聚合指标。
"""

from __future__ import annotations

import json
import os
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from utils.async_eval_queue import (
    evaluation_task_queue,
    get_task_snapshot,
    register_pending_task,
    set_evaluation_task_fn,
    start_evaluation_worker,
)
from utils.inference_utils import display_model_name, is_valid_vllm_triple
from utils.task_types import EvaluationTaskType

router = APIRouter()


class EvaluatorConfig(BaseModel):
    model_name_or_path: Optional[str] = Field(default=None)
    device: str = Field(default="cpu")
    max_new_tokens: int = Field(default=2048)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.9)
    vllm_model_name: Optional[str] = Field(default=None)
    vllm_base_url: Optional[str] = Field(default=None)
    vllm_api_key: Optional[str] = Field(default=None)
    prompt_key: str = Field(default="prompt")
    ground_true_answer_key: str = Field(default="answer")
    padding_side: str = Field(default="left")
    use_fast: bool = Field(default=True)
    enable_thinking: bool = Field(default=False)
    max_length: int = Field(default=2048)
    beta: float = Field(default=0.1)
    query_key: str = Field(default="query")
    chosen_key: str = Field(default="chosen")
    rejected_key: str = Field(default="rejected")
    execution_timeout: int = Field(default=30)
    max_memory_mb: int = Field(default=256)
    sandbox_working_dir: str = Field(default="/tmp/code_eval")
    enable_security_check: bool = Field(default=True)


class EvaluationRequest(BaseModel):
    evaluation_task_id: Optional[str] = Field(default=None)
    task_type: EvaluationTaskType
    dataset_path: str = Field(..., description="JSON 数据集路径")
    question_key: str = Field(default="question")
    model_answer_key: str = Field(default="output")
    ground_true_answer_key: str = Field(default="answer")
    evaluator_config: EvaluatorConfig = Field(default_factory=EvaluatorConfig)


class EvaluationResponse(BaseModel):
    evaluation_task_id: str
    metrics: Dict[str, float] = Field(default_factory=dict)
    model_name: str = ""
    status: str = "pending"
    process_time: float = 0.0
    additional_info: Optional[Dict[str, Any]] = None


def _load_dataset(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"数据集文件不存在: {path}")
    if not str(path).lower().endswith(".json"):
        raise ValueError("仅支持 .json 数据集")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("数据集 JSON 必须为对象数组")
    return data


def _normalize_rows(
    rows: List[Dict[str, Any]],
    question_key: str,
    model_answer_key: str,
    ground_true_answer_key: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        merged = dict(row)
        merged["question"] = row.get(question_key, "")
        merged["output"] = row.get(model_answer_key, "")
        merged["answer"] = row.get(ground_true_answer_key, "")
        merged["prompt"] = merged["question"]
        merged["code"] = merged["output"]
        out.append(merged)
    return out


def _aggregate_medqa(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}
    acc = sum(1 for r in rows if r.get("is_correct")) / len(rows)
    return {"accuracy": float(acc), "num_samples": float(len(rows))}


def _aggregate_bleu(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}
    return {
        "bleu": float(sum(r.get("bleu", 0) for r in rows) / len(rows)),
        "rougeL": float(sum(r.get("rouge", 0) for r in rows) / len(rows)),
        "perplexity": float(sum(r.get("perplexity", 0) for r in rows) / len(rows)),
    }


def _aggregate_math(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}
    acc = sum(float(r.get("math_accuracy", 0)) for r in rows) / len(rows)
    return {"math_accuracy": float(acc), "num_samples": float(len(rows))}


def _aggregate_code(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}
    passed = sum(1 for r in rows if r.get("success"))
    return {
        "sample_pass_rate": float(passed / len(rows)),
        "mean_pass_rate": float(sum(float(r.get("pass_rate", 0)) for r in rows) / len(rows)),
        "num_samples": float(len(rows)),
    }


def _aggregate_dpo(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}
    out: Dict[str, float] = {"num_samples": float(len(rows))}
    for k in rows[0].keys():
        vals = [float(r[k]) for r in rows if k in r and isinstance(r[k], (int, float))]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out


def _get_evaluator_class(task_type: EvaluationTaskType):
    if task_type == EvaluationTaskType.MEDQA_LLM:
        from utils.medqa_llm_evaluator import MedQALLMEvaluator
        return MedQALLMEvaluator
    elif task_type == EvaluationTaskType.BLEU_ROUGE:
        from utils.bleu_rouge_evaluator import BleuRougeEvaluator
        return BleuRougeEvaluator
    elif task_type == EvaluationTaskType.MATH:
        from utils.math_evaluator import MathEvaluator
        return MathEvaluator
    elif task_type == EvaluationTaskType.CODE:
        from utils.code_evaluator import CodeEvaluator
        return CodeEvaluator
    elif task_type == EvaluationTaskType.DPO_QUALITY:
        from utils.dpo_quality_evaluator import DPOQualityEvaluator
        return DPOQualityEvaluator
    else:
        raise ValueError(f"不支持的评估任务类型: {task_type}")


def execute_evaluation_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    prev_cwd = Path.cwd()
    os.chdir(project_root)
    t0 = time.time()

    task_type = EvaluationTaskType(payload["task_type"])
    dataset_path = payload["dataset_path"]
    question_key = payload["question_key"]
    model_answer_key = payload["model_answer_key"]
    ground_true_answer_key = payload["ground_true_answer_key"]
    eval_cfg = payload["evaluator_config"]

    raw = _load_dataset(dataset_path)
    samples = _normalize_rows(raw, question_key, model_answer_key, ground_true_answer_key)

    vllm_ok = is_valid_vllm_triple(
        eval_cfg.get("vllm_model_name"),
        eval_cfg.get("vllm_base_url"),
        eval_cfg.get("vllm_api_key"),
    )
    model_path = (eval_cfg.get("model_name_or_path") or "").strip()
    if not vllm_ok and not model_path:
        raise ValueError("vLLM 配置无效且未提供 model_name_or_path，无法评估")

    config = dict(eval_cfg)
    config["_vllm_config_valid"] = vllm_ok
    config["_use_vllm_api"] = vllm_ok
    if vllm_ok and not model_path:
        config["model_name_or_path"] = "__vllm_only__"

    _mt = os.environ.get("EVAL_MAX_NEW_TOKENS")
    if _mt and str(_mt).strip().isdigit():
        config["max_new_tokens"] = int(_mt)

    if task_type == EvaluationTaskType.DPO_QUALITY:
        config["query_key"] = "question"
        config["chosen_key"] = "output"
        config["rejected_key"] = "answer"

    ev = None
    additional: Dict[str, Any] = {"num_rows": len(samples), "use_vllm_api": vllm_ok}
    metrics: Dict[str, float] = {}

    try:
        eval_cls = _get_evaluator_class(task_type)
        ev = eval_cls(config)
        rows_out = [ev.evaluate_one_sample(s) for s in samples]

        if task_type == EvaluationTaskType.MEDQA_LLM:
            metrics = _aggregate_medqa(rows_out)
        elif task_type == EvaluationTaskType.BLEU_ROUGE:
            metrics = _aggregate_bleu(rows_out)
        elif task_type == EvaluationTaskType.MATH:
            metrics = _aggregate_math(rows_out)
        elif task_type == EvaluationTaskType.CODE:
            metrics = _aggregate_code(rows_out)
        elif task_type == EvaluationTaskType.DPO_QUALITY:
            metrics = _aggregate_dpo(rows_out)
    finally:
        if ev is not None and hasattr(ev, "cleanup") and callable(getattr(ev, "cleanup")):
            try:
                ev.cleanup()
            except Exception:
                pass
        os.chdir(prev_cwd)

    elapsed = time.time() - t0
    mname = display_model_name(
        vllm_valid=vllm_ok,
        model_name=eval_cfg.get("vllm_model_name"),
        model_path=model_path,
    )
    return {
        "metrics": metrics,
        "process_time": elapsed,
        "additional_info": additional,
        "model_name": mname,
    }


set_evaluation_task_fn(execute_evaluation_task)


@router.post("/evaluate_model", response_model=EvaluationResponse)
async def evaluate_model(request: EvaluationRequest) -> EvaluationResponse:
    await start_evaluation_worker()
    task_id = (request.evaluation_task_id or "").strip() or str(uuid.uuid4())

    vllm_ok = is_valid_vllm_triple(
        request.evaluator_config.vllm_model_name,
        request.evaluator_config.vllm_base_url,
        request.evaluator_config.vllm_api_key,
    )
    display_name = display_model_name(
        vllm_valid=vllm_ok,
        model_name=request.evaluator_config.vllm_model_name,
        model_path=request.evaluator_config.model_name_or_path or "",
    )
    register_pending_task(task_id, display_name)

    await evaluation_task_queue.put({
        "evaluation_task_id": task_id,
        "task_type": request.task_type.value,
        "dataset_path": request.dataset_path,
        "question_key": request.question_key,
        "model_answer_key": request.model_answer_key,
        "ground_true_answer_key": request.ground_true_answer_key,
        "evaluator_config": request.evaluator_config.model_dump(),
    })

    snap = get_task_snapshot(task_id) or {}
    return EvaluationResponse(
        evaluation_task_id=task_id,
        metrics=snap.get("metrics") or {},
        model_name=str(snap.get("model_name") or display_name),
        status=snap.get("status", "pending"),
        process_time=float(snap.get("process_time") or 0.0),
        additional_info=snap.get("additional_info"),
    )


@router.get("/get_evaluation_result/{evaluation_task_id}", response_model=EvaluationResponse)
async def get_evaluation_result(evaluation_task_id: str) -> EvaluationResponse:
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
