"""
评估任务的同步执行入口。

由 async_eval_queue 的 worker 通过 asyncio.to_thread() 调用，
在独立线程中运行（不阻塞事件循环）。内部各 evaluator 可再开
ThreadPoolExecutor 并发处理样本，与异步队列完全兼容。
"""

from __future__ import annotations

import time
from typing import Any, Dict

from evaluation.evaluator_factory import EvaluatorFactory
from evaluation.task_types import EvaluationTaskType
from utils.logger import setup_logger

logger = setup_logger(__name__, level="INFO")


def run_evaluation_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    同步评估入口，被 asyncio.to_thread() 调用。

    Args:
        payload: 由 evaluation 路由 put 进队列的字典，包含:
            - evaluation_task_id
            - task_type
            - dataset_path
            - question_key
            - model_answer_key
            - ground_true_answer_key
            - model_name
            - base_url
            - api_key
            - temperature (optional)
            - max_tokens (optional)
            - max_workers (optional)

    Returns:
        包含 metrics / process_time / model_name / additional_info 的字典
    """
    task_type_str: str = payload.get("task_type", EvaluationTaskType.MEDQA.value)
    model_name: str = payload.get("model_name", "")

    config: Dict[str, Any] = {
        "dataset_path":            payload.get("dataset_path", ""),
        "question_key":            payload.get("question_key", "question"),
        "model_answer_key":        payload.get("model_answer_key", "output"),
        "ground_true_answer_key":  payload.get("ground_true_answer_key", "answer"),
        "model_name":              model_name,
        "base_url":                payload.get("base_url", ""),
        "api_key":                 payload.get("api_key", ""),
        "temperature":             payload.get("temperature", 0.7),
        "max_tokens":              payload.get("max_tokens", 2048),
        "max_workers":             payload.get("max_workers", 4),
    }

    logger.info(
        "开始评估任务 [%s] task_type=%s dataset=%s model=%s",
        payload.get("evaluation_task_id", ""),
        task_type_str,
        config["dataset_path"],
        model_name,
    )

    t0 = time.perf_counter()
    evaluator = EvaluatorFactory.create(EvaluationTaskType(task_type_str), config)
    metrics: Dict[str, float] = evaluator.evaluate()
    elapsed = time.perf_counter() - t0

    logger.info(
        "评估任务 [%s] 完成，耗时 %.2fs，指标: %s",
        payload.get("evaluation_task_id", ""),
        elapsed,
        metrics,
    )

    return {
        "metrics": metrics,
        "process_time": elapsed,
        "model_name": model_name,
        "additional_info": {},
    }
