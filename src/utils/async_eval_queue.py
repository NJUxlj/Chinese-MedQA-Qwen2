"""评估任务异步队列与内存结果表。"""

from __future__ import annotations

import asyncio
import traceback
from typing import Any, Dict, Optional, Callable

from utils.logger import setup_logger

logger = setup_logger(__name__, level="INFO")

evaluation_task_queue: asyncio.Queue = asyncio.Queue()

EVALUATION_TASKS: Dict[str, Dict[str, Any]] = {}

_worker_task: Optional[asyncio.Task] = None

_evaluation_task_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None


def set_evaluation_task_fn(fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
    global _evaluation_task_fn
    _evaluation_task_fn = fn


async def _worker_loop() -> None:
    while True:
        payload: Dict[str, Any] = await evaluation_task_queue.get()
        tid: str = payload["evaluation_task_id"]
        try:
            entry = EVALUATION_TASKS.get(tid)
            if entry is None:
                continue
            entry["status"] = "running"
            if _evaluation_task_fn is None:
                raise RuntimeError("evaluation_task_fn 未注册")
            result = await asyncio.to_thread(_evaluation_task_fn, payload)
            entry["metrics"] = result.get("metrics", {})
            entry["process_time"] = result.get("process_time", 0.0)
            entry["additional_info"] = {**(entry.get("additional_info") or {}), **(result.get("additional_info") or {})}
            entry["model_name"] = result.get("model_name", entry.get("model_name", ""))
            entry["status"] = "stop"
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("评估任务失败: %s\n%s", e, traceback.format_exc())
            if tid in EVALUATION_TASKS:
                EVALUATION_TASKS[tid]["status"] = "stop"
                EVALUATION_TASKS[tid]["additional_info"] = {
                    **(EVALUATION_TASKS[tid].get("additional_info") or {}),
                    "error": str(e),
                }
        finally:
            evaluation_task_queue.task_done()


async def start_evaluation_worker() -> None:
    global _worker_task
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_worker_loop())
        logger.info("评估异步队列 worker 已启动")


async def stop_evaluation_worker() -> None:
    global _worker_task
    if _worker_task and not _worker_task.done():
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
        _worker_task = None
        logger.info("评估异步队列 worker 已停止")


def register_pending_task(task_id: str, model_name: str) -> None:
    EVALUATION_TASKS[task_id] = {
        "evaluation_task_id": task_id,
        "metrics": {},
        "model_name": model_name,
        "status": "pending",
        "process_time": 0.0,
        "additional_info": {},
    }


def get_task_snapshot(task_id: str) -> Optional[Dict[str, Any]]:
    return EVALUATION_TASKS.get(task_id)
