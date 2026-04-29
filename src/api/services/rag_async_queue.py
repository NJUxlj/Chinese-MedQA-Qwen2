"""RAG 服务异步队列与结果管理

支持高并发场景：
- 使用 asyncio.Queue 接收请求（支持无限并发连接）
- 使用 ThreadPoolExecutor 执行 CPU/GPU 密集型 RAG 计算
- 使用 asyncio.Lock 保护共享状态
- 使用 threading.Lock 保护线程安全操作

架构：
    Client (50+ concurrent) 
        -> FastAPI Endpoint (async)
        -> asyncio.Queue (request buffer)
        -> Worker Pool (ThreadPoolExecutor)
        -> RAG Service (CPU/GPU bound)
        -> Result Store (asyncio.Future)
"""

from __future__ import annotations

import asyncio
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable

from utils.logger import setup_logger

logger = setup_logger(__name__, level="INFO")


@dataclass
class RagTask:
    """RAG 任务数据结构"""
    task_id: str
    query: str
    model_name: Optional[str] = None
    top_k: int = 5
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())
    created_at: float = field(default_factory=time.time)


# ============================================================================
# 全局状态（使用锁保护）
# ============================================================================

# 请求队列：存储待处理的 RAG 任务
rag_task_queue: asyncio.Queue[RagTask] = asyncio.Queue()

# 线程池：执行 CPU/GPU 密集型 RAG 计算
# max_workers 控制并发计算的 RAG 任务数，根据 GPU 显存和 CPU 核心数调整
_rag_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()

# Worker 任务引用
_worker_task: Optional[asyncio.Task] = None
_worker_lock = asyncio.Lock()

# RAG 服务获取函数（由外部注入）
_rag_service_fn: Optional[Callable[[], Any]] = None
_service_lock = threading.Lock()

# 统计信息（使用锁保护）
_stats_lock = threading.Lock()
_stats: Dict[str, Any] = {
    "total_requests": 0,
    "completed_requests": 0,
    "failed_requests": 0,
    "queue_size_history": [],
}


def set_rag_service_fn(fn: Callable[[], Any]) -> None:
    """注入 RAG 服务获取函数"""
    global _rag_service_fn
    with _service_lock:
        _rag_service_fn = fn
    logger.info("RAG 服务函数已注册")


def _get_rag_executor() -> ThreadPoolExecutor:
    """获取或创建线程池（线程安全）"""
    global _rag_executor
    with _executor_lock:
        if _rag_executor is None or _rag_executor._shutdown:
            # max_workers: 控制同时执行的 RAG 任务数
            # 建议值：GPU 显存 / 单个 RAG 任务显存占用，或 CPU 核心数
            max_workers = 4  # 可根据实际情况调整
            _rag_executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="rag_worker_"
            )
            logger.info(f"RAG 线程池已创建，max_workers={max_workers}")
        return _rag_executor


def _execute_rag_task_sync(task: RagTask) -> Dict[str, Any]:
    """在线程池中同步执行 RAG 任务
    
    注意：此函数在 ThreadPoolExecutor 中运行，不在 asyncio 事件循环中
    """
    with _service_lock:
        if _rag_service_fn is None:
            raise RuntimeError("RAG 服务函数未注册")
        rag_service = _rag_service_fn()
    
    start_time = time.time()
    
    try:
        # 执行 RAG 计算（CPU/GPU 密集型）
        response = rag_service.generate_response(
            query=task.query,
            model_name=task.model_name,
            top_k=task.top_k
        )
        
        process_time = time.time() - start_time
        
        # 更新统计
        with _stats_lock:
            _stats["completed_requests"] += 1
        
        return {
            "success": True,
            "response": response.get("response", ""),
            "context": response.get("context", ""),
            "source_documents": response.get("source_documents", []),
            "model": task.model_name or "default",
            "process_time": process_time,
        }
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"RAG 任务执行失败 [task_id={task.task_id}]: {e}\n{traceback.format_exc()}")
        
        with _stats_lock:
            _stats["failed_requests"] += 1
        
        return {
            "success": False,
            "error": str(e),
            "process_time": process_time,
        }


async def _worker_loop() -> None:
    """RAG Worker 主循环
    
    从队列中获取任务，提交到线程池执行
    """
    executor = _get_rag_executor()
    
    logger.info("RAG Worker 循环已启动")
    
    while True:
        try:
            # 从队列获取任务（阻塞等待）
            task: RagTask = await rag_task_queue.get()
            
            # 提交到线程池执行
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(executor, _execute_rag_task_sync, task)
            
            # 设置回调，将结果写入 task.future
            async def _on_complete(f: asyncio.Future, t: RagTask):
                try:
                    result = await f
                    if not t.future.done():
                        t.future.set_result(result)
                except Exception as e:
                    if not t.future.done():
                        t.future.set_exception(e)
                finally:
                    rag_task_queue.task_done()
            
            # 创建后台任务处理完成回调
            asyncio.create_task(_on_complete(future, task))
            
        except asyncio.CancelledError:
            logger.info("RAG Worker 循环收到取消信号")
            break
        except Exception as e:
            logger.error(f"RAG Worker 循环异常: {e}\n{traceback.format_exc()}")


async def start_rag_worker() -> None:
    """启动 RAG Worker（幂等）"""
    global _worker_task
    
    async with _worker_lock:
        if _worker_task is None or _worker_task.done():
            _worker_task = asyncio.create_task(_worker_loop())
            logger.info("RAG Worker 已启动")


async def stop_rag_worker() -> None:
    """停止 RAG Worker"""
    global _worker_task, _rag_executor
    
    async with _worker_lock:
        if _worker_task and not _worker_task.done():
            _worker_task.cancel()
            try:
                await _worker_task
            except asyncio.CancelledError:
                pass
            _worker_task = None
            logger.info("RAG Worker 已停止")
    
    # 关闭线程池
    with _executor_lock:
        if _rag_executor and not _rag_executor._shutdown:
            _rag_executor.shutdown(wait=True)
            _rag_executor = None
            logger.info("RAG 线程池已关闭")


async def submit_rag_task(
    query: str,
    model_name: Optional[str] = None,
    top_k: int = 5,
    timeout: Optional[float] = 300.0
) -> Dict[str, Any]:
    """提交 RAG 任务并等待结果
    
    Args:
        query: 查询问题
        model_name: 模型名称
        top_k: 检索文档数
        timeout: 超时时间（秒）
        
    Returns:
        RAG 结果字典
        
    Raises:
        TimeoutError: 任务超时
        Exception: 任务执行失败
    """
    # 确保 worker 已启动
    await start_rag_worker()
    
    # 创建任务
    task = RagTask(
        task_id=str(uuid.uuid4()),
        query=query,
        model_name=model_name,
        top_k=top_k
    )
    
    # 更新统计
    with _stats_lock:
        _stats["total_requests"] += 1
        _stats["queue_size_history"].append(rag_task_queue.qsize())
        # 限制历史记录长度
        if len(_stats["queue_size_history"]) > 1000:
            _stats["queue_size_history"] = _stats["queue_size_history"][-500:]
    
    # 提交到队列
    await rag_task_queue.put(task)
    
    logger.debug(f"RAG 任务已提交 [task_id={task.task_id}], 队列大小: {rag_task_queue.qsize()}")
    
    # 等待结果
    try:
        result = await asyncio.wait_for(task.future, timeout=timeout)
        return result
    except asyncio.TimeoutError:
        logger.warning(f"RAG 任务超时 [task_id={task.task_id}]")
        raise TimeoutError(f"RAG 任务执行超时（>{timeout}秒）")


def get_rag_stats() -> Dict[str, Any]:
    """获取 RAG 服务统计信息"""
    with _stats_lock:
        stats = _stats.copy()
        stats["current_queue_size"] = rag_task_queue.qsize()
        stats["worker_running"] = _worker_task is not None and not _worker_task.done()
    return stats


def get_queue_size() -> int:
    """获取当前队列大小"""
    return rag_task_queue.qsize()
