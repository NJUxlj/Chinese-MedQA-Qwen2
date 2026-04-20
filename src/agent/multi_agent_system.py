"""
医疗多Agent协作系统 - 2025年最佳实践实现

本模块实现符合2025年主流多Agent系统架构的医疗问答系统，包含：
- 状态图工作流编排（LangGraph模式）
- MCP（Multi-Agent Communication Protocol）协议
- 分层记忆系统（短期/长期/人格记忆）
- 完整可观测性体系（Tracing、Metrics、Logging）
- 动态任务路由与自适应协作
- Agent生命周期管理与健康检查
- 错误恢复与容错机制
- 异步消息通信（事件驱动架构）
"""

from typing import Dict, Any, Optional, List, Union, Tuple, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
import time
import uuid
import json
import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from abc import ABC, abstractmethod
import weakref
import threading
from concurrent.futures import ThreadPoolExecutor
import copy

from .base_agent import BaseAgent
from .agent_factory import AgentFactory
from .agent_manager import AgentManager
from ..utils.logger import setup_logger
from providers import LLMProvider
from ..rag.rag_pipeline import RAGPipeline

logger = setup_logger(__name__, level="INFO")


class AgentState(Enum):
    """Agent状态枚举"""
    IDLE = auto()
    RUNNING = auto()
    WAITING = auto()
    ERROR = auto()
    TERMINATED = auto()
    HEALTHY = auto()
    UNHEALTHY = auto()


class MessageType(Enum):
    """MCP消息类型"""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    BROADCAST = "broadcast"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    ACK = "ack"


class MessagePriority(Enum):
    """消息优先级"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class ProtocolVersion(Enum):
    """MCP协议版本"""
    V1_0 = "1.0"
    V1_1 = "1.1"


class ProtocolErrorCode(Enum):
    """协议错误代码"""
    INVALID_MESSAGE = "INVALID_MESSAGE"
    UNKNOWN_MESSAGE_TYPE = "UNKNOWN_MESSAGE_TYPE"
    MISSING_FIELD = "MISSING_FIELD"
    INVALID_CHECKSUM = "INVALID_CHECKSUM"
    TIMEOUT = "TIMEOUT"
    AGENT_UNAVAILABLE = "AGENT_UNAVAILABLE"
    QUEUE_FULL = "QUEUE_FULL"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass
class ProtocolConfig:
    """MCP协议配置"""
    version: str = ProtocolVersion.V1_1.value
    max_message_size: int = 1024 * 1024
    request_timeout: float = 30.0
    heartbeat_interval: float = 10.0
    enable_compression: bool = False
    enable_encryption: bool = False


class MessageHandler:
    """消息处理器基类"""
    
    async def handle(self, message: AgentMessage, 
                    message_bus: MessageBus) -> Optional[AgentMessage]:
        """处理消息
        
        Args:
            message: 收到的消息
            message_bus: 消息总线实例
            
        Returns:
            可选的响应消息
        """
        raise NotImplementedError


class RequestHandler(MessageHandler):
    """请求消息处理器"""
    
    async def handle(self, message: AgentMessage,
                    message_bus: MessageBus) -> AgentMessage:
        """处理请求消息"""
        return message.create_response(
            {"status": "received", "handler": "RequestHandler"},
            success=True
        )


class ResponseHandler(MessageHandler):
    """响应消息处理器"""
    
    async def handle(self, message: AgentMessage,
                    message_bus: MessageBus) -> None:
        """处理响应消息"""
        if message.correlation_id:
            logger.debug(f"收到响应: {message.correlation_id}")


class EventHandler(MessageHandler):
    """事件消息处理器"""
    
    async def handle(self, message: AgentMessage,
                    message_bus: MessageBus) -> None:
        """处理事件消息"""
        logger.info(f"收到事件: {message.content.get('event_type')}")


class HeartbeatHandler(MessageHandler):
    """心跳消息处理器"""
    
    async def handle(self, message: AgentMessage,
                    message_bus: MessageBus) -> Optional[AgentMessage]:
        """处理心跳消息"""
        if message.msg_type == MessageType.HEARTBEAT:
            return message.create_response(
                {"heartbeat": "alive"},
                success=True
            )
        return None


class ErrorHandler(MessageHandler):
    """错误消息处理器"""
    
    async def handle(self, message: AgentMessage,
                    message_bus: MessageBus) -> None:
        """处理错误消息"""
        logger.error(
            f"收到错误消息: {message.content.get('error_code')} - "
            f"{message.content.get('error_message')}"
        )


class MCPProtocolDispatcher:
    """MCP协议调度器
    
    负责将消息路由到正确的处理器，支持多种消息类型的自动分发。
    """
    
    def __init__(self, config: Optional[ProtocolConfig] = None):
        self.config = config or ProtocolConfig()
        self._handlers: Dict[MessageType, MessageHandler] = {}
        self._custom_handlers: Dict[str, MessageHandler] = {}
        self._error_handlers: List[MessageHandler] = []
        self._lock = threading.RLock()
        
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """注册默认处理器"""
        self._handlers[MessageType.REQUEST] = RequestHandler()
        self._handlers[MessageType.RESPONSE] = ResponseHandler()
        self._handlers[MessageType.EVENT] = EventHandler()
        self._handlers[MessageType.HEARTBEAT] = HeartbeatHandler()
        self._handlers[MessageType.ERROR] = ErrorHandler()

    def register_handler(self, message_type: MessageType, 
                        handler: MessageHandler) -> None:
        """注册消息类型处理器"""
        with self._lock:
            self._handlers[message_type] = handler

    def register_custom_handler(self, handler_id: str,
                               handler: MessageHandler) -> None:
        """注册自定义处理器"""
        with self._lock:
            self._custom_handlers[handler_id] = handler

    def register_error_handler(self, handler: MessageHandler) -> None:
        """注册错误处理器"""
        with self._lock:
            self._error_handlers.append(handler)

    async def dispatch(self, message: AgentMessage,
                      message_bus: MessageBus) -> Optional[AgentMessage]:
        """分发消息到对应处理器
        
        Args:
            message: 要分发的消息
            message_bus: 消息总线实例
            
        Returns:
            可选的响应消息
        """
        try:
            if self.config.version and message.protocol_version != self.config.version:
                logger.warning(
                    f"协议版本不匹配: 期望 {self.config.version}, "
                    f"收到 {message.protocol_version}"
                )

            valid, error_msg = message.validate()
            if not valid:
                error_msg_obj = AgentMessage(
                    msg_id=str(uuid.uuid4()),
                    msg_type=MessageType.ERROR,
                    sender_id=message.receiver_id,
                    receiver_id=message.sender_id,
                    content={
                        "error_code": ProtocolErrorCode.INVALID_MESSAGE.value,
                        "error_message": error_msg
                    },
                    correlation_id=message.correlation_id
                )
                await self._handle_error(error_msg_obj, message_bus)
                return error_msg_obj

            handler = None
            if message.msg_type in self._handlers:
                handler = self._handlers[message.msg_type]
            elif message.metadata.get("custom_handler"):
                handler = self._custom_handlers.get(
                    message.metadata["custom_handler"]
                )

            if handler:
                return await handler.handle(message, message_bus)
            
            return message.create_response(
                {"error": "No handler found"},
                success=False
            )

        except Exception as e:
            logger.error(f"消息处理异常: {e}")
            return await self._handle_exception(message, e, message_bus)

    async def _handle_error(self, error_msg: AgentMessage,
                          message_bus: MessageBus) -> None:
        """处理错误消息"""
        for handler in self._error_handlers:
            try:
                await handler.handle(error_msg, message_bus)
            except Exception as e:
                logger.error(f"错误处理器执行失败: {e}")

    async def _handle_exception(self, message: AgentMessage,
                               exception: Exception,
                               message_bus: MessageBus) -> AgentMessage:
        """处理异常"""
        error_msg = message.create_error(
            ProtocolErrorCode.INTERNAL_ERROR.value,
            str(exception)
        )
        await self._handle_error(error_msg, message_bus)
        return error_msg

    def get_handler_info(self) -> Dict[str, Any]:
        """获取处理器信息"""
        with self._lock:
            return {
                "default_handlers": {
                    mt.value: handler.__class__.__name__ 
                    for mt, handler in self._handlers.items()
                },
                "custom_handlers": list(self._custom_handlers.keys()),
                "error_handlers_count": len(self._error_handlers),
                "config": {
                    "version": self.config.version,
                    "max_message_size": self.config.max_message_size,
                    "request_timeout": self.config.request_timeout
                }
            }


class AgentSession:
    """Agent通信会话
    
    管理Agent之间的通信会话，支持请求-响应模式。
    """
    
    def __init__(self, session_id: str, agent_id: str,
                message_bus: MessageBus):
        self.session_id = session_id
        self.agent_id = agent_id
        self.message_bus = message_bus
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._created_at = time.time()
        self._last_activity = time.time()
        self._lock = asyncio.Lock()
        self._is_active = True

    async def send_request(self, target_agent: str, 
                          content: Dict[str, Any],
                          timeout: Optional[float] = None) -> Optional[AgentMessage]:
        """发送请求并等待响应"""
        async with self._lock:
            if not self._is_active:
                raise ValueError("Session is not active")
            
            future = asyncio.Future()
            request_id = str(uuid.uuid4())
            self._pending_requests[request_id] = future
            
            message = AgentMessage(
                msg_id=request_id,
                msg_type=MessageType.REQUEST,
                sender_id=self.agent_id,
                receiver_id=target_agent,
                content=content,
                correlation_id=request_id,
                priority=MessagePriority.NORMAL.value
            )
            
            self._last_activity = time.time()
            
            try:
                await self.message_bus.publish(f"agent.{target_agent}", message)
                
                response = await asyncio.wait_for(
                    future,
                    timeout=timeout or 30.0
                )
                return response
            except asyncio.TimeoutError:
                logger.warning(f"请求超时: {request_id}")
                del self._pending_requests[request_id]
                return None
            except Exception as e:
                logger.error(f"请求失败: {e}")
                del self._pending_requests[request_id]
                raise
            finally:
                self._pending_requests.pop(request_id, None)

    async def send_response(self, original_message: AgentMessage,
                           content: Dict[str, Any],
                           success: bool = True) -> None:
        """发送响应"""
        response = original_message.create_response(content, success)
        await self.message_bus.publish(
            f"agent.{response.receiver_id}",
            response
        )
        self._last_activity = time.time()

    def complete_request(self, request_id: str, response: AgentMessage) -> bool:
        """完成挂起的请求"""
        future = self._pending_requests.get(request_id)
        if future and not future.done():
            future.set_result(response)
            return True
        return False

    def close(self) -> None:
        """关闭会话"""
        self._is_active = False
        
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        
        self._pending_requests.clear()
        logger.info(f"会话已关闭: {self.session_id}")

    @property
    def is_active(self) -> bool:
        """检查会话是否活跃"""
        return self._is_active

    @property
    def duration(self) -> float:
        """获取会话持续时间"""
        return time.time() - self._created_at


class SessionManager:
    """会话管理器"""
    
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self._sessions: Dict[str, AgentSession] = {}
        self._agent_sessions: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.Lock()
        self._max_sessions_per_agent = 10

    def create_session(self, agent_id: str) -> AgentSession:
        """为Agent创建新会话"""
        with self._lock:
            if len(self._agent_sessions.get(agent_id, set())) >= self._max_sessions_per_agent:
                oldest = min(
                    self._agent_sessions[agent_id],
                    key=lambda s: self._sessions[s]._created_at
                )
                self._sessions[oldest].close()
                del self._sessions[oldest]
                self._agent_sessions[agent_id].discard(oldest)
            
            session_id = f"{agent_id}_{uuid.uuid4().hex[:8]}"
            session = AgentSession(session_id, agent_id, self.message_bus)
            
            self._sessions[session_id] = session
            self._agent_sessions[agent_id].add(session_id)
            
            return session

    def get_session(self, session_id: str) -> Optional[AgentSession]:
        """获取会话"""
        return self._sessions.get(session_id)

    def close_session(self, session_id: str) -> bool:
        """关闭会话"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.close()
                self._agent_sessions[session.agent_id].discard(session_id)
                del self._sessions[session_id]
                return True
            return False

    def close_all_sessions(self, agent_id: Optional[str] = None) -> int:
        """关闭所有会话"""
        with self._lock:
            if agent_id:
                session_ids = list(self._agent_sessions.get(agent_id, set()))
            else:
                session_ids = list(self._sessions.keys())
            
            for session_id in session_ids:
                self.close_session(session_id)
            
            return len(session_ids)

    def get_stats(self) -> Dict[str, Any]:
        """获取会话统计"""
        with self._lock:
            return {
                "total_sessions": len(self._sessions),
                "sessions_by_agent": {
                    agent: len(sessions) 
                    for agent, sessions in self._agent_sessions.items()
                },
                "active_sessions": sum(
                    1 for s in self._sessions.values() if s.is_active
                )
            }


class TaskComplexity(Enum):
    """任务复杂度等级"""
    SIMPLE = 0
    MODERATE = 1
    COMPLEX = 2
    VERY_COMPLEX = 3


class CollaborationPattern(Enum):
    """协作模式"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


class TaskType(Enum):
    """任务类型"""
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    PREVENTION = "prevention"
    EXAMINATION = "examination"
    ROUTING = "routing"
    COORDINATION = "coordination"
    GENERAL = "general"


@dataclass
class TaskContext:
    """任务上下文"""
    task_id: str
    original_query: str
    complexity: TaskComplexity = TaskComplexity.MODERATE
    task_type: TaskType = TaskType.GENERAL
    required_capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_task_id: Optional[str] = None
    sub_tasks: List['TaskContext'] = field(default_factory=list)
    final_result: Optional[Dict[str, Any]] = None


@dataclass
class AgentCapability:
    """Agent能力描述"""
    agent_id: str
    agent_type: str
    capabilities: List[str]
    performance_score: float = 0.8
    current_load: int = 0
    success_rate: float = 0.9
    avg_response_time: float = 1.0
    specialties: List[str] = field(default_factory=list)
    availability: float = 1.0


class DynamicRouter:
    """动态任务路由器
    
    基于任务特征和Agent能力动态选择最佳执行路径。
    支持多种路由策略和自适应学习。
    """

    def __init__(self, registry: 'AgentRegistry',
                enable_learning: bool = True,
                learning_rate: float = 0.1):
        self.registry = registry
        self.enable_learning = enable_learning
        self.learning_rate = learning_rate
        
        self._routing_history: List[Dict[str, Any]] = []
        self._agent_scores: Dict[str, float] = defaultdict(float)
        self._task_patterns: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.RLock()
        
        self._init_default_routing_rules()

    def _init_default_routing_rules(self) -> None:
        """初始化默认路由规则"""
        self._routing_rules = {
            TaskType.DIAGNOSIS: {
                "capabilities": ["diagnosis", "symptom_analysis", "medical_knowledge"],
                "preferred_agents": ["诊断Agent"],
                "complexity_threshold": TaskComplexity.MODERATE
            },
            TaskType.TREATMENT: {
                "capabilities": ["treatment", "pharmacology", "therapy_planning"],
                "preferred_agents": ["治疗Agent"],
                "complexity_threshold": TaskComplexity.MODERATE
            },
            TaskType.PREVENTION: {
                "capabilities": ["prevention", "health_education", "lifestyle"],
                "preferred_agents": ["预防Agent"],
                "complexity_threshold": TaskComplexity.SIMPLE
            },
            TaskType.EXAMINATION: {
                "capabilities": ["examination", "lab_interpretation", "imaging"],
                "preferred_agents": ["医学检查Agent"],
                "complexity_threshold": TaskComplexity.MODERATE
            }
        }

    def analyze_task(self, query: str,
                    context: Optional[TaskContext] = None) -> TaskAnalysis:
        """分析任务特征
        
        Args:
            query: 用户查询
            context: 任务上下文
            
        Returns:
            任务分析结果
        """
        query_lower = query.lower()
        
        keywords = {
            "diagnosis": ["诊断", "症状", "病因", "鉴别", "可能", "什么病"],
            "treatment": ["治疗", "用药", "药物", "方案", "疗法", "吃药"],
            "prevention": ["预防", "保健", "健康", "养生", "生活方式"],
            "examination": ["检查", "化验", "检验", "结果", "报告", "指标"]
        }
        
        scores = {}
        for task_type, kws in keywords.items():
            score = sum(1 for kw in kws if kw in query_lower)
            scores[task_type] = score
        
        detected_type = max(scores, key=scores.get)
        max_score = scores[detected_type]
        
        if max_score == 0:
            task_type = TaskType.GENERAL
        else:
            task_type = TaskType(detected_type)
        
        complexity_indicators = [
            len(query.split()) > 20,
            "、" in query,
            "和" in query and "或" in query,
            "请分析" in query or "请比较" in query,
            "可能原因" in query or "如何处理" in query
        ]
        
        complexity_score = sum(complexity_indicators)
        if complexity_score <= 1:
            complexity = TaskComplexity.SIMPLE
        elif complexity_score <= 3:
            complexity = TaskComplexity.MODERATE
        else:
            complexity = TaskComplexity.COMPLEX
        
        required_capabilities = self._routing_rules.get(task_type, {}).get(
            "capabilities", []
        )
        
        return TaskAnalysis(
            task_type=task_type,
            complexity=complexity,
            confidence=min(0.95, max_score * 0.2 + 0.3),
            required_capabilities=required_capabilities,
            keywords=[kw for kws in keywords.values() for kw in kws if kw in query_lower],
            suggested_agents=self._get_suggested_agents(task_type),
            routing_strategy=self._determine_strategy(task_type, complexity)
        )

    def _get_suggested_agents(self, task_type: TaskType) -> List[str]:
        """获取建议的Agent列表"""
        rule = self._routing_rules.get(task_type, {})
        return rule.get("preferred_agents", [])

    def _determine_strategy(self, task_type: TaskType,
                           complexity: TaskComplexity) -> CollaborationPattern:
        """确定协作策略"""
        if complexity == TaskComplexity.SIMPLE:
            return CollaborationPattern.SEQUENTIAL
        elif complexity == TaskComplexity.COMPLEX:
            return CollaborationPattern.HIERARCHICAL
        elif task_type in [TaskType.DIAGNOSIS, TaskType.TREATMENT]:
            return CollaborationPattern.PARALLEL
        else:
            return CollaborationPattern.ADAPTIVE

    def select_agents(self, analysis: TaskAnalysis,
                     count: Optional[int] = None) -> List[AgentCapability]:
        """选择最佳Agent组合
        
        Args:
            analysis: 任务分析结果
            count: 需要选择的Agent数量
            
        Returns:
            选中的Agent能力列表
        """
        with self._lock:
            all_agents = self.registry.get_all_agents()
            
            candidates = []
            for agent_id, info in all_agents.items():
                agent_capabilities = info.get("capabilities", [])
                
                capability_score = self._calculate_capability_score(
                    analysis.required_capabilities,
                    agent_capabilities
                )
                
                perf_score = self._agent_scores.get(agent_id, 0.8)
                
                load_factor = 1.0 - (info.get("current_tasks", 0) * 0.1)
                load_factor = max(0.3, load_factor)
                
                success_rate = info.get("success_rate", 0.9)
                
                total_score = (
                    capability_score * 0.4 +
                    perf_score * 0.3 +
                    load_factor * 0.15 +
                    success_rate * 0.15
                )
                
                candidates.append(AgentCapability(
                    agent_id=agent_id,
                    agent_type=info.get("agent_type", "general"),
                    capabilities=agent_capabilities,
                    performance_score=perf_score,
                    current_load=info.get("current_tasks", 0),
                    success_rate=success_rate,
                    avg_response_time=info.get("avg_response_time", 1.0),
                    specialties=info.get("specialties", [])
                ))
            
            candidates.sort(key=lambda x: -x.performance_score)
            
            if count is None:
                if analysis.complexity == TaskComplexity.SIMPLE:
                    count = 1
                elif analysis.complexity == TaskComplexity.MODERATE:
                    count = 2
                else:
                    count = 3
            
            selected = candidates[:count]
            
            for agent in selected:
                self._record_routing(analysis, agent)
            
            return selected

    def _calculate_capability_score(self, required: List[str],
                                   available: List[str]) -> float:
        """计算能力匹配分数"""
        if not required:
            return 0.8
        
        required_set = set(required)
        available_set = set(available)
        
        intersection = required_set & available_set
        union = required_set | available_set
        
        return len(intersection) / len(union) if union else 0

    def _record_routing(self, analysis: TaskAnalysis,
                       agent: AgentCapability) -> None:
        """记录路由决策用于学习"""
        if not self.enable_learning:
            return
        
        self._routing_history.append({
            "task_type": analysis.task_type.value,
            "complexity": analysis.complexity.value,
            "agent_id": agent.agent_id,
            "timestamp": time.time()
        })
        
        if len(self._routing_history) > 1000:
            self._routing_history = self._routing_history[-500:]

    def update_agent_performance(self, agent_id: str,
                                success: bool,
                                response_time: float) -> None:
        """更新Agent性能评估"""
        with self._lock:
            if success:
                self._agent_scores[agent_id] = min(
                    1.0,
                    self._agent_scores[agent_id] * (1 - self.learning_rate) +
                    1.0 * self.learning_rate
                )
            else:
                self._agent_scores[agent_id] = max(
                    0.1,
                    self._agent_scores[agent_id] * (1 - self.learning_rate) +
                    0.5 * self.learning_rate
                )

    def get_routing_stats(self) -> Dict[str, Any]:
        """获取路由统计信息"""
        with self._lock:
            agent_stats = defaultdict(lambda: {"success": 0, "total": 0})
            
            for record in self._routing_history:
                agent_id = record["agent_id"]
                agent_stats[agent_id]["total"] += 1
                if record.get("success"):
                    agent_stats[agent_id]["success"] += 1
            
            return {
                "total_routings": len(self._routing_history),
                "agent_performance": {
                    aid: {
                        "success_rate": stats["success"] / max(1, stats["total"]),
                        "total_tasks": stats["total"]
                    }
                    for aid, stats in agent_stats.items()
                },
                "learning_enabled": self.enable_learning
            }


@dataclass
class TaskAnalysis:
    """任务分析结果"""
    task_type: TaskType
    complexity: TaskComplexity
    confidence: float
    required_capabilities: List[str]
    keywords: List[str]
    suggested_agents: List[str]
    routing_strategy: CollaborationPattern


class AdaptiveCollaborator:
    """自适应协作管理器
    
    根据任务特征动态选择协作模式，支持并行、串行、层级等多种协作方式。
    """

    def __init__(self, message_bus: MessageBus,
                registry: 'AgentRegistry',
                router: DynamicRouter):
        self.message_bus = message_bus
        self.registry = registry
        self.router = router
        
        self._collaboration_history: List[Dict[str, Any]] = []
        self._performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()

    async def execute_parallel(self, task: TaskContext,
                              agents: List[AgentCapability]) -> Dict[str, Any]:
        """并行执行任务
        
        多个Agent同时处理同一任务，然后综合结果。
        """
        span = self.message_bus.observability.start_request(
            operation_name="parallel_collaboration"
        )
        span.set_attribute("task_id", task.task_id)
        span.set_attribute("agent_count", len(agents))

        self.message_bus.metrics.increment("collaboration_parallel_total")

        async def process_with_agent(agent: AgentCapability) -> Dict[str, Any]:
            start_time = time.time()
            
            try:
                message = AgentMessage(
                    msg_id=str(uuid.uuid4()),
                    msg_type=MessageType.REQUEST,
                    sender_id="collaborator",
                    receiver_id=agent.agent_id,
                    content={
                        "task": task.original_query,
                        "task_type": task.task_type.value,
                        "context": task.metadata
                    },
                    priority=MessagePriority.HIGH.value
                )
                
                response = await self.message_bus.request(
                    agent.agent_id,
                    message,
                    timeout=30.0
                )
                
                elapsed = time.time() - start_time
                self.router.update_agent_performance(agent.agent_id, True, elapsed)
                
                return {
                    "agent_id": agent.agent_id,
                    "success": True,
                    "response": response.content.get("data") if response else None,
                    "elapsed": elapsed
                }
            except Exception as e:
                elapsed = time.time() - start_time
                self.router.update_agent_performance(agent.agent_id, False, elapsed)
                
                return {
                    "agent_id": agent.agent_id,
                    "success": False,
                    "error": str(e),
                    "elapsed": elapsed
                }

        tasks = [process_with_agent(agent) for agent in agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        span.set_attribute("results_count", len([r for r in results if isinstance(r, dict) and r.get("success")]))
        
        self.message_bus.observability.end_request(span, "success")

        with self._lock:
            self._collaboration_history.append({
                "pattern": "parallel",
                "task_id": task.task_id,
                "agent_count": len(agents),
                "success_count": sum(1 for r in results if isinstance(r, dict) and r.get("success")),
                "timestamp": time.time()
            })

        return {
            "pattern": "parallel",
            "results": results,
            "synthesized": self._synthesize_results(results)
        }

    async def execute_sequential(self, task: TaskContext,
                                agents: List[AgentCapability],
                                depends_on: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """串行执行任务
        
        Agent按顺序处理任务，每个Agent的输出作为下一个Agent的输入。
        """
        span = self.message_bus.observability.start_request(
            operation_name="sequential_collaboration"
        )
        span.set_attribute("task_id", task.task_id)
        span.set_attribute("agent_count", len(agents))

        self.message_bus.metrics.increment("collaboration_sequential_total")

        current_query = task.original_query
        accumulated_results = []
        
        for i, agent in enumerate(agents):
            start_time = time.time()
            
            try:
                message = AgentMessage(
                    msg_id=str(uuid.uuid4()),
                    msg_type=MessageType.REQUEST,
                    sender_id="collaborator",
                    receiver_id=agent.agent_id,
                    content={
                        "task": current_query,
                        "task_type": task.task_type.value,
                        "step": i + 1,
                        "total_steps": len(agents),
                        "previous_results": accumulated_results[-1:] if accumulated_results else None
                    },
                    priority=MessagePriority.NORMAL.value
                )
                
                response = await self.message_bus.request(
                    agent.agent_id,
                    message,
                    timeout=30.0
                )
                
                elapsed = time.time() - start_time
                self.router.update_agent_performance(agent.agent_id, True, elapsed)
                
                if response:
                    current_query = response.content.get("data", {}).get(
                        "refined_query", current_query
                    )
                    accumulated_results.append({
                        "agent_id": agent.agent_id,
                        "output": response.content.get("data"),
                        "elapsed": elapsed
                    })
                
            except Exception as e:
                elapsed = time.time() - start_time
                self.router.update_agent_performance(agent.agent_id, False, elapsed)
                
                accumulated_results.append({
                    "agent_id": agent.agent_id,
                    "error": str(e),
                    "elapsed": elapsed
                })
                break

        span.set_attribute("completed_steps", len(accumulated_results))
        self.message_bus.observability.end_request(span, "success")

        with self._lock:
            self._collaboration_history.append({
                "pattern": "sequential",
                "task_id": task.task_id,
                "agent_count": len(agents),
                "completed_steps": len(accumulated_results),
                "timestamp": time.time()
            })

        return {
            "pattern": "sequential",
            "results": accumulated_results,
            "final_query": current_query,
            "synthesized": self._synthesize_results(accumulated_results)
        }

    async def execute_hierarchical(self, task: TaskContext,
                                  coordinator: AgentCapability,
                                  workers: List[AgentCapability]) -> Dict[str, Any]:
        """层级执行任务
        
        协调员Agent负责任务分解和结果综合，工作Agent负责具体执行。
        """
        span = self.message_bus.observability.start_request(
            operation_name="hierarchical_collaboration"
        )
        span.set_attribute("task_id", task.task_id)
        span.set_attribute("worker_count", len(workers))

        self.message_bus.metrics.increment("collaboration_hierarchical_total")

        try:
            decomposition_message = AgentMessage(
                msg_id=str(uuid.uuid4()),
                msg_type=MessageType.REQUEST,
                sender_id="collaborator",
                receiver_id=coordinator.agent_id,
                content={
                    "task": task.original_query,
                    "task_type": task.task_type.value,
                    "available_agents": [a.agent_id for a in workers],
                    "action": "decompose"
                },
                priority=MessagePriority.HIGH.value
            )
            
            decomposition_response = await self.message_bus.request(
                coordinator.agent_id,
                decomposition_message,
                timeout=30.0
            )
            
            subtasks = decomposition_response.content.get("data", {}).get(
                "subtasks", []
            )
            
            worker_assignments = decomposition_response.content.get("data", {}).get(
                "assignments", {}
            )
            
            worker_tasks = defaultdict(list)
            for subtask in subtasks:
                assigned_agent = worker_assignments.get(subtask["id"], workers[0].agent_id)
                worker_tasks[assigned_agent].append(subtask)
            
            parallel_results = {}
            for agent_id, agent_tasks in worker_tasks.items():
                agent = next((a for a in workers if a.agent_id == agent_id), workers[0])
                
                subtask_results = []
                for subtask in agent_tasks:
                    result = await self._execute_subtask(agent, subtask)
                    subtask_results.append(result)
                
                parallel_results[agent_id] = subtask_results
            
            synthesis_message = AgentMessage(
                msg_id=str(uuid.uuid4()),
                msg_type=MessageType.REQUEST,
                sender_id="collaborator",
                receiver_id=coordinator.agent_id,
                content={
                    "task": task.original_query,
                    "worker_results": parallel_results,
                    "action": "synthesize"
                },
                priority=MessagePriority.HIGH.value
            )
            
            synthesis_response = await self.message_bus.request(
                coordinator.agent_id,
                synthesis_message,
                timeout=30.0
            )
            
            span.set_attribute("subtask_count", len(subtasks))
            self.message_bus.observability.end_request(span, "success")

            with self._lock:
                self._collaboration_history.append({
                    "pattern": "hierarchical",
                    "task_id": task.task_id,
                    "subtask_count": len(subtasks),
                    "coordinator_id": coordinator.agent_id,
                    "timestamp": time.time()
                })

            return {
                "pattern": "hierarchical",
                "subtasks": subtasks,
                "worker_results": parallel_results,
                "synthesized": synthesis_response.content.get("data") if synthesis_response else None
            }
            
        except Exception as e:
            self.message_bus.observability.end_request(span, "error", {"error": str(e)})
            raise

    async def _execute_subtask(self, agent: AgentCapability,
                              subtask: Dict[str, Any]) -> Dict[str, Any]:
        """执行子任务"""
        start_time = time.time()
        
        try:
            message = AgentMessage(
                msg_id=str(uuid.uuid4()),
                msg_type=MessageType.REQUEST,
                sender_id="collaborator",
                receiver_id=agent.agent_id,
                content={
                    "subtask": subtask,
                    "original_task": subtask.get("description", "")
                },
                priority=MessagePriority.NORMAL.value
            )
            
            response = await self.message_bus.request(
                agent.agent_id,
                message,
                timeout=30.0
            )
            
            elapsed = time.time() - start_time
            self.router.update_agent_performance(agent.agent_id, True, elapsed)
            
            return {
                "subtask_id": subtask["id"],
                "agent_id": agent.agent_id,
                "success": True,
                "result": response.content.get("data") if response else None,
                "elapsed": elapsed
            }
        except Exception as e:
            elapsed = time.time() - start_time
            self.router.update_agent_performance(agent.agent_id, False, elapsed)
            
            return {
                "subtask_id": subtask["id"],
                "agent_id": agent.agent_id,
                "success": False,
                "error": str(e),
                "elapsed": elapsed
            }

    async def execute_adaptive(self, task: TaskContext,
                             agents: List[AgentCapability]) -> Dict[str, Any]:
        """自适应执行任务
        
        根据执行过程中的反馈动态调整协作策略。
        """
        span = self.message_bus.observability.start_request(
            operation_name="adaptive_collaboration"
        )
        span.set_attribute("task_id", task.task_id)

        self.message_bus.metrics.increment("collaboration_adaptive_total")

        initial_analysis = self.router.analyze_task(task.original_query)
        selected_agents = agents[: min(2, len(agents))]
        
        try:
            first_result = await self._quick_evaluation(selected_agents[0], task)
            
            if first_result.get("confidence", 0) > 0.8:
                span.set_attribute("strategy", "single_agent")
                return {
                    "pattern": "adaptive",
                    "strategy": "single_agent",
                    "result": first_result
                }
            
            if len(selected_agents) > 1:
                parallel_results = await self.execute_parallel(task, selected_agents)
                
                confidence = parallel_results["synthesized"].get("confidence", 0.5) if parallel_results.get("synthesized") else 0.5
                
                if confidence < 0.6 and len(agents) > 2:
                    span.set_attribute("strategy", "expanded_parallel")
                    additional_agents = agents[2:4]
                    expanded_results = await self.execute_parallel(task, additional_agents)
                    
                    return {
                        "pattern": "adaptive",
                        "strategy": "expanded_parallel",
                        "initial_results": parallel_results,
                        "expanded_results": expanded_results,
                        "synthesized": self._synthesize_results([
                            parallel_results.get("synthesized", {}),
                            expanded_results.get("synthesized", {})
                        ])
                    }
                
                span.set_attribute("strategy", "parallel")
                return {
                    "pattern": "adaptive",
                    "strategy": "parallel",
                    "results": parallel_results,
                    "synthesized": parallel_results.get("synthesized")
                }
            
            span.set_attribute("strategy", "single_agent_fallback")
            return {
                "pattern": "adaptive",
                "strategy": "single_agent_fallback",
                "result": first_result
            }
            
        except Exception as e:
            self.message_bus.observability.end_request(span, "error", {"error": str(e)})
            raise

    async def _quick_evaluation(self, agent: AgentCapability,
                               task: TaskContext) -> Dict[str, Any]:
        """快速评估单个Agent的处理能力"""
        start_time = time.time()
        
        try:
            message = AgentMessage(
                msg_id=str(uuid.uuid4()),
                msg_type=MessageType.REQUEST,
                sender_id="collaborator",
                receiver_id=agent.agent_id,
                content={
                    "task": task.original_query,
                    "evaluation_mode": True
                },
                priority=MessagePriority.HIGH.value
            )
            
            response = await self.message_bus.request(
                agent.agent_id,
                message,
                timeout=10.0
            )
            
            elapsed = time.time() - start_time
            self.router.update_agent_performance(agent.agent_id, True, elapsed)
            
            return {
                "agent_id": agent.agent_id,
                "success": True,
                "response": response.content.get("data") if response else None,
                "confidence": response.content.get("data", {}).get("confidence", 0.5) if response else 0.3,
                "elapsed": elapsed
            }
        except Exception as e:
            elapsed = time.time() - start_time
            self.router.update_agent_performance(agent.agent_id, False, elapsed)
            
            return {
                "agent_id": agent.agent_id,
                "success": False,
                "error": str(e),
                "confidence": 0.0,
                "elapsed": elapsed
            }

    def _synthesize_results(self, results: Any) -> Dict[str, Any]:
        """综合多个结果"""
        if isinstance(results, list):
            successful = [r for r in results if isinstance(r, dict) and r.get("success")]
            failed = [r for r in results if isinstance(r, dict) and not r.get("success")]
            
            if not successful:
                return {"success": False, "error": "All agents failed"}
            
            responses = [r.get("response") or r.get("result") for r in successful if r.get("response") or r.get("result")]
            
            return {
                "success": True,
                "agent_count": len(successful),
                "failed_count": len(failed),
                "combined_response": responses,
                "confidence": min(0.95, len(successful) * 0.25)
            }
        elif isinstance(results, dict):
            return results
        else:
            return {"success": True, "result": results}

    def get_collaboration_stats(self) -> Dict[str, Any]:
        """获取协作统计信息"""
        with self._lock:
            pattern_stats = defaultdict(lambda: {"count": 0, "success": 0})
            
            for record in self._collaboration_history:
                pattern = record.get("pattern", "unknown")
                pattern_stats[pattern]["count"] += 1
                if record.get("success_count", 0) > 0:
                    pattern_stats[pattern]["success"] += 1
            
            return {
                "total_collaborations": len(self._collaboration_history),
                "by_pattern": {
                    pattern: {
                        "count": stats["count"],
                        "success_rate": stats["success"] / max(1, stats["count"])
                    }
                    for pattern, stats in pattern_stats.items()
                }
            }


@dataclass
class AgentMessage:
    """MCP协议消息格式
    
    完整的MCP（Multi-Agent Communication Protocol）消息定义，
    支持请求-响应模式、事件发布、心跳机制等。
    """
    msg_id: str
    msg_type: MessageType
    sender_id: str
    receiver_id: str
    content: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    protocol_version: str = "1.1"
    sequence_num: Optional[int] = None
    checksum: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.msg_type, str):
            self.msg_type = MessageType(self.msg_type)
        if isinstance(self.priority, MessagePriority):
            self.priority = self.priority.value
        self._generate_checksum()

    def _generate_checksum(self) -> None:
        """生成消息校验和"""
        data = f"{self.msg_id}{self.sender_id}{self.receiver_id}{self.timestamp}"
        self.checksum = hashlib.md5(data.encode()).hexdigest()[:16]

    def validate(self) -> Tuple[bool, str]:
        """验证消息完整性"""
        if not self.msg_id or not isinstance(self.msg_id, str):
            return False, "Invalid msg_id"
        if not self.sender_id or not isinstance(self.sender_id, str):
            return False, "Invalid sender_id"
        if not self.receiver_id or not isinstance(self.receiver_id, str):
            return False, "Invalid receiver_id"
        return True, "Valid"

    def create_response(self, content: Dict[str, Any], 
                       success: bool = True) -> 'AgentMessage':
        """创建响应消息"""
        return AgentMessage(
            msg_id=str(uuid.uuid4()),
            msg_type=MessageType.RESPONSE,
            sender_id=self.receiver_id,
            receiver_id=self.sender_id,
            content={
                "original_msg_id": self.msg_id,
                "success": success,
                "data": content
            },
            correlation_id=self.correlation_id or self.msg_id,
            reply_to=self.msg_id,
            priority=self.priority,
            metadata={"response_to": self.msg_id}
        )

    def create_error(self, error_code: str, error_msg: str) -> 'AgentMessage':
        """创建错误消息"""
        return AgentMessage(
            msg_id=str(uuid.uuid4()),
            msg_type=MessageType.ERROR,
            sender_id=self.receiver_id,
            receiver_id=self.sender_id,
            content={
                "error_code": error_code,
                "error_message": error_msg,
                "original_msg_id": self.msg_id
            },
            correlation_id=self.correlation_id or self.msg_id,
            reply_to=self.msg_id,
            priority=self.priority,
            metadata={"error": True}
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "msg_id": self.msg_id,
            "msg_type": self.msg_type.value,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "content": self.content,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "priority": self.priority,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        return cls(
            msg_id=data["msg_id"],
            msg_type=MessageType(data["msg_type"]),
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            content=data["content"],
            timestamp=data["timestamp"],
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to"),
            priority=data.get("priority", 0),
            metadata=data.get("metadata", {})
        )


@dataclass
class TaskContext:
    """任务上下文"""
    task_id: str
    original_query: str
    status: TaskStatus = TaskStatus.PENDING
    assigned_agents: List[str] = field(default_factory=list)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    final_result: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    execution_time: float = 0.0
    retry_count: int = 0
    error_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskContext':
        data = data.copy()
        data["status"] = TaskStatus(data["status"])
        return cls(**data)


@dataclass
class MemoryEntry:
    """记忆条目"""
    entry_id: str
    memory_type: str
    content: str
    embedding: Optional[List[float]] = None
    importance: float = 0.5
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryEntry:
    """记忆条目"""
    entry_id: str
    memory_type: str
    content: str
    embedding: Optional[List[float]] = None
    importance: float = 0.5
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    decay_factor: float = 0.95
    emotional_valence: float = 0.0
    context_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def calculate_decay(self, current_time: float) -> float:
        """计算记忆衰减"""
        time_diff = current_time - self.created_at
        hours_elapsed = time_diff / 3600
        return self.decay_factor ** hours_elapsed

    def update_importance(self, access_count_increment: int = 1) -> None:
        """更新重要性分数"""
        self.access_count += access_count_increment
        recency_boost = min(0.2, self.access_count * 0.02)
        self.importance = min(1.0, self.importance * 0.9 + recency_boost)


class WorkingMemory:
    """工作记忆 - 会话期间的临时记忆

    类似于人类的"工作台"，用于存储当前任务相关的临时信息。
    采用固定大小的循环缓冲区实现。
    """

    def __init__(self, max_items: int = 20):
        self.max_items = max_items
        self._buffer: deque = deque(maxlen=max_items)
        self._focus_area: Optional[Dict[str, Any]] = None
        self._attention_mask: Set[str] = set()
        self._lock = threading.RLock()

    def add(self, item: Dict[str, Any], category: str = "general") -> None:
        """添加工作记忆项"""
        with self._lock:
            memory_item = {
                "category": category,
                "data": item,
                "timestamp": time.time()
            }
            self._buffer.append(memory_item)
            self._attention_mask.add(category)

    def get_recent(self, category: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的工作记忆"""
        with self._lock:
            items = list(self._buffer)
            if category:
                items = [i for i in items if i["category"] == category]
            return items[-limit:]

    def clear(self, category: Optional[str] = None) -> None:
        """清除工作记忆"""
        with self._lock:
            if category:
                self._buffer = deque(
                    [i for i in self._buffer if i["category"] != category],
                    maxlen=self.max_items
                )
            else:
                self._buffer.clear()
            self._attention_mask.clear()

    def set_focus(self, focus_data: Dict[str, Any]) -> None:
        """设置当前焦点"""
        with self._lock:
            self._focus_area = focus_data

    def get_focus(self) -> Optional[Dict[str, Any]]:
        """获取当前焦点"""
        return self._focus_area

    def get_summary(self) -> Dict[str, Any]:
        """获取工作记忆摘要"""
        with self._lock:
            return {
                "buffer_size": len(self._buffer),
                "attention_categories": list(self._attention_mask),
                "has_focus": self._focus_area is not None
            }


class EpisodicMemory:
    """情景记忆 - 存储特定经历和对话

    记录具体的事件、对话和经历，支持时间顺序检索。
    """

    def __init__(self, max_episodes: int = 500):
        self.max_episodes = max_episodes
        self._episodes: List[Dict[str, Any]] = []
        self._episode_index: Dict[str, List[int]] = defaultdict(list)
        self._lock = threading.RLock()

    def add_episode(self, episode_type: str, content: Dict[str, Any],
                   participants: List[str], outcome: str = "neutral",
                   emotional_impact: float = 0.0) -> str:
        """添加情景记忆"""
        with self._lock:
            episode_id = str(uuid.uuid4())
            episode = {
                "episode_id": episode_id,
                "episode_type": episode_type,
                "content": content,
                "participants": participants,
                "outcome": outcome,
                "emotional_impact": emotional_impact,
                "timestamp": time.time(),
                "sequences": []
            }
            self._episodes.append(episode)
            self._episode_index[episode_type].append(len(self._episodes) - 1)

            if len(self._episodes) > self.max_episodes:
                old_episode = self._episodes.pop(0)
                self._rebuild_index()

            return episode_id

    def _rebuild_index(self) -> None:
        """重建索引"""
        self._episode_index.clear()
        for idx, episode in enumerate(self._episodes):
            self._episode_index[episode["episode_type"]].append(idx)

    def get_recent_episodes(self, episode_type: Optional[str] = None,
                           limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的情景记忆"""
        with self._lock:
            episodes = self._episodes
            if episode_type:
                indices = self._episode_index.get(episode_type, [])
                episodes = [self._episodes[i] for i in indices[-limit:]]
            return episodes[-limit:]

    def search_by_outcome(self, outcome: str) -> List[Dict[str, Any]]:
        """按结果搜索情景记忆"""
        with self._lock:
            return [e for e in self._episodes if e["outcome"] == outcome]

    def get_stats(self) -> Dict[str, Any]:
        """获取情景记忆统计"""
        with self._lock:
            return {
                "total_episodes": len(self._episodes),
                "by_type": {
                    ep_type: len(indices) 
                    for ep_type, indices in self._episode_index.items()
                }
            }


class SemanticMemory:
    """语义记忆 - 存储抽象知识和概念

    使用向量嵌入进行语义相似度搜索。
    """

    def __init__(self, max_concepts: int = 1000, similarity_threshold: float = 0.7):
        self.max_concepts = max_concepts
        self.similarity_threshold = similarity_threshold
        self._concepts: List[Dict[str, Any]] = []
        self._category_index: Dict[str, Set[int]] = defaultdict(set)
        self._lock = threading.RLock()

    def add_concept(self, concept: str, definition: str,
                   category: str, embedding: Optional[List[float]] = None,
                   related_concepts: Optional[List[str]] = None) -> str:
        """添加概念"""
        with self._lock:
            concept_id = str(uuid.uuid4())
            concept_entry = {
                "concept_id": concept_id,
                "concept": concept,
                "definition": definition,
                "category": category,
                "embedding": embedding,
                "related_concepts": related_concepts or [],
                "usage_count": 0,
                "created_at": time.time()
            }
            self._concepts.append(concept_entry)
            self._category_index[category].add(len(self._concepts) - 1)

            if len(self._concepts) > self.max_concepts:
                self._concepts.pop(0)
                self._rebuild_category_index()

            return concept_id

    def _rebuild_category_index(self) -> None:
        """重建类别索引"""
        self._category_index.clear()
        for idx, concept in enumerate(self._concepts):
            self._category_index[concept["category"]].add(idx)

    def find_related(self, query: str, embedding: Optional[List[float]] = None,
                    category: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """查找相关概念"""
        with self._lock:
            candidates = []

            for idx, concept in enumerate(self._concepts):
                if category and concept["category"] != category:
                    continue

                if embedding and concept["embedding"]:
                    similarity = self._cosine_similarity(embedding, concept["embedding"])
                    if similarity >= self.similarity_threshold:
                        concept_copy = concept.copy()
                        concept_copy["similarity"] = similarity
                        concept_copy["concept_id"] = concept["concept_id"]
                        candidates.append(concept_copy)
                elif query.lower() in concept["definition"].lower() or query.lower() in concept["concept"].lower():
                    concept_copy = concept.copy()
                    concept_copy["similarity"] = 0.5
                    candidates.append(concept_copy)

            candidates.sort(key=lambda x: -x["similarity"])
            return candidates[:limit]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        """获取指定类别的所有概念"""
        with self._lock:
            return [
                self._concepts[idx].copy() 
                for idx in self._category_index.get(category, set())
            ]

    def increment_usage(self, concept_id: str) -> bool:
        """增加概念使用计数"""
        with self._lock:
            for concept in self._concepts:
                if concept["concept_id"] == concept_id:
                    concept["usage_count"] += 1
                    return True
            return False


class MetaMemory:
    """元记忆 - 关于记忆的记忆

    跟踪记忆的使用模式和质量评估。
    """

    def __init__(self):
        self._memory_about_memories: Dict[str, Dict[str, Any]] = {}
        self._retrieval_patterns: List[Dict[str, Any]] = []
        self._confidence_scores: Dict[str, float] = {}
        self._lock = threading.RLock()

    def record_memory_metadata(self, memory_id: str, memory_type: str,
                              accuracy: float, usefulness: float,
                              retrieval_count: int) -> None:
        """记录记忆的元数据"""
        with self._lock:
            self._memory_about_memories[memory_id] = {
                "memory_type": memory_type,
                "accuracy": accuracy,
                "usefulness": usefulness,
                "retrieval_count": retrieval_count,
                "last_evaluated": time.time()
            }

    def record_retrieval_pattern(self, memory_id: str, query_type: str,
                                retrieval_success: bool, response_time: float) -> None:
        """记录检索模式"""
        with self._lock:
            self._retrieval_patterns.append({
                "memory_id": memory_id,
                "query_type": query_type,
                "retrieval_success": retrieval_success,
                "response_time": response_time,
                "timestamp": time.time()
            })

    def update_confidence(self, memory_id: str, confidence: float) -> None:
        """更新置信度"""
        with self._lock:
            self._confidence_scores[memory_id] = confidence

    def get_memory_quality(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """获取记忆质量评估"""
        with self._lock:
            metadata = self._memory_about_memories.get(memory_id)
            if not metadata:
                return None

            return {
                **metadata,
                "confidence": self._confidence_scores.get(memory_id, 0.5)
            }

    def get_poorly_recalled_memories(self, threshold: float = 0.3) -> List[str]:
        """获取回忆效果差的记忆"""
        with self._lock:
            return [
                mid for mid, conf in self._confidence_scores.items()
                if conf < threshold
            ]


class MemoryManager:
    """分层记忆管理器 - 统一管理所有记忆层

    提供统一的接口访问短期记忆、长期记忆、工作记忆、
    情景记忆、语义记忆和元记忆。
    """

    def __init__(self, agent_id: str = "default",
                max_short_term: int = 100,
                max_long_term: int = 1000,
                max_working: int = 20,
                max_episodes: int = 500,
                max_semantic: int = 1000,
                consolidation_interval: float = 300.0,
                forgetting_threshold: float = 0.1):
        self.agent_id = agent_id
        self.max_short_term = max_short_term
        self.max_long_term = max_long_term
        self.consolidation_interval = consolidation_interval
        self.forgetting_threshold = forgetting_threshold

        self.short_term: deque = deque(maxlen=max_short_term)
        self.long_term: List[MemoryEntry] = []
        self.working_memory = WorkingMemory(max_working)
        self.episodic_memory = EpisodicMemory(max_episodes)
        self.semantic_memory = SemanticMemory(max_semantic)
        self.meta_memory = MetaMemory()
        self.personality: Dict[str, Any] = {}

        self._last_consolidation = time.time()
        self._lock = threading.RLock()

    def add_short_term(self, content: str, metadata: Optional[Dict[str, Any]] = None,
                      importance: float = 0.3, tags: Optional[List[str]] = None) -> str:
        """添加短期记忆"""
        with self._lock:
            entry = MemoryEntry(
                entry_id=str(uuid.uuid4()),
                memory_type="short_term",
                content=content,
                importance=importance,
                metadata=metadata or {},
                context_tags=tags or []
            )
            self.short_term.append(entry)

            if len(self.short_term) >= self.max_short_term * 0.8:
                self._consolidate_memories()

            return entry.entry_id

    def add_long_term(self, content: str, importance: float = 0.5,
                      embedding: Optional[List[float]] = None,
                      metadata: Optional[Dict[str, Any]] = None,
                      emotional_valence: float = 0.0) -> str:
        """添加长期记忆"""
        with self._lock:
            entry = MemoryEntry(
                entry_id=str(uuid.uuid4()),
                memory_type="long_term",
                content=content,
                importance=importance,
                embedding=embedding,
                metadata=metadata or {},
                emotional_valence=emotional_valence
            )
            self.long_term.append(entry)

            if len(self.long_term) > self.max_long_term:
                self._prune_long_term()

            return entry.entry_id

    def add_working_memory(self, item: Dict[str, Any], category: str = "general") -> None:
        """添加工作记忆"""
        self.working_memory.add(item, category)

    def add_episodic_memory(self, episode_type: str, content: Dict[str, Any],
                           participants: List[str], outcome: str = "neutral",
                           emotional_impact: float = 0.0) -> str:
        """添加情景记忆"""
        return self.episodic_memory.add_episode(
            episode_type, content, participants, outcome, emotional_impact
        )

    def add_semantic_memory(self, concept: str, definition: str,
                           category: str, embedding: Optional[List[float]] = None,
                           related_concepts: Optional[List[str]] = None) -> str:
        """添加语义记忆"""
        return self.semantic_memory.add_concept(
            concept, definition, category, embedding, related_concepts
        )

    def add_personality(self, key: str, value: Any) -> None:
        """添加人格记忆（用户偏好）"""
        with self._lock:
            self.personality[key] = {
                "value": value,
                "updated_at": time.time()
            }

    def get_short_term(self, limit: int = 10) -> List[MemoryEntry]:
        """获取短期记忆"""
        with self._lock:
            return list(self.short_term)[-limit:]

    def search_long_term(self, query: str, limit: int = 5,
                        embedding: Optional[List[float]] = None) -> List[MemoryEntry]:
        """搜索长期记忆"""
        with self._lock:
            keywords = query.lower().split()
            scored = []

            for entry in self.long_term:
                if embedding and entry.embedding:
                    score = self._semantic_similarity(embedding, entry.embedding)
                else:
                    score = sum(1 for kw in keywords if kw in entry.content.lower())

                if score > 0:
                    entry.last_accessed = time.time()
                    entry.update_importance()
                    scored.append((score, entry))

            scored.sort(key=lambda x: (-x[0], -x[1].importance, -x[1].access_count))
            return [entry for _, entry in scored[:limit]]

    def _semantic_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算语义相似度"""
        if not vec1 or not vec2:
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def search_semantic(self, query: str,
                       category: Optional[str] = None,
                       embedding: Optional[List[float]] = None,
                       limit: int = 5) -> List[Dict[str, Any]]:
        """搜索语义记忆"""
        return self.semantic_memory.find_related(
            query, embedding, category, limit
        )

    def get_working_memory(self, category: Optional[str] = None,
                          limit: int = 10) -> List[Dict[str, Any]]:
        """获取工作记忆"""
        return self.working_memory.get_recent(category, limit)

    def get_episodic_memory(self, episode_type: Optional[str] = None,
                           limit: int = 10) -> List[Dict[str, Any]]:
        """获取情景记忆"""
        return self.episodic_memory.get_recent_episodes(episode_type, limit)

    def get_personality(self, key: Optional[str] = None) -> Any:
        """获取人格记忆"""
        with self._lock:
            if key is None:
                return self.personality.copy()
            return self.personality.get(key)

    def recall_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """召回特定记忆"""
        with self._lock:
            for entry in self.long_term:
                if entry.entry_id == memory_id:
                    entry.update_importance()
                    self.meta_memory.record_retrieval_pattern(
                        memory_id, "specific_query", True, 0.1
                    )
                    return entry
            return None

    def _consolidate_memories(self) -> None:
        """整合记忆 - 将重要短期记忆转移到长期记忆"""
        current_time = time.time()

        if current_time - self._last_consolidation < self.consolidation_interval:
            return

        with self._lock:
            candidates = []

            for entry in self.short_term:
                decay = entry.calculate_decay(current_time)
                effective_importance = entry.importance * decay

                if effective_importance > 0.3:
                    entry_copy = copy.deepcopy(entry)
                    entry_copy.importance = effective_importance
                    candidates.append(entry_copy)

            if candidates:
                candidates.sort(key=lambda x: -x.importance)

                for candidate in candidates[:len(candidates) // 2]:
                    self.long_term.append(candidate)

            self._last_consolidation = current_time

            if len(self.long_term) > self.max_long_term:
                self._prune_long_term()

    def _prune_long_term(self) -> None:
        """清理低重要性和遗忘的记忆"""
        with self._lock:
            current_time = time.time()

            valid_memories = []
            for entry in self.long_term:
                decay = entry.calculate_decay(current_time)
                effective_importance = entry.importance * decay

                if effective_importance >= self.forgetting_threshold:
                    valid_memories.append(entry)

            valid_memories.sort(key=lambda x: (
                -x.importance * x.calculate_decay(current_time),
                -x.access_count
            ))

            self.long_term = valid_memories[:self.max_long_term]

    def forget_unused_memories(self, days_unused: float = 30.0,
                             min_importance: float = 0.2) -> int:
        """遗忘长期未使用的记忆"""
        with self._lock:
            cutoff_time = time.time() - (days_unused * 86400)

            original_count = len(self.long_term)
            self.long_term = [
                e for e in self.long_term
                if e.last_accessed > cutoff_time or e.importance >= min_importance
            ]

            return original_count - len(self.long_term)

    def get_memory_summary(self) -> Dict[str, Any]:
        """获取完整记忆摘要"""
        with self._lock:
            return {
                "short_term": {
                    "count": len(self.short_term),
                    "max": self.max_short_term
                },
                "long_term": {
                    "count": len(self.long_term),
                    "max": self.max_long_term
                },
                "working_memory": self.working_memory.get_summary(),
                "episodic": self.episodic_memory.get_stats(),
                "semantic": {
                    "concept_count": len(self.semantic_memory._concepts),
                    "categories": list(self.semantic_memory._category_index.keys())
                },
                "personality": {
                    "keys": list(self.personality.keys()),
                    "count": len(self.personality)
                },
                "consolidation": {
                    "last_consolidation": self._last_consolidation,
                    "interval": self.consolidation_interval
                }
            }

    def export_memory(self, memory_type: Optional[str] = None) -> Dict[str, Any]:
        """导出记忆数据"""
        with self._lock:
            if memory_type == "long_term":
                return {"long_term": [e.to_dict() for e in self.long_term]}
            elif memory_type == "short_term":
                return {"short_term": [e.to_dict() for e in self.short_term]}
            elif memory_type == "personality":
                return {"personality": copy.deepcopy(self.personality)}
            else:
                return self.get_memory_summary()

    def clear(self, memory_type: Optional[str] = None) -> None:
        """清除记忆"""
        with self._lock:
            if memory_type == "short_term":
                self.short_term.clear()
            elif memory_type == "long_term":
                self.long_term.clear()
            elif memory_type == "working":
                self.working_memory.clear()
            elif memory_type == "episodic":
                self.episodic_memory = EpisodicMemory()
            elif memory_type == "semantic":
                self.semantic_memory = SemanticMemory()
            elif memory_type == "personality":
                self.personality.clear()
            else:
                self.short_term.clear()
                self.long_term.clear()
                self.working_memory.clear()
                self.episodic_memory = EpisodicMemory()
                self.semantic_memory = SemanticMemory()
                self.personality.clear()


class Span:
    """分布式追踪 Span

    代表单个操作或工作单元，用于追踪请求在系统中的流动。
    """

    def __init__(self, name: str, trace_id: Optional[str] = None,
                parent_span_id: Optional[str] = None,
                span_type: str = "internal"):
        self.span_id = str(uuid.uuid4())[:16]
        self.trace_id = trace_id or str(uuid.uuid4())[:16]
        self.parent_span_id = parent_span_id
        self.name = name
        self.span_type = span_type
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None
        self.status = "ok"
        self.error_message: Optional[str] = None
        self.attributes: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

    def set_attribute(self, key: str, value: Any) -> None:
        """设置Span属性"""
        with self._lock:
            self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """添加Span事件"""
        with self._lock:
            self.events.append({
                "name": name,
                "timestamp": time.time(),
                "attributes": attributes or {}
            })

    def set_status(self, status: str, message: Optional[str] = None) -> None:
        """设置Span状态"""
        with self._lock:
            self.status = status
            if message:
                self.error_message = message

    def finish(self) -> None:
        """完成Span"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "span_type": self.span_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "status": self.status,
            "error_message": self.error_message,
            "attributes": self.attributes,
            "events": self.events
        }


class TraceManager:
    """分布式追踪管理器

    收集和管理分布式追踪数据，支持采样和追踪导出。
    """

    def __init__(self, max_spans: int = 10000, sampling_rate: float = 1.0):
        self.max_spans = max_spans
        self.sampling_rate = sampling_rate
        self._spans: List[Span] = []
        self._active_spans: Dict[str, Span] = {}
        self._trace_context: Dict[str, str] = {}
        self._lock = threading.RLock()

    def create_span(self, name: str, span_type: str = "internal",
                   parent: Optional[Span] = None) -> Span:
        """创建新的Span"""
        parent_span_id = parent.span_id if parent else None
        trace_id = parent.trace_id if parent else None

        if trace_id is None and self.sampling_rate < 1.0:
            if random.random() > self.sampling_rate:
                return Span(name, span_type=span_type)

        span = Span(
            name=name,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            span_type=span_type
        )

        with self._lock:
            self._active_spans[span.span_id] = span
            self._trace_context["current_span"] = span.span_id

        return span

    def finish_span(self, span: Span) -> None:
        """完成Span并添加到追踪记录"""
        span.finish()

        with self._lock:
            if span.span_id in self._active_spans:
                del self._active_spans[span.span_id]

            self._spans.append(span)

            if len(self._spans) > self.max_spans:
                self._spans = self._spans[-self.max_spans:]

    def get_current_span(self) -> Optional[Span]:
        """获取当前活动的Span"""
        with self._lock:
            span_id = self._trace_context.get("current_span")
            if span_id:
                return self._active_spans.get(span_id)
            return None

    def get_trace(self, trace_id: str) -> List[Span]:
        """获取完整追踪"""
        with self._lock:
            return [s for s in self._spans if s.trace_id == trace_id]

    def get_recent_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的追踪记录"""
        with self._lock:
            traces = defaultdict(list)
            for span in self._spans:
                traces[span.trace_id].append(span.to_dict())

            trace_summaries = []
            for trace_id, spans in traces.items():
                total_duration = sum(
                    s.get("duration", 0) or 0 for s in spans
                )
                trace_summaries.append({
                    "trace_id": trace_id,
                    "span_count": len(spans),
                    "total_duration": total_duration,
                    "spans": spans[-5:]
                })

            trace_summaries.sort(key=lambda x: -x["total_duration"])
            return trace_summaries[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """获取追踪统计"""
        with self._lock:
            error_count = sum(1 for s in self._spans if s.status == "error")
            return {
                "total_spans": len(self._spans),
                "active_spans": len(self._active_spans),
                "error_count": error_count,
                "error_rate": error_count / max(1, len(self._spans))
            }


class MetricsCollector:
    """指标收集器

    收集和聚合系统指标，支持多种指标类型和导出格式。
    """

    def __init__(self, window_size: float = 60.0):
        self.window_size = window_size
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()

    def increment(self, name: str, value: float = 1.0,
                 tags: Optional[Dict[str, str]] = None) -> None:
        """增加计数器"""
        key = self._make_key(name, tags)
        with self._lock:
            self._counters[key] += value
            self._timestamps[key] = time.time()

    def gauge(self, name: str, value: float,
             tags: Optional[Dict[str, str]] = None) -> None:
        """设置Gauge值"""
        key = self._make_key(name, tags)
        with self._lock:
            self._gauges[key] = value
            self._timestamps[key] = time.time()

    def histogram(self, name: str, value: float,
                 tags: Optional[Dict[str, str]] = None) -> None:
        """记录直方图值"""
        key = self._make_key(name, tags)
        with self._lock:
            self._histograms[key].append(value)
            self._timestamps[key] = time.time()

    def timer(self, name: str, duration: float,
             tags: Optional[Dict[str, str]] = None) -> None:
        """记录计时器值"""
        key = self._make_key(name, tags)
        with self._lock:
            self._timers[key].append(duration)
            self._timestamps[key] = time.time()

    def _make_key(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """生成唯一键"""
        if tags:
            tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            return f"{name}[{tag_str}]"
        return name

    def get_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """获取计数器值"""
        key = self._make_key(name, tags)
        with self._lock:
            return self._counters.get(key, 0.0)

    def get_gauge(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[float]:
        """获取Gauge值"""
        key = self._make_key(name, tags)
        with self._lock:
            return self._gauges.get(key)

    def get_histogram_stats(self, name: str,
                           tags: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """获取直方图统计"""
        key = self._make_key(name, tags)
        with self._lock:
            values = self._histograms.get(key, [])
            if not values:
                return None

            values.sort()
            return {
                "count": len(values),
                "sum": sum(values),
                "min": values[0],
                "max": values[-1],
                "avg": sum(values) / len(values),
                "p50": values[len(values) // 2],
                "p95": values[int(len(values) * 0.95)],
                "p99": values[int(len(values) * 0.99)]
            }

    def get_timer_stats(self, name: str,
                       tags: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """获取计时器统计"""
        key = self._make_key(name, tags)
        with self._lock:
            values = self._timers.get(key, [])
            if not values:
                return None

            values.sort()
            return {
                "count": len(values),
                "sum": sum(values),
                "min": values[0],
                "max": values[-1],
                "avg": sum(values) / len(values),
                "p50": values[len(values) // 2],
                "p95": values[int(len(values) * 0.95)]
            }

    def get_all_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        with self._lock:
            metrics = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {},
                "timers": {}
            }

            for name in self._histograms:
                stats = self.get_histogram_stats(name)
                if stats:
                    metrics["histograms"][name] = stats

            for name in self._timers:
                stats = self.get_timer_stats(name)
                if stats:
                    metrics["timers"][name] = stats

            return metrics

    def reset(self, older_than: Optional[float] = None) -> None:
        """重置指标"""
        with self._lock:
            if older_than:
                cutoff = time.time() - older_than
                self._counters = {
                    k: v for k, v in self._counters.items()
                    if self._timestamps.get(k, 0) >= cutoff
                }
                self._gauges = {
                    k: v for k, v in self._gauges.items()
                    if self._timestamps.get(k, 0) >= cutoff
                }
                self._histograms = defaultdict(
                    list, {
                        k: v for k, v in self._histograms.items()
                        if self._timestamps.get(k, 0) >= cutoff
                    }
                )
                self._timers = defaultdict(
                    list, {
                        k: v for k, v in self._timers.items()
                        if self._timestamps.get(k, 0) >= cutoff
                    }
                )
            else:
                self._counters.clear()
                self._gauges.clear()
                self._histograms.clear()
                self._timers.clear()
                self._timestamps.clear()


class HealthMonitor:
    """健康监控器

    监控系统和Agent的健康状态，提供健康检查和告警功能。
    """

    def __init__(self, check_interval: float = 10.0):
        self.check_interval = check_interval
        self._components: Dict[str, Dict[str, Any]] = {}
        self._health_history: List[Dict[str, Any]] = []
        self._alerts: List[Dict[str, Any]] = []
        self._thresholds: Dict[str, Dict[str, float]] = {}
        self._lock = threading.RLock()

    def register_component(self, name: str, component_type: str,
                          check_func: Callable[[], Dict[str, Any]],
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """注册监控组件"""
        with self._lock:
            self._components[name] = {
                "type": component_type,
                "check_func": check_func,
                "metadata": metadata or {},
                "status": "unknown",
                "last_check": 0,
                "last_result": None,
                "consecutive_failures": 0
            }

    def set_threshold(self, component: str, metric: str,
                     warning: float, critical: float) -> None:
        """设置告警阈值"""
        with self._lock:
            if component not in self._thresholds:
                self._thresholds[component] = {}
            self._thresholds[component][metric] = {
                "warning": warning,
                "critical": critical
            }

    def check_component(self, name: str) -> Dict[str, Any]:
        """检查单个组件"""
        with self._lock:
            component = self._components.get(name)
            if not component:
                return {"status": "not_found"}

            try:
                result = component["check_func"]()
                current_time = time.time()

                if result.get("healthy", True):
                    status = "healthy"
                    component["consecutive_failures"] = 0
                else:
                    status = "unhealthy"
                    component["consecutive_failures"] += 1

                component["status"] = status
                component["last_check"] = current_time
                component["last_result"] = result

                self._check_thresholds(name, result)

                return {"status": status, "result": result}

            except Exception as e:
                component["status"] = "error"
                component["consecutive_failures"] += 1
                component["last_result"] = {"error": str(e)}
                return {"status": "error", "error": str(e)}

    def _check_thresholds(self, name: str, result: Dict[str, Any]) -> None:
        """检查阈值并生成告警"""
        thresholds = self._thresholds.get(name, {})
        for metric, threshold in thresholds.items():
            if metric in result:
                value = result[metric]
                if value >= threshold["critical"]:
                    self._create_alert(
                        name, "critical",
                        f"{metric}达到关键阈值: {value} >= {threshold['critical']}"
                    )
                elif value >= threshold["warning"]:
                    self._create_alert(
                        name, "warning",
                        f"{metric}达到警告阈值: {value} >= {threshold['warning']}"
                    )

    def _create_alert(self, component: str, severity: str, message: str) -> None:
        """创建告警"""
        alert = {
            "component": component,
            "severity": severity,
            "message": message,
            "timestamp": time.time()
        }
        self._alerts.append(alert)

        if len(self._alerts) > 100:
            self._alerts = self._alerts[-100:]

    def check_all(self) -> Dict[str, Any]:
        """检查所有组件"""
        results = {}
        for name in self._components:
            results[name] = self.check_component(name)

        overall_status = "healthy"
        if any(r["status"] == "error" for r in results.values()):
            overall_status = "error"
        elif any(r["status"] == "unhealthy" for r in results.values()):
            overall_status = "unhealthy"

        health_record = {
            "timestamp": time.time(),
            "overall_status": overall_status,
            "components": results
        }
        self._health_history.append(health_record)

        if len(self._health_history) > 1000:
            self._health_history = self._health_history[-1000:]

        return health_record

    def get_component_status(self, name: str) -> Optional[Dict[str, Any]]:
        """获取组件状态"""
        with self._lock:
            component = self._components.get(name)
            if not component:
                return None

            return {
                "name": name,
                "type": component["type"],
                "status": component["status"],
                "last_check": component["last_check"],
                "consecutive_failures": component["consecutive_failures"],
                "metadata": component["metadata"]
            }

    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康摘要"""
        with self._lock:
            check_result = self.check_all()
            return {
                "overall_status": check_result["overall_status"],
                "component_count": len(self._components),
                "healthy_count": sum(
                    1 for c in check_result["components"].values()
                    if c["status"] == "healthy"
                ),
                "unhealthy_count": sum(
                    1 for c in check_result["components"].values()
                    if c["status"] == "unhealthy"
                ),
                "error_count": sum(
                    1 for c in check_result["components"].values()
                    if c["status"] == "error"
                ),
                "recent_alerts": self._alerts[-10:]
            }

    def get_recent_alerts(self, severity: Optional[str] = None,
                         limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近告警"""
        with self._lock:
            alerts = self._alerts
            if severity:
                alerts = [a for a in alerts if a["severity"] == severity]
            return alerts[-limit:]

    def clear_alerts(self, older_than: Optional[float] = None) -> int:
        """清除告警"""
        with self._lock:
            if older_than:
                cutoff = time.time() - older_than
                original_count = len(self._alerts)
                self._alerts = [a for a in self._alerts if a["timestamp"] >= cutoff]
                return original_count - len(self._alerts)
            else:
                count = len(self._alerts)
                self._alerts.clear()
                return count


class StructuredLogger:
    """结构化日志记录器

    提供JSON格式的结构化日志输出，便于日志分析。
    """

    def __init__(self, name: str, level: int = logging.INFO,
                include_context: bool = True):
        self.name = name
        self.level = level
        self.include_context = include_context
        self._context: Dict[str, Any] = {}
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._handler = logging.StreamHandler()
        self._handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        if not self._logger.handlers:
            self._logger.addHandler(self._handler)

    def set_context(self, key: str, value: Any) -> None:
        """设置日志上下文"""
        self._context[key] = value

    def clear_context(self) -> None:
        """清除日志上下文"""
        self._context.clear()

    def _log(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """内部日志方法"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "logger": self.name,
            "level": logging.getLevelName(level),
            "message": message
        }

        if self.include_context:
            log_data["context"] = self._context.copy()

        if extra:
            log_data.update(extra)

        self._logger.log(level, message, extra={"data": log_data})

    def debug(self, message: str, **extra) -> None:
        """调试日志"""
        self._log(logging.DEBUG, message, extra)

    def info(self, message: str, **extra) -> None:
        """信息日志"""
        self._log(logging.INFO, message, extra)

    def warning(self, message: str, **extra) -> None:
        """警告日志"""
        self._log(logging.WARNING, message, extra)

    def error(self, message: str, error: Optional[Exception] = None, **extra) -> None:
        """错误日志"""
        error_info = {"error_type": type(error).__name__ if error else None}
        if error:
            error_info["error_message"] = str(error)
            error_info["traceback"] = traceback.format_exc() if hasattr(error, '__traceback__') else None

        log_extra = {**(extra or {}), "error": error_info}
        self._log(logging.ERROR, message, log_extra)

    def critical(self, message: str, **extra) -> None:
        """严重日志"""
        self._log(logging.CRITICAL, message, extra)

    def log_performance(self, operation: str, duration: float,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """记录性能日志"""
        self._log(logging.INFO, f"性能: {operation} 完成于 {duration:.3f}s", {
            "performance": {
                "operation": operation,
                "duration": duration
            },
            **(metadata or {})
        })


class ObservabilitySystem:
    """可观测性系统 - 统一管理所有监控组件

    整合追踪、指标、日志和健康监控。
    """

    def __init__(self, service_name: str = "multi_agent_system"):
        self.service_name = service_name
        self.trace_manager = TraceManager()
        self.metrics_collector = MetricsCollector()
        self.health_monitor = HealthMonitor()
        self.logger = StructuredLogger(service_name)

        self._request_id: Optional[str] = None
        self._parent_span: Optional[Span] = None

    def start_request(self, request_id: Optional[str] = None,
                     operation_name: str = "request") -> Span:
        """开始请求追踪"""
        self._request_id = request_id or str(uuid.uuid4())[:16]
        span = self.trace_manager.create_span(
            operation_name,
            span_type="request",
            parent=self._parent_span
        )
        self._parent_span = span

        self.logger.set_context("request_id", self._request_id)
        self.logger.set_context("trace_id", span.trace_id)

        self.metrics_collector.increment("requests_total")

        return span

    def end_request(self, span: Span, status: str = "success",
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """结束请求追踪"""
        span.set_status(status)
        span.finish()

        self.trace_manager.finish_span(span)

        self.metrics_collector.histogram("request_duration", span.duration or 0)

        if status != "success":
            self.metrics_collector.increment("requests_errors")

        self.logger.clear_context()

        if self._parent_span:
            self._parent_span = None

    def record_agent_action(self, agent_id: str, action: str,
                           duration: Optional[float] = None,
                           success: bool = True) -> None:
        """记录Agent动作"""
        self.metrics_collector.increment(
            f"agent_{agent_id}_actions",
            tags={"action": action, "success": str(success)}
        )

        if duration:
            self.metrics_collector.timer(
                f"agent_{agent_id}_action_duration",
                duration,
                tags={"action": action}
            )

        self.logger.info(
            f"Agent {agent_id} 执行动作 {action}",
            agent_id=agent_id,
            action=action,
            success=success,
            duration=duration
        )

    def record_message(self, msg_type: str, direction: str,
                      size: Optional[int] = None) -> None:
        """记录消息"""
        self.metrics_collector.increment(
            "messages_total",
            tags={"type": msg_type, "direction": direction}
        )

        if size:
            self.metrics_collector.histogram("message_size", size)

    def get_dashboard(self) -> Dict[str, Any]:
        """获取监控仪表板数据"""
        return {
            "service": self.service_name,
            "timestamp": time.time(),
            "tracing": self.trace_manager.get_stats(),
            "metrics": self.metrics_collector.get_all_metrics(),
            "health": self.health_monitor.get_health_summary(),
            "recent_traces": self.trace_manager.get_recent_traces(5)
        }
    """Agent注册中心"""

    def __init__(self):
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._heartbeat: Dict[str, float] = {}
        self._lock = threading.RLock()

    def register(self, agent_id: str, agent_type: str, capabilities: List[str],
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """注册Agent"""
        with self._lock:
            self._agents[agent_id] = {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "capabilities": capabilities,
                "status": AgentState.HEALTHY,
                "metadata": metadata or {},
                "registered_at": time.time()
            }
            self._heartbeat[agent_id] = time.time()
            logger.info(f"Agent注册成功: {agent_id} ({agent_type})")

    def unregister(self, agent_id: str) -> bool:
        """注销Agent"""
        with self._lock:
            if agent_id in self._agents:
                del self._agents[agent_id]
                self._heartbeat.pop(agent_id, None)
                logger.info(f"Agent注销成功: {agent_id}")
                return True
            return False

    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """获取Agent信息"""
        with self._lock:
            return self._agents.get(agent_id)

    def find_agents(self, capabilities: Optional[List[str]] = None,
                    agent_type: Optional[str] = None,
                    status: Optional[AgentState] = None) -> List[str]:
        """查找符合条件的所有Agent"""
        with self._lock:
            results = []

            for agent_id, info in self._agents.items():
                if status and info["status"] != status:
                    continue
                if agent_type and info["agent_type"] != agent_type:
                    continue
                if capabilities:
                    if not all(cap in info["capabilities"] for cap in capabilities):
                        continue
                results.append(agent_id)

            return results

    def update_heartbeat(self, agent_id: str) -> None:
        """更新心跳"""
        with self._lock:
            if agent_id in self._heartbeat:
                self._heartbeat[agent_id] = time.time()

    def check_health(self, agent_id: str, timeout: float = 30.0) -> bool:
        """检查Agent健康状态"""
        with self._lock:
            last_beat = self._heartbeat.get(agent_id)
            if last_beat is None:
                return False
            is_healthy = (time.time() - last_beat) < timeout
            if agent_id in self._agents:
                self._agents[agent_id]["status"] = AgentState.HEALTHY if is_healthy else AgentState.UNHEALTHY
            return is_healthy

    def get_all_agents(self) -> Dict[str, Dict[str, Any]]:
        """获取所有Agent信息"""
        with self._lock:
            return copy.deepcopy(self._agents)


class CircuitBreaker:
    """熔断器"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = defaultdict(int)
        self.last_failure: Dict[str, float] = {}
        self.state: Dict[str, str] = defaultdict(lambda: "closed")
        self._lock = threading.RLock()

    def call(self, agent_id: str, func: Callable, *args, **kwargs) -> Any:
        """执行函数调用"""
        with self._lock:
            if self.state[agent_id] == "open":
                if time.time() - self.last_failure.get(agent_id, 0) > self.recovery_timeout:
                    self.state[agent_id] = "half-open"
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker is open for {agent_id}")

        try:
            result = func(*args, **kwargs)
            with self._lock:
                self.failures[agent_id] = 0
                self.state[agent_id] = "closed"
            return result
        except Exception as e:
            with self._lock:
                self.failures[agent_id] += 1
                self.last_failure[agent_id] = time.time()
                if self.failures[agent_id] >= self.failure_threshold:
                    self.state[agent_id] = "open"
            raise


class CircuitBreakerOpenError(Exception):
    """熔断器开启错误"""
    pass


class RetryPolicy:
    """重试策略"""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0,
                 max_delay: float = 60.0, exponential_base: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    def execute(self, func: Callable, *args, **kwargs) -> Tuple[Any, int]:
        """执行带重试的函数调用"""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs), attempt
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}, retrying in {delay}s")
                    time.sleep(delay)

        raise last_exception


class TracingManager:
    """分布式追踪管理器"""

    def __init__(self, service_name: str = "multi-agent-system"):
        self.service_name = service_name
        self.traces: Dict[str, List[Dict[str, Any]]] = {}
        self.current_span: Optional[str] = None
        self._lock = threading.RLock()

    def start_span(self, span_id: Optional[str] = None,
                   parent_id: Optional[str] = None,
                   operation_name: str = "default") -> str:
        """开始追踪跨度"""
        span_id = span_id or str(uuid.uuid4())
        timestamp = time.time()

        with self._lock:
            if span_id not in self.traces:
                self.traces[span_id] = []

            span = {
                "span_id": span_id,
                "parent_id": parent_id,
                "operation_name": operation_name,
                "start_time": timestamp,
                "end_time": None,
                "duration": None,
                "status": "started",
                "events": [],
                "attributes": {}
            }
            self.traces[span_id].append(span)
            self.current_span = span_id

        return span_id

    def end_span(self, span_id: Optional[str] = None,
                 status: str = "ok", attributes: Optional[Dict[str, Any]] = None) -> None:
        """结束追踪跨度"""
        span_id = span_id or self.current_span
        if span_id is None:
            return

        end_time = time.time()

        with self._lock:
            if span_id in self.traces:
                spans = self.traces[span_id]
                if spans:
                    current_span = spans[-1]
                    current_span["end_time"] = end_time
                    current_span["duration"] = end_time - current_span["start_time"]
                    current_span["status"] = status
                    if attributes:
                        current_span["attributes"].update(attributes)

        self.current_span = None

    def add_event(self, span_id: Optional[str] = None, event_name: str = "event",
                  attributes: Optional[Dict[str, Any]] = None) -> None:
        """添加追踪事件"""
        span_id = span_id or self.current_span
        if span_id is None:
            return

        with self._lock:
            if span_id in self.traces:
                spans = self.traces[span_id]
                if spans:
                    spans[-1]["events"].append({
                        "name": event_name,
                        "timestamp": time.time(),
                        "attributes": attributes or {}
                    })

    def get_trace(self, trace_id: str) -> Optional[List[Dict[str, Any]]]:
        """获取追踪记录"""
        with self._lock:
            return self.traces.get(trace_id)

    def get_all_traces(self) -> Dict[str, List[Dict[str, Any]]]:
        """获取所有追踪记录"""
        with self._lock:
            return copy.deepcopy(self.traces)


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.counters: Dict[str, int] = defaultdict(int)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.gauges: Dict[str, float] = {}
        self._lock = threading.RLock()

    def increment(self, metric: str, value: int = 1) -> None:
        """增加计数器"""
        with self._lock:
            self.counters[metric] += value

    def timing(self, metric: str, value: float) -> None:
        """记录时间"""
        with self._lock:
            self.timers[metric].append(value)

    def gauge(self, metric: str, value: float) -> None:
        """设置仪表值"""
        with self._lock:
            self.gauges[metric] = value

    def get_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        with self._lock:
            return {
                "counters": dict(self.counters),
                "timers": {k: {"count": len(v), "avg": sum(v)/len(v) if v else 0, "max": max(v) if v else 0} for k, v in self.timers.items()},
                "gauges": dict(self.gauges)
            }


class MessageBus:
    """增强型事件驱动消息总线
    
    支持优先级队列、死信队列、消息过滤、通配符订阅等高级特性。
    采用发布-订阅模式，支持异步消息通信。
    """
    
    def __init__(self, max_queue_size: int = 1000, enable_dlq: bool = True):
        self._queues: Dict[str, asyncio.Queue] = {}
        self._priority_queues: Dict[str, Dict[int, asyncio.Queue]] = {}
        self._subscribers: Dict[str, Set[str]] = defaultdict(set)
        self._wildcard_subscribers: Dict[str, Set[str]] = defaultdict(set)
        self._filters: Dict[str, Dict[str, Callable]] = {}
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        self.max_queue_size = max_queue_size
        self.enable_dlq = enable_dlq
        self._dlq: List[AgentMessage] = []
        self._message_history: deque = deque(maxlen=10000)
        self._total_messages = 0
        self._failed_messages = 0
        self._sequence_counters: Dict[str, int] = {}

    def _get_priority_queue(self, agent_id: str, priority: int) -> asyncio.Queue:
        """获取或创建优先级队列"""
        if agent_id not in self._priority_queues:
            self._priority_queues[agent_id] = {}
        if priority not in self._priority_queues[agent_id]:
            self._priority_queues[agent_id][priority] = asyncio.Queue(maxsize=self.max_queue_size)
        return self._priority_queues[agent_id][priority]

    def subscribe(self, topic: str, agent_id: str, 
                 filter_func: Optional[Callable] = None) -> None:
        """订阅主题
        
        Args:
            topic: 主题名称，支持通配符（如 "medical.*"）
            agent_id: 订阅者ID
            filter_func: 可选的消息过滤函数
        """
        with self._lock:
            if '*' in topic or '+' in topic:
                self._wildcard_subscribers[topic].add(agent_id)
            else:
                self._subscribers[topic].add(agent_id)
            
            if filter_func:
                if topic not in self._filters:
                    self._filters[topic] = {}
                self._filters[topic][agent_id] = filter_func
            
            if agent_id not in self._queues:
                self._queues[agent_id] = asyncio.Queue(maxsize=self.max_queue_size)
            
            logger.info(f"Agent {agent_id} 订阅主题: {topic}")

    def unsubscribe(self, topic: str, agent_id: str) -> bool:
        """取消订阅"""
        with self._lock:
            if topic in self._wildcard_subscribers:
                self._wildcard_subscribers[topic].discard(agent_id)
            self._subscribers[topic].discard(agent_id)
            
            if topic in self._filters and agent_id in self._filters[topic]:
                del self._filters[topic][agent_id]
            
            logger.info(f"Agent {agent_id} 取消订阅主题: {topic}")
            return True

    async def publish(self, topic: str, message: AgentMessage,
                      guarantee_delivery: bool = True) -> int:
        """发布消息到主题
        
        Args:
            topic: 目标主题
            message: 要发布的消息
            guarantee_delivery: 是否保证送达
            
        Returns:
            成功接收消息的订阅者数量
        """
        async with self._async_lock:
            subscribers = self._get_matching_subscribers(topic)
            delivered = 0
            
            self._total_messages += 1
            
            for agent_id in subscribers:
                try:
                    if self._should_deliver(agent_id, topic, message):
                        if message.priority <= MessagePriority.HIGH.value:
                            priority_queue = self._get_priority_queue(
                                agent_id, message.priority
                            )
                            await priority_queue.put(message)
                        else:
                            if agent_id not in self._queues:
                                self._queues[agent_id] = asyncio.Queue(
                                    maxsize=self.max_queue_size
                                )
                            await self._queues[agent_id].put(message)
                        
                        delivered += 1
                        
                        if agent_id not in self._sequence_counters:
                            self._sequence_counters[agent_id] = 0
                        self._sequence_counters[agent_id] += 1
                        
                except asyncio.QueueFull:
                    self._failed_messages += 1
                    logger.warning(f"队列满，无法投递到 agent {agent_id}")
                    if self.enable_dlq:
                        self._dlq.append(message)
            
            self._message_history.append({
                "topic": topic,
                "msg_id": message.msg_id,
                "delivered": delivered,
                "timestamp": time.time()
            })
            
            return delivered

    def _get_matching_subscribers(self, topic: str) -> Set[str]:
        """获取匹配主题的所有订阅者"""
        subscribers = set()
        
        subscribers.update(self._subscribers.get(topic, set()))
        
        for pattern, wildcard_subs in self._wildcard_subscribers.items():
            if self._match_topic(pattern, topic):
                subscribers.update(wildcard_subs)
        
        return subscribers

    def _match_topic(self, pattern: str, topic: str) -> bool:
        """匹配主题模式"""
        if pattern == topic:
            return True
        
        pattern_parts = pattern.split('.')
        topic_parts = topic.split('.')
        
        i = 0
        while i < len(pattern_parts):
            if pattern_parts[i] == '+':
                i += 1
                continue
            elif pattern_parts[i] == '*':
                if i == len(pattern_parts) - 1:
                    return True
                i += 1
            elif i < len(topic_parts) and pattern_parts[i] == topic_parts[i]:
                i += 1
            else:
                return False
        
        return i == len(pattern_parts) and i == len(topic_parts)

    def _should_deliver(self, agent_id: str, topic: str, 
                       message: AgentMessage) -> bool:
        """检查是否应该投递消息"""
        if topic in self._filters and agent_id in self._filters[topic]:
            try:
                return self._filters[topic][agent_id](message)
            except Exception:
                return True
        return True

    async def receive(self, agent_id: str, timeout: float = 0.1,
                     priority: Optional[int] = None) -> Optional[AgentMessage]:
        """接收消息
        
        Args:
            agent_id: 接收者ID
            timeout: 超时时间
            priority: 优先接收特定优先级的消息
        """
        if priority is not None and agent_id in self._priority_queues:
            priority_queue = self._priority_queues[agent_id].get(priority)
            if priority_queue:
                try:
                    return await asyncio.wait_for(
                        priority_queue.get(),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    pass
        
        if agent_id not in self._queues:
            return None

        try:
            return await asyncio.wait_for(
                self._queues[agent_id].get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None

    async def request(self, target_agent: str, message: AgentMessage,
                     timeout: float = 5.0) -> Optional[AgentMessage]:
        """发送请求并等待响应（请求-响应模式）"""
        response_queue = asyncio.Queue(maxsize=1)
        
        correlation_id = message.correlation_id or message.msg_id
        
        wrapped_content = {
            **message.content,
            "_response_queue": response_queue,
            "_correlation_id": correlation_id
        }
        
        wrapped_message = AgentMessage(
            msg_id=str(uuid.uuid4()),
            msg_type=MessageType.REQUEST,
            sender_id=message.sender_id,
            receiver_id=target_agent,
            content=wrapped_content,
            correlation_id=correlation_id,
            priority=message.priority
        )
        
        await self.publish(f"agent.{target_agent}", wrapped_message)
        
        try:
            return await asyncio.wait_for(
                response_queue.get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"请求超时: {message.msg_id}")
            return None

    async def broadcast(self, message: AgentMessage, 
                       exclude_sender: bool = True) -> int:
        """广播消息到所有订阅者"""
        all_topics = set(self._subscribers.keys()) | set(
            self._wildcard_subscribers.keys()
        )
        
        total_delivered = 0
        for topic in all_topics:
            if exclude_sender and message.sender_id in self._subscribers.get(topic, set()):
                continue
            delivered = await self.publish(topic, message, guarantee_delivery=False)
            total_delivered += delivered
        
        return total_delivered

    def get_dlq(self) -> List[AgentMessage]:
        """获取死信队列"""
        return list(self._dlq)

    def clear_dlq(self) -> int:
        """清空死信队列"""
        count = len(self._dlq)
        self._dlq.clear()
        return count

    def get_stats(self) -> Dict[str, Any]:
        """获取消息总线统计"""
        with self._lock:
            priority_queue_info = {}
            for agent_id, queues in self._priority_queues.items():
                priority_queue_info[agent_id] = {
                    p: q.qsize() for p, q in queues.items()
                }
            
            return {
                "total_topics": len(self._subscribers) + len(self._wildcard_subscribers),
                "total_subscribers": sum(
                    len(s) for s in self._subscribers.values()
                ) + sum(len(s) for s in self._wildcard_subscribers.values()),
                "queue_sizes": {
                    k: v.qsize() for k, v in self._queues.items()
                },
                "priority_queue_sizes": priority_queue_info,
                "subscriber_counts": {
                    k: len(v) for k, v in self._subscribers.items()
                },
                "total_messages": self._total_messages,
                "failed_messages": self._failed_messages,
                "dlq_size": len(self._dlq),
                "delivery_rate": (
                    (self._total_messages - self._failed_messages) / 
                    self._total_messages if self._total_messages > 0 else 1.0
                ),
                "wildcard_topics": list(self._wildcard_subscribers.keys())
            }

    async def health_check(self, agent_id: str) -> Dict[str, Any]:
        """检查Agent消息队列健康状态"""
        stats = self.get_stats()
        queue_size = stats.get("queue_sizes", {}).get(agent_id, 0)
        
        return {
            "agent_id": agent_id,
            "queue_size": queue_size,
            "is_healthy": queue_size < self.max_queue_size * 0.9,
            "capacity_remaining": self.max_queue_size - queue_size
        }


class WorkflowState(Enum):
    """工作流状态"""
    IDLE = "idle"
    ROUTING = "routing"
    EXECUTING = "executing"
    COORDINATING = "coordinating"
    COMPLETING = "completing"
    ERROR = "error"


@dataclass
class WorkflowContext:
    """工作流上下文"""
    workflow_id: str
    current_state: WorkflowState = WorkflowState.IDLE
    task_context: Optional[TaskContext] = None
    agent_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowEngine:
    """状态图工作流引擎"""

    def __init__(self):
        self.states: Dict[WorkflowState, Callable] = {}
        self.transitions: Dict[Tuple[WorkflowState, str], WorkflowState] = {}
        self._lock = threading.RLock()

    def register_state(self, state: WorkflowState, handler: Callable) -> None:
        """注册状态处理器"""
        with self._lock:
            self.states[state] = handler

    def register_transition(self, from_state: WorkflowState, event: str,
                            to_state: WorkflowState) -> None:
        """注册状态转换"""
        with self._lock:
            self.transitions[(from_state, event)] = to_state

    async def execute(self, context: WorkflowContext) -> WorkflowContext:
        """执行工作流"""
        while context.current_state in self.states:
            handler = self.states[context.current_state]
            event = await handler(context)
            context.history.append({
                "state": context.current_state.value,
                "timestamp": time.time(),
                "event": event
            })

            next_state = self.transitions.get((context.current_state, event))
            if next_state is None:
                break

            context.current_state = next_state

            if next_state == WorkflowState.COMPLETING:
                break

        return context


class DynamicRouter:
    """动态任务路由器（带学习能力）"""

    def __init__(self, agent_registry: AgentRegistry):
        self.registry = agent_registry
        self.routing_history: List[Dict[str, Any]] = []
        self.success_rates: Dict[Tuple[str, str], float] = {}
        self._lock = threading.RLock()

    def route(self, query: str, available_agents: Dict[str, Dict[str, Any]]) -> List[str]:
        """动态路由任务到合适的Agent"""
        with self._lock:
            query_lower = query.lower()
            query_hash = hashlib.md5(query_lower.encode()).hexdigest()

            routing_result = {
                "query_hash": query_hash,
                "query_preview": query_lower[:100],
                "routed_agents": [],
                "timestamp": time.time()
            }

            agent_scores = []
            for agent_id, agent_info in available_agents.items():
                score = self._calculate_score(query_lower, agent_info)
                agent_scores.append((agent_id, score))

            agent_scores.sort(key=lambda x: -x[1])
            selected_agents = [aid for aid, _ in agent_scores[:3] if _ > 0]

            if not selected_agents:
                selected_agents = list(available_agents.keys())

            routing_result["routed_agents"] = selected_agents
            self.routing_history.append(routing_result)

            return selected_agents

    def _calculate_score(self, query: str, agent_info: Dict[str, Any]) -> float:
        """计算Agent与查询的匹配分数"""
        score = 0.0

        capabilities = agent_info.get("capabilities", [])
        agent_type = agent_info.get("agent_type", "")

        keywords_map = {
            "diagnosis": ["症状", "诊断", "病因", "表现", "检查", "symptom", "diagnosis"],
            "treatment": ["治疗", "药物", "用药", "方案", "treatment", "medicine"],
            "prevention": ["预防", "保健", "健康", "预防", "prevention", "health"],
            "examination": ["检查", "化验", "检验", "结果", "examination", "test"]
        }

        for keyword_list in keywords_map.values():
            if any(kw in query for kw in keyword_list):
                score += 0.3

        if "诊断" in agent_type or "diagnosis" in capabilities:
            if any(kw in query for kw in keywords_map["diagnosis"]):
                score += 0.5

        if "治疗" in agent_type or "treatment" in capabilities:
            if any(kw in query for kw in keywords_map["treatment"]):
                score += 0.5

        if "预防" in agent_type or "prevention" in capabilities:
            if any(kw in query for kw in keywords_map["prevention"]):
                score += 0.5

        if "检查" in agent_type or "examination" in capabilities:
            if any(kw in query for kw in keywords_map["examination"]):
                score += 0.5

        return min(score, 1.0)

    def record_outcome(self, query: str, agents: List[str], success: bool) -> None:
        """记录路由结果用于优化"""
        with self._lock:
            query_hash = hashlib.md5(query.lower().encode()).hexdigest()
            for agent_id in agents:
                key = (query_hash, agent_id)
                current = self.success_rates.get(key, 0.5)
                self.success_rates[key] = current + (0.1 if success else -0.1)


class HealthMonitor:
    """Agent健康监控器"""

    def __init__(self, registry: AgentRegistry, check_interval: float = 30.0):
        self.registry = registry
        self.check_interval = check_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []

    def start(self) -> None:
        """启动健康监控"""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("健康监控已启动")

    def stop(self) -> None:
        """停止健康监控"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("健康监控已停止")

    def register_callback(self, callback: Callable) -> None:
        """注册状态变化回调"""
        self._callbacks.append(callback)

    def _monitor_loop(self) -> None:
        """监控循环"""
        while self._running:
            try:
                for agent_id in list(self.registry._agents.keys()):
                    is_healthy = self.registry.check_health(agent_id)
                    for callback in self._callbacks:
                        try:
                            callback(agent_id, is_healthy)
                        except Exception as e:
                            logger.error(f"回调执行失败: {e}")
            except Exception as e:
                logger.error(f"健康检查失败: {e}")

            time.sleep(self.check_interval)


class MultiAgentSystem:
    """2025年最佳实践多Agent协作系统"""

    def __init__(
        self,
        model: LLMProvider,
        rag_pipeline: Optional[RAGPipeline] = None,
        agent_manager: Optional[AgentManager] = None,
        pipeline_id: Optional[str] = None,
        name: str = "医疗多Agent系统",
        description: str = "一个由多个专业Agent组成的医疗问答系统",
        max_iterations: int = 10,
        verbose: bool = False,
        enable_tracing: bool = True,
        enable_metrics: bool = True,
        enable_monitoring: bool = True
    ) -> None:
        self.model = model
        self.rag_pipeline = rag_pipeline
        self.agent_manager = agent_manager or AgentManager()
        self.pipeline_id = pipeline_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.max_iterations = max_iterations
        self.verbose = verbose

        self.is_initialized = False
        self.conversation_history = []

        self.registry = AgentRegistry()
        self.memory_manager = MemoryManager()
        self.circuit_breaker = CircuitBreaker()
        self.retry_policy = RetryPolicy()
        self.tracing = TracingManager(service_name=name) if enable_tracing else None
        self.metrics = MetricsCollector() if enable_metrics else None
        self.message_bus = MessageBus()
        self.health_monitor = HealthMonitor(self.registry) if enable_monitoring else None
        self.workflow_engine = WorkflowEngine()
        self.dynamic_router = DynamicRouter(self.registry)

        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=10)

        self._setup_workflow()

    def _setup_workflow(self) -> None:
        """配置工作流引擎"""
        self.workflow_engine.register_state(WorkflowState.ROUTING, self._route_task_state)
        self.workflow_engine.register_state(WorkflowState.EXECUTING, self._execute_agents_state)
        self.workflow_engine.register_state(WorkflowState.COORDINATING, self._coordinate_state)
        self.workflow_engine.register_state(WorkflowState.COMPLETING, self._complete_state)

        self.workflow_engine.register_transition(WorkflowState.IDLE, "route", WorkflowState.ROUTING)
        self.workflow_engine.register_transition(WorkflowState.ROUTING, "execute", WorkflowState.EXECUTING)
        self.workflow_engine.register_transition(WorkflowState.EXECUTING, "coordinate", WorkflowState.COORDINATING)
        self.workflow_engine.register_transition(WorkflowState.COORDINATING, "complete", WorkflowState.COMPLETING)
        self.workflow_engine.register_transition(WorkflowState.EXECUTING, "error", WorkflowState.ERROR)
        self.workflow_engine.register_transition(WorkflowState.COORDINATING, "error", WorkflowState.ERROR)

    async def _route_task_state(self, context: WorkflowContext) -> str:
        """路由任务状态处理器"""
        if self.tracing:
            span_id = self.tracing.start_span(
                parent_id=context.workflow_id,
                operation_name="task_routing"
            )

        agents = self.dynamic_router.route(
            context.task_context.original_query,
            self.registry.get_all_agents()
        )

        context.task_context.assigned_agents = agents
        context.task_context.status = TaskStatus.ROUTING

        if self.tracing:
            self.tracing.add_event(span_id, "routing_complete", {"agents": agents})
            self.tracing.end_span(span_id)

        return "execute"

    async def _execute_agents_state(self, context: WorkflowContext) -> str:
        """执行Agent状态处理器"""
        if self.tracing:
            span_id = self.tracing.start_span(
                parent_id=context.workflow_id,
                operation_name="agent_execution"
            )

        context.task_context.status = TaskStatus.PROCESSING
        tasks = []

        for agent_name in context.task_context.assigned_agents:
            agent = self.agent_manager.get_agent(agent_name)
            if agent:
                task = asyncio.create_task(self._execute_agent(agent_name, context.task_context))
                tasks.append(task)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for agent_name, result in zip(context.task_context.assigned_agents, results):
                if isinstance(result, Exception):
                    context.agent_results[agent_name] = {
                        "response": f"处理失败: {str(result)}",
                        "error": str(result)
                    }
                else:
                    context.agent_results[agent_name] = result

        if self.tracing:
            self.tracing.add_event(span_id, "execution_complete", {"agent_count": len(context.agent_results)})
            self.tracing.end_span(span_id)

        return "coordinate"

    async def _execute_agent(self, agent_name: str, task_context: TaskContext) -> Dict[str, Any]:
        """执行单个Agent任务"""
        agent = self.agent_manager.get_agent(agent_name)
        if not agent:
            return {"response": f"Agent {agent_name} 未找到", "error": "agent_not_found"}

        try:
            self.circuit_breaker.call(agent_name, agent.reset)
            result, attempts = self.retry_policy.execute(
                agent.run,
                task_context.original_query
            )

            if self.metrics:
                self.metrics.increment(f"agent_{agent_name}_executions")
                self.metrics.timing(f"agent_{agent_name}_execution_time", time.time() - task_context.created_at)

            return result

        except CircuitBreakerOpenError:
            logger.warning(f"Circuit breaker open for {agent_name}")
            return {"response": f"Agent {agent_name} 暂时不可用", "error": "circuit_breaker_open"}
        except Exception as e:
            logger.error(f"Agent {agent_name} 执行失败: {e}")
            return {"response": f"处理失败: {str(e)}", "error": str(e)}

    async def _coordinate_state(self, context: WorkflowContext) -> str:
        """协调状态处理器"""
        if self.tracing:
            span_id = self.tracing.start_span(
                parent_id=context.workflow_id,
                operation_name="response_coordination"
            )

        context.task_context.status = TaskStatus.COORDINATING

        coordinator = self.agent_manager.get_agent("协调员")
        if coordinator and context.agent_results:
            coordination_prompt = (
                f"请协调以下多个专业Agent对用户查询的响应，提供一个连贯、一致的最终回答。\n\n"
                f"用户查询: {context.task_context.original_query}\n\n"
            )

            for agent_name, response_data in context.agent_results.items():
                agent_response = response_data.get("response", "")
                coordination_prompt += f"【{agent_name}】响应:\n{agent_response}\n\n"

            coordination_prompt += "请整合以上专业Agent的回答，提供最终一致的响应。"

            try:
                final_result, _ = self.retry_policy.execute(coordinator.run, coordination_prompt)
                context.task_context.final_result = final_result
            except Exception as e:
                logger.error(f"协调失败: {e}")
                context.task_context.final_result = {
                    "response": self._combine_responses(context.agent_results),
                    "metadata": {"coordinated": False, "error": str(e)}
                }
        else:
            context.task_context.final_result = {
                "response": self._combine_responses(context.agent_results),
                "metadata": {"coordinated": False}
            }

        if self.tracing:
            self.tracing.add_event(span_id, "coordination_complete")
            self.tracing.end_span(span_id)

        return "complete"

    def _combine_responses(self, agent_responses: Dict[str, Dict[str, Any]]) -> str:
        """组合多个Agent的响应"""
        combined = "多Agent系统回答:\n\n"
        for agent_name, response_data in agent_responses.items():
            combined += f"【{agent_name}】: {response_data.get('response', '')}\n\n"
        return combined

    async def _complete_state(self, context: WorkflowContext) -> str:
        """完成状态处理器"""
        context.task_context.status = TaskStatus.COMPLETED
        context.task_context.updated_at = time.time()
        context.task_context.execution_time = time.time() - context.task_context.created_at

        if self.metrics:
            self.metrics.increment("total_tasks_completed")
            self.metrics.timing("task_execution_time", context.task_context.execution_time)

        return "complete"

    def _log(self, message: str) -> None:
        """记录日志"""
        if self.verbose:
            print(f"[{self.name}] {message}")

    def initialize(self) -> None:
        """初始化多Agent系统"""
        if self.is_initialized:
            self._log("系统已初始化")
            return

        try:
            self._create_specialized_agents()
            self._setup_agent_registry()

            if self.health_monitor:
                self.health_monitor.register_callback(self._health_callback)
                self.health_monitor.start()

            self.is_initialized = True
            self._log("多Agent系统初始化完成")

        except Exception as e:
            logger.error(f"初始化失败: {str(e)}")
            raise

    def _health_callback(self, agent_id: str, is_healthy: bool) -> None:
        """健康状态回调"""
        if not is_healthy:
            logger.warning(f"Agent {agent_id} 健康检查失败")
            if self.metrics:
                self.metrics.increment(f"agent_{agent_id}_health_failures")

    def _setup_agent_registry(self) -> None:
        """设置Agent注册中心"""
        agent_info = {
            "任务路由器": {"type": "router", "capabilities": ["routing", "task_analysis"]},
            "协调员": {"type": "coordinator", "capabilities": ["coordination", "synthesis"]},
            "诊断Agent": {"type": "medical", "capabilities": ["diagnosis", "symptom_analysis"]},
            "治疗Agent": {"type": "medical", "capabilities": ["treatment", "medicine"]},
            "预防Agent": {"type": "medical", "capabilities": ["prevention", "health"]},
            "医学检查Agent": {"type": "medical", "capabilities": ["examination", "lab_results"]}
        }

        for agent_name, info in agent_info.items():
            agent_id = self.agent_manager.get_agent(agent_name)
            if agent_id:
                self.registry.register(
                    agent_id=str(agent_id),
                    agent_type=info["type"],
                    capabilities=info["capabilities"],
                    metadata={"name": agent_name}
                )

    def _create_specialized_agents(self) -> None:
        """创建专业Agent"""
        router_system_prompt = (
            "你是一个医疗任务路由器。你的职责是分析用户的医疗问题，确定最适合处理该问题的专业Agent。"
        )

        router_agent_id = self.agent_manager.create_agent(
            agent_type="medical",
            model=self.model,
            name="任务路由器",
            description="负责将用户查询分配给适当的专业Agent",
            system_prompt=router_system_prompt,
            verbose=self.verbose
        )
        self._log(f"创建任务路由器: {router_agent_id}")

        coordinator_system_prompt = (
            "你是一个医疗多Agent系统的协调员。你的职责是综合多个专业Agent的回答，提供最终的一致性响应。"
        )

        coordinator_agent_id = self.agent_manager.create_agent(
            agent_type="medical",
            model=self.model,
            name="协调员",
            description="负责整合多个专业Agent的回答，提供最终一致性响应",
            system_prompt=coordinator_system_prompt,
            verbose=self.verbose
        )
        self._log(f"创建协调员: {coordinator_agent_id}")

        diagnosis_system_prompt = (
            "你是一个专注于医疗诊断的Agent。你的职责是分析用户描述的症状和体征，提供可能的诊断和鉴别诊断。"
        )

        diagnosis_agent_id = self.agent_manager.create_agent(
            agent_type="medical",
            model=self.model,
            rag_pipeline=self.rag_pipeline,
            name="诊断Agent",
            description="专注于疾病诊断和症状分析",
            system_prompt=diagnosis_system_prompt,
            verbose=self.verbose
        )
        self._log(f"创建诊断Agent: {diagnosis_agent_id}")

        treatment_system_prompt = (
            "你是一个专注于医疗治疗的Agent。你的职责是提供关于治疗方法、药物使用和治疗计划的信息。"
        )

        treatment_agent_id = self.agent_manager.create_agent(
            agent_type="medical",
            model=self.model,
            rag_pipeline=self.rag_pipeline,
            name="治疗Agent",
            description="专注于治疗方案和药物建议",
            system_prompt=treatment_system_prompt,
            verbose=self.verbose
        )
        self._log(f"创建治疗Agent: {treatment_agent_id}")

        prevention_system_prompt = (
            "你是一个专注于疾病预防和健康维护的Agent。你的职责是提供关于健康生活方式、疾病预防和健康监测的信息。"
        )

        prevention_agent_id = self.agent_manager.create_agent(
            agent_type="medical",
            model=self.model,
            rag_pipeline=self.rag_pipeline,
            name="预防Agent",
            description="专注于健康预防和生活方式建议",
            system_prompt=prevention_system_prompt,
            verbose=self.verbose
        )
        self._log(f"创建预防Agent: {prevention_agent_id}")

        examination_system_prompt = (
            "你是一个专注于医学检查和实验室结果解释的Agent。你的职责是提供关于各种医学检查、实验室检验的信息和结果解读。"
        )

        examination_agent_id = self.agent_manager.create_agent(
            agent_type="medical",
            model=self.model,
            rag_pipeline=self.rag_pipeline,
            name="医学检查Agent",
            description="专注于医学检查和实验室结果解释",
            system_prompt=examination_system_prompt,
            verbose=self.verbose
        )
        self._log(f"创建医学检查Agent: {examination_agent_id}")

    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """执行多Agent系统处理用户查询"""
        if not self.is_initialized:
            self.initialize()

        start_time = time.time()

        trace_id = self.tracing.start_span(operation_name="full_pipeline") if self.tracing else None

        self.conversation_history.append({
            "role": "user",
            "content": query
        })

        task_context = TaskContext(
            task_id=str(uuid.uuid4()),
            original_query=query,
            metadata=kwargs
        )

        workflow_context = WorkflowContext(
            workflow_id=str(uuid.uuid4()),
            task_context=task_context
        )

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                final_context = loop.run_until_complete(
                    self.workflow_engine.execute(workflow_context)
                )
            finally:
                loop.close()

            result = task_context.final_result or {}

        except Exception as e:
            logger.error(f"工作流执行失败: {str(e)}")
            result = {
                "query": query,
                "response": f"处理失败: {str(e)}",
                "metadata": {"error": str(e)}
            }
            if self.tracing:
                self.tracing.add_event(trace_id, "workflow_error", {"error": str(e)})

        if trace_id and self.tracing:
            self.tracing.end_span(trace_id)

        self.conversation_history.append({
            "role": "assistant",
            "content": result.get("response", "")
        })

        execution_time = time.time() - start_time

        if "metadata" not in result:
            result["metadata"] = {}

        result["metadata"]["execution_time"] = execution_time
        result["metadata"]["pipeline_id"] = self.pipeline_id
        result["metadata"]["task_id"] = task_context.task_id

        self.memory_manager.add_short_term(
            f"用户查询: {query}\n系统响应: {result.get('response', '')[:500]}",
            {"task_id": task_context.task_id, "execution_time": execution_time}
        )

        if self.metrics:
            self.metrics.timing("total_execution_time", execution_time)

        return result

    async def run_async(self, query: str, **kwargs) -> Dict[str, Any]:
        """异步执行多Agent系统"""
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            lambda: self.run(query, **kwargs)
        )

    def run_batch(self, queries: List[str], **kwargs) -> List[Dict[str, Any]]:
        """批量执行查询"""
        return [self.run(query, **kwargs) for query in queries]

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "is_initialized": self.is_initialized,
            "registered_agents": len(self.registry.get_all_agents()),
            "memory_summary": self.memory_manager.get_summary(),
            "message_bus_stats": self.message_bus.get_stats(),
            "tracing_enabled": self.tracing is not None,
            "metrics_enabled": self.metrics is not None,
            "monitoring_enabled": self.health_monitor is not None
        }

    def get_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        if self.metrics:
            return self.metrics.get_summary()
        return {}

    def get_traces(self, trace_id: Optional[str] = None) -> Any:
        """获取追踪记录"""
        if self.tracing:
            if trace_id:
                return self.tracing.get_trace(trace_id)
            return self.tracing.get_all_traces()
        return {}

    def reset(self) -> None:
        """重置系统状态"""
        self.conversation_history = []
        self.memory_manager = MemoryManager()
        self.circuit_breaker = CircuitBreaker()
        self.dynamic_router = DynamicRouter(self.registry)

        for agent_id in self.registry.get_all_agents():
            agent = self.agent_manager.get_agent(agent_id)
            if agent:
                try:
                    agent.reset()
                except Exception:
                    pass

        self._log("系统已重置")

    def shutdown(self) -> None:
        """关闭系统"""
        if self.health_monitor:
            self.health_monitor.stop()

        self._executor.shutdown(wait=False)

        for agent_id in list(self.registry.get_all_agents().keys()):
            self.registry.unregister(agent_id)

        logger.info("系统已关闭")
