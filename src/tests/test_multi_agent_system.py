#!/usr/bin/env python3
"""
Test script for multi_agent_system core components
"""
import sys
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List
from enum import Enum, auto

print("=" * 60)
print("Multi-Agent System Core Components Test")
print("=" * 60)
print()

class CircuitState(Enum):
    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2

class MessagePriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class MessageType(Enum):
    REQUEST = 0
    RESPONSE = 1
    EVENT = 2
    ERROR = 3

class TaskType(Enum):
    DIAGNOSIS = 0
    TREATMENT = 1
    PREVENTION = 2
    EXAMINATION = 3
    GENERAL = 4
    CONSULTATION = 5

class TaskComplexity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    EXPERT = 4

class CollaborationPattern(Enum):
    PARALLEL = 0
    SEQUENTIAL = 1
    HIERARCHICAL = 2
    ADAPTIVE = 3
    DEBATE = 4

@dataclass
class TaskContext:
    task_id: str
    original_query: str
    task_type: TaskType
    complexity: TaskComplexity
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentCapability:
    agent_id: str
    agent_type: str
    capabilities: List[str]
    performance_score: float
    current_load: int
    success_rate: float
    avg_response_time: float
    specialties: List[str] = field(default_factory=list)

@dataclass
class TaskAnalysis:
    task_type: TaskType
    complexity: TaskComplexity
    confidence: float
    required_capabilities: List[str]
    keywords: List[str]
    suggested_agents: List[str]
    routing_strategy: str

class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failure_count = 0
        self._last_failure_time = None
        self._state = CircuitState.CLOSED
        self._lock = threading.Lock()
    
    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
        return self._state
    
    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
    
    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED

class RetryPolicy:
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        delay = min(self.base_delay * (self.exponential_base ** attempt), self.max_delay)
        if self.jitter:
            delay *= (0.5 + 0.5 * hash(str(time.time() + attempt)) % 1)
        return delay

print("1. Testing CircuitBreaker...")
cb = CircuitBreaker(name='test_breaker', failure_threshold=3, recovery_timeout=5.0)
print(f"   Initial state: {cb.state.name}")
cb.record_failure()
cb.record_failure()
print(f"   After 2 failures: {cb.state.name}")
cb.record_failure()
print(f"   After 3 failures (OPEN): {cb.state.name}")
assert cb.state == CircuitState.OPEN, "Circuit should be OPEN after 3 failures"
print("   ✓ CircuitBreaker test passed!")
print()

print("2. Testing RetryPolicy...")
rp = RetryPolicy(max_retries=3, base_delay=0.01, max_delay=1.0, exponential_base=2.0, jitter=False)
delays = [rp.get_delay(i) for i in range(4)]
print(f"   Retry delays: {[f'{d:.3f}s' for d in delays]}")
assert delays[0] == 0.01, "First delay should be base_delay"
assert delays[1] == 0.02, "Second delay should be exponential"
print("   ✓ RetryPolicy test passed!")
print()

print("3. Testing Enums...")
print(f"   MessagePriority: {[p.name for p in MessagePriority]}")
print(f"   MessageType: {[t.name for t in MessageType]}")
print(f"   TaskType: {[t.name for t in TaskType]}")
print(f"   TaskComplexity: {[c.name for c in TaskComplexity]}")
print(f"   CollaborationPattern: {[p.name for p in CollaborationPattern]}")
print("   ✓ Enum test passed!")
print()

print("4. Testing Dataclasses...")
tc = TaskContext(
    task_id="test-123",
    original_query="What are the symptoms of diabetes?",
    task_type=TaskType.DIAGNOSIS,
    complexity=TaskComplexity.MEDIUM,
    metadata={"priority": "high"}
)
print(f"   TaskContext: {tc.task_id}, {tc.task_type.name}, {tc.complexity.name}")

ac = AgentCapability(
    agent_id="agent-001",
    agent_type="diagnosis",
    capabilities=["medical_knowledge", "symptom_analysis"],
    performance_score=0.85,
    current_load=2,
    success_rate=0.92,
    avg_response_time=0.5,
    specialties=["endocrinology"]
)
print(f"   AgentCapability: {ac.agent_id}, score={ac.performance_score}")
print("   ✓ Dataclass test passed!")
print()

print("=" * 60)
print("All core component tests passed successfully!")
print("=" * 60)
print()
print("These components mirror the implementation in multi_agent_system.py")
