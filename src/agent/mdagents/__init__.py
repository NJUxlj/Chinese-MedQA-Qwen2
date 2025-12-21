"""
MDAgents: Medical Decision-making Agents

基于MDAgents论文实现的医疗决策多智能体系统，支持动态协作分配、
复杂度评估、专家代理协作和共识机制。

主要模块：
- core: 核心控制模块
- agents: 智能体实现
- tools: 工具和辅助功能
- config: 配置管理
"""

from .core.main_controller import MainController
from .agents.base_agent import BaseAgent
from .core.complexity_analyzer import ComplexityAnalyzer
from .core.collaboration_allocator import CollaborationAllocator
from .core.consensus_mechanism import ConsensusMechanism

__version__ = "1.0.0"
__author__ = "MDAgents Implementation Team"

__all__ = [
    "MainController",
    "BaseAgent", 
    "ComplexityAnalyzer",
    "CollaborationAllocator",
    "ConsensusMechanism"
]