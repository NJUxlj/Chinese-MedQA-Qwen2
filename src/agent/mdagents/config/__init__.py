"""
配置模块 - 管理MDAgents系统的配置参数
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from enum import Enum

# 导入新的系统配置
from .system_config import (
    SystemConfig, LLMConfig, AgentPoolConfig, 
    ConsensusConfig, ComplexityConfig, LoggingConfig,
    ConfigManager, get_config_manager, get_system_config,
    load_config_from_file, create_default_config_file
)

# 原有配置枚举和类（保持兼容性）
class ComplexityLevel(Enum):
    """复杂度等级枚举"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

class AgentType(Enum):
    """智能体类型枚举"""
    MODERATOR = "moderator"
    PCC = "pcc"  # Primary Care Clinician
    SPECIALIST = "specialist"
    RECRUITER = "recruiter"
    REVIEWER = "reviewer"
    INTEGRATOR = "integrator"

class CollaborationPattern(Enum):
    """协作模式枚举"""
    SINGLE_AGENT = "single_agent"
    GROUP_COLLABORATION = "group_collaboration"
    HIERARCHICAL_CONSULTATION = "hierarchical_consultation"

class DecisionType(Enum):
    """决策类型枚举"""
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    MEDICATION = "medication"
    REFERRAL = "referral"

@dataclass
class ComplexityThresholds:
    """复杂度阈值配置"""
    simple_max_score: float = 0.3
    moderate_max_score: float = 0.6
    complex_min_score: float = 0.6
    high_complexity_threshold: float = 0.8
    
    # 复杂度特征权重
    text_length_weight: float = 0.1
    symptom_count_weight: float = 0.2
    medical_terms_weight: float = 0.2
    specialty_count_weight: float = 0.2
    chronic_conditions_weight: float = 0.15
    emergency_keywords_weight: float = 0.15
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ComplexityThresholds':
        return cls(**config_dict)

@dataclass
class AgentConfig:
    """智能体配置"""
    name: str
    agent_type: str
    description: str
    capabilities: List[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    system_prompt: str = ""
    specialty: Optional[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []

# 保持原有的MDAgentsConfig类以确保向后兼容
@dataclass
class MDAgentsConfig:
    """MDAgents主配置（兼容性别名）"""
    complexity_thresholds: Dict[str, float] = None
    agents: Dict[str, AgentConfig] = None
    collaboration_patterns: Dict[str, str] = None
    consensus_methods: List[str] = None
    api_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.complexity_thresholds is None:
            self.complexity_thresholds = {
                "low": 0.3,
                "medium": 0.6,
                "high": 1.0
            }
        
        if self.collaboration_patterns is None:
            self.collaboration_patterns = {
                "simple": "single_agent",
                "moderate": "group_collaboration", 
                "complex": "hierarchical_consultation"
            }
        
        if self.consensus_methods is None:
            self.consensus_methods = [
                "simple_majority",
                "weighted_voting",
                "borda_count",
                "condorcet_method",
                "iterative_consensus",
                "fuzzy_consensus"
            ]
        
        if self.api_config is None:
            self.api_config = {
                "timeout": 30,
                "max_retries": 3,
                "temperature_range": (0.3, 0.9),
                "stream": False
            }

# 导出所有配置类和函数
__all__ = [
    # 新的系统配置
    "SystemConfig",
    "LLMConfig", 
    "AgentPoolConfig",
    "ConsensusConfig",
    "ComplexityConfig",
    "LoggingConfig",
    "ConfigManager",
    "get_config_manager",
    "get_system_config", 
    "load_config_from_file",
    "create_default_config_file",
    
    # 原有配置（保持兼容性）
    "ComplexityLevel",
    "ComplexityThresholds",
    "AgentType",
    "CollaborationPattern", 
    "DecisionType",
    "AgentConfig",
    "MDAgentsConfig"
]