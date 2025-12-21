"""
系统配置模块
提供MDAgents系统的完整配置管理功能
"""

import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class LLMConfig:
    """LLM配置"""
    model_name: str = "qwen2"
    api_base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 30
    retry_times: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LLMConfig':
        return cls(**config_dict)


@dataclass
class AgentPoolConfig:
    """智能体池配置"""
    pcc_agents: int = 1
    specialist_agents: int = 3
    moderator_agents: int = 1
    recruiter_agents: int = 1
    reviewer_agents: int = 1
    integrator_agents: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentPoolConfig':
        return cls(**config_dict)


@dataclass
class ConsensusConfig:
    """共识机制配置"""
    default_algorithm: str = "weighted_voting"
    min_expert_count: int = 2
    consensus_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.consensus_thresholds is None:
            self.consensus_thresholds = {
                "unanimous": 1.0,
                "super_majority": 0.8,
                "majority": 0.6,
                "plurality": 0.4,
                "minimum": 0.3
            }
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConsensusConfig':
        return cls(**config_dict)


@dataclass
class ComplexityConfig:
    """复杂度分析配置"""
    enable_ai_analysis: bool = True
    manual_complexity_mapping: Dict[str, str] = None
    complexity_keywords: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.manual_complexity_mapping is None:
            self.manual_complexity_mapping = {
                "急诊": "COMPLEX",
                "复杂": "COMPLEX",
                "罕见": "COMPLEX",
                "严重": "COMPLEX",
                "普通": "SIMPLE",
                "常见": "MODERATE",
                "一般": "SIMPLE"
            }
        
        if self.complexity_keywords is None:
            self.complexity_keywords = {
                "SIMPLE": ["头痛", "感冒", "发烧", "咳嗽", "胃痛", "普通", "常见"],
                "MODERATE": ["胸痛", "呼吸困难", "腹痛", "头晕", "失眠", "高血压"],
                "COMPLEX": ["多系统疾病", "罕见病", "并发症", "急诊", "手术", "肿瘤"]
            }
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ComplexityConfig':
        return cls(**config_dict)


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_console: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LoggingConfig':
        return cls(**config_dict)


@dataclass
class SystemConfig:
    """系统主配置"""
    llm: LLMConfig
    agent_pool: AgentPoolConfig
    consensus: ConsensusConfig
    complexity: ComplexityConfig
    logging: LoggingConfig
    
    # 系统参数
    max_concurrent_sessions: int = 10
    session_timeout: int = 3600  # 1小时
    enable_async_processing: bool = True
    cache_enabled: bool = True
    cache_ttl: int = 300  # 5分钟
    
    # 协作配置
    collaboration_timeout: int = 300  # 5分钟
    expert_response_timeout: int = 120  # 2分钟
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SystemConfig':
        return cls(
            llm=LLMConfig.from_dict(config_dict.get("llm", {})),
            agent_pool=AgentPoolConfig.from_dict(config_dict.get("agent_pool", {})),
            consensus=ConsensusConfig.from_dict(config_dict.get("consensus", {})),
            complexity=ComplexityConfig.from_dict(config_dict.get("complexity", {})),
            logging=LoggingConfig.from_dict(config_dict.get("logging", {})),
            **{k: v for k, v in config_dict.items() if k not in [
                "llm", "agent_pool", "consensus", "complexity", "logging"
            ]}
        )


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config_path = config_path
        self._config: Optional[SystemConfig] = None
    
    @property
    def config(self) -> SystemConfig:
        """获取系统配置"""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def load_config(self) -> SystemConfig:
        """加载配置"""
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                return SystemConfig.from_dict(config_dict)
            except Exception as e:
                print(f"加载配置文件失败: {e}")
                print("使用默认配置")
        
        return self._create_default_config()
    
    def save_config(self, config: SystemConfig, config_path: Optional[str] = None):
        """保存配置"""
        path = config_path or self.config_path
        if not path:
            raise ValueError("未指定配置文件路径")
        
        try:
            # 确保目录存在
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            raise RuntimeError(f"保存配置文件失败: {e}")
    
    def _create_default_config(self) -> SystemConfig:
        """创建默认配置"""
        return SystemConfig(
            llm=LLMConfig(),
            agent_pool=AgentPoolConfig(),
            consensus=ConsensusConfig(),
            complexity=ComplexityConfig(),
            logging=LoggingConfig()
        )
    
    def update_config(self, updates: Dict[str, Any]):
        """更新配置"""
        current_config = self.config
        
        # 更新LLM配置
        if "llm" in updates:
            current_config.llm = LLMConfig.from_dict(updates["llm"])
        
        # 更新智能体池配置
        if "agent_pool" in updates:
            current_config.agent_pool = AgentPoolConfig.from_dict(updates["agent_pool"])
        
        # 更新共识配置
        if "consensus" in updates:
            current_config.consensus = ConsensusConfig.from_dict(updates["consensus"])
        
        # 更新复杂度配置
        if "complexity" in updates:
            current_config.complexity = ComplexityConfig.from_dict(updates["complexity"])
        
        # 更新日志配置
        if "logging" in updates:
            current_config.logging = LoggingConfig.from_dict(updates["logging"])
        
        # 更新其他系统参数
        for key, value in updates.items():
            if key not in ["llm", "agent_pool", "consensus", "complexity", "logging"]:
                setattr(current_config, key, value)
        
        self._config = current_config
    
    def get_llm_config(self) -> LLMConfig:
        """获取LLM配置"""
        return self.config.llm
    
    def get_agent_pool_config(self) -> AgentPoolConfig:
        """获取智能体池配置"""
        return self.config.agent_pool
    
    def get_consensus_config(self) -> ConsensusConfig:
        """获取共识配置"""
        return self.config.consensus
    
    def get_complexity_config(self) -> ComplexityConfig:
        """获取复杂度配置"""
        return self.config.complexity
    
    def get_logging_config(self) -> LoggingConfig:
        """获取日志配置"""
        return self.config.logging
    
    def validate_config(self) -> List[str]:
        """验证配置有效性"""
        errors = []
        config = self.config
        
        # 验证LLM配置
        if not config.llm.model_name:
            errors.append("LLM模型名称不能为空")
        
        if config.llm.max_tokens <= 0:
            errors.append("LLM最大token数必须大于0")
        
        if not 0 <= config.llm.temperature <= 2:
            errors.append("LLM温度参数必须在0-2之间")
        
        # 验证智能体池配置
        if config.agent_pool.pcc_agents < 1:
            errors.append("全科医生智能体数量至少为1")
        
        # 验证共识配置
        if config.consensus.min_expert_count < 2:
            errors.append("最小专家数量至少为2")
        
        # 验证系统配置
        if config.max_concurrent_sessions <= 0:
            errors.append("最大并发会话数必须大于0")
        
        if config.session_timeout <= 0:
            errors.append("会话超时时间必须大于0")
        
        return errors


# 全局配置管理器实例
_config_manager = None

def get_config_manager() -> ConfigManager:
    """获取全局配置管理器"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_system_config() -> SystemConfig:
    """获取系统配置"""
    return get_config_manager().config

def load_config_from_file(config_path: str) -> SystemConfig:
    """从文件加载配置"""
    manager = ConfigManager(config_path)
    return manager.config

def create_default_config_file(config_path: str):
    """创建默认配置文件"""
    manager = ConfigManager()
    default_config = manager._create_default_config()
    manager.save_config(default_config, config_path)