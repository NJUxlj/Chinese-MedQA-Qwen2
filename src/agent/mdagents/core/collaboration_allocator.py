"""
动态协作分配模块 - 根据复杂度动态选择协作模式
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..config import ComplexityLevel, AgentType, CollaborationPattern
from ..tools.logger import setup_logger

@dataclass
class TeamConfig:
    """团队配置"""
    agents: List[AgentType]
    max_response_time: int = 60
    min_confidence: float = 0.7
    consensus_method: str = "simple_majority"
    estimated_time: int = 60
    collaboration_pattern: str = "sequential"
    resource_requirements: Dict[str, Any] = None
    description: str = ""
    
    def __post_init__(self):
        if self.resource_requirements is None:
            self.resource_requirements = {
                "cpu": "low",
                "memory": "low",
                "network": "low"
            }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agents": [agent.value for agent in self.agents],
            "max_response_time": self.max_response_time,
            "min_confidence": self.min_confidence,
            "consensus_method": self.consensus_method,
            "estimated_time": self.estimated_time,
            "collaboration_pattern": self.collaboration_pattern,
            "resource_requirements": self.resource_requirements,
            "description": self.description
        }

@dataclass
class CollaborationStrategy:
    """协作策略"""
    name: str
    complexity_level: ComplexityLevel
    team_config: TeamConfig
    adaptive_rules: List[str] = None
    fallback_strategies: List[str] = None
    
    def __post_init__(self):
        if self.adaptive_rules is None:
            self.adaptive_rules = []
        if self.fallback_strategies is None:
            self.fallback_strategies = []
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "complexity_level": self.complexity_level,
            "complexity_level_value": self.complexity_level.value,
            "team_config": self.team_config.to_dict(),
            "adaptive_rules": self.adaptive_rules,
            "fallback_strategies": self.fallback_strategies,
            "agents_needed": [agent.value for agent in self.team_config.agents],
            "pattern": self.complexity_level,
            "pattern_value": self.complexity_level.value,
            "reasoning": f"为{self.complexity_level.value}复杂度查询分配{self.name}",
            "estimated_time": self.team_config.estimated_time,
            "confidence": self.team_config.min_confidence
        }

class CollaborationAllocator:
    """动态协作分配器"""
    
    def __init__(self):
        """初始化协作分配器"""
        self.logger = setup_logger(self.__class__.__name__)
        
        # 预定义的协作策略
        self.strategies = self._initialize_strategies()
        
        # 专科医生映射
        self.specialist_mapping = {
            'cardiology': AgentType.SPECIALIST,
            'neurology': AgentType.SPECIALIST,
            'oncology': AgentType.SPECIALIST,
            'endocrinology': AgentType.SPECIALIST,
            'pulmonology': AgentType.SPECIALIST,
            'gastroenterology': AgentType.SPECIALIST,
            'nephrology': AgentType.SPECIALIST,
            'rheumatology': AgentType.SPECIALIST,
            'dermatology': AgentType.SPECIALIST,
            'psychiatry': AgentType.SPECIALIST
        }
    
    def _initialize_strategies(self) -> Dict[ComplexityLevel, CollaborationStrategy]:
        """初始化协作策略"""
        strategies = {}
        
        # 低复杂度策略：单专家模式
        low_complexity_strategy = CollaborationStrategy(
            name="单专家快速处理",
            complexity_level=ComplexityLevel.SIMPLE,
            team_config=TeamConfig(
                agents=[AgentType.PCC],
                max_response_time=60,
                min_confidence=0.7
            ),
            adaptive_rules=[
                "如PCC无法处理则升级为中等复杂度",
                "如发现复杂症状则重新评估"
            ],
            fallback_strategies=["medium_complexity"]
        )
        
        # 中等复杂度策略：多学科团队
        medium_complexity_strategy = CollaborationStrategy(
            name="多学科团队协作",
            complexity_level=ComplexityLevel.MODERATE,
            team_config=TeamConfig(
                agents=[AgentType.PCC, AgentType.SPECIALIST, AgentType.REVIEWER],
                max_response_time=120,
                min_confidence=0.8
            ),
            adaptive_rules=[
                "根据涉及的专科动态调整专家类型",
                "如意见分歧严重则升级为高复杂度"
            ],
            fallback_strategies=["high_complexity"]
        )
        
        # 高复杂度策略：综合护理团队
        high_complexity_strategy = CollaborationStrategy(
            name="综合护理团队",
            complexity_level=ComplexityLevel.COMPLEX,
            team_config=TeamConfig(
                agents=[AgentType.RECRUITER, AgentType.PCC, AgentType.SPECIALIST, 
                       AgentType.SPECIALIST, AgentType.SPECIALIST, AgentType.REVIEWER, AgentType.INTEGRATOR],
                max_response_time=300,
                min_confidence=0.9
            ),
            adaptive_rules=[
                "根据专科数量动态调整专家数量",
                "如处理超时则优先保证关键决策"
            ],
            fallback_strategies=["emergency_protocol"]
        )
        
        # 初始化协作策略
        self.strategies = {
            ComplexityLevel.SIMPLE: low_complexity_strategy,
            ComplexityLevel.MODERATE: medium_complexity_strategy,
            ComplexityLevel.COMPLEX: high_complexity_strategy
        }
        
        return self.strategies
    
    def allocate_collaboration(self, complexity_level: ComplexityLevel, 
                             query: str, context: Dict[str, Any] = None) -> CollaborationStrategy:
        """
        分配协作策略
        
        Args:
            complexity_level: 复杂度等级
            query: 医疗查询
            context: 上下文信息
            
        Returns:
            协作策略
        """
        self.logger.info(f"为复杂度等级 {complexity_level.value} 分配协作策略")
        
        base_strategy = self.strategies[complexity_level]
        
        # 根据查询内容动态调整策略
        adapted_strategy = self._adapt_strategy(base_strategy, query, context)
        
        self.logger.info(f"选择的协作策略: {adapted_strategy.name}")
        self.logger.info(f"团队配置: {len(adapted_strategy.team_config.agents)} 个智能体")
        self.logger.info(f"预计处理时间: {adapted_strategy.team_config.estimated_time} 秒")
        
        return adapted_strategy
    
    def _adapt_strategy(self, strategy: CollaborationStrategy, 
                       query: str, context: Dict[str, Any] = None) -> CollaborationStrategy:
        """根据查询内容自适应调整策略"""
        adapted_agents = strategy.team_config.agents.copy()
        adapted_consensus = strategy.team_config.consensus_method
        adapted_time = strategy.team_config.estimated_time
        
        # 根据涉及的专科调整专家类型
        required_specialties = self._identify_required_specialties(query)
        
        if strategy.complexity_level == ComplexityLevel.MODERATE:
            # 中等复杂度下，根据专科调整
            if required_specialties:
                # 保持PCC和REVIEWER，替换SPECIALIST为具体专科
                adapted_agents = [AgentType.PCC] + [self.specialist_mapping[s] for s in required_specialties[:2]] + [AgentType.REVIEWER]
        
        elif strategy.complexity_level == ComplexityLevel.COMPLEX:
            # 高复杂度下，根据专科数量调整
            if len(required_specialties) > 3:
                # 添加更多专家
                additional_specialists = [self.specialist_mapping[s] for s in required_specialties[3:6]]
                adapted_agents = [AgentType.RECRUITER, AgentType.PCC] + \
                               [self.specialist_mapping[s] for s in required_specialties[:3]] + \
                               additional_specialists + [AgentType.REVIEWER, AgentType.INTEGRATOR]
        
        # 根据上下文调整共识方法
        if context:
            if context.get('high_stakes', False):
                adapted_consensus = "staged_consensus"
                adapted_time *= 1.2  # 增加处理时间
            
            if context.get('urgent', False):
                adapted_consensus = "majority_vote"  # 快速决策
                adapted_time *= 0.8  # 减少处理时间
        
        # 创建调整后的团队配置
        adapted_team_config = TeamConfig(
            agents=adapted_agents,
            collaboration_pattern=strategy.team_config.collaboration_pattern,
            consensus_method=adapted_consensus,
            estimated_time=adapted_time,
            resource_requirements=strategy.team_config.resource_requirements.copy(),
            description=strategy.team_config.description
        )
        
        # 创建调整后的策略
        adapted_strategy = CollaborationStrategy(
            name=strategy.name,
            complexity_level=strategy.complexity_level,
            team_config=adapted_team_config,
            adaptive_rules=strategy.adaptive_rules,
            fallback_strategies=strategy.fallback_strategies
        )
        
        return adapted_strategy
    
    def _identify_required_specialties(self, query: str) -> List[str]:
        """识别查询所需的医疗专科"""
        required_specialties = []
        query_lower = query.lower()
        
        # 专科关键词匹配
        specialty_keywords = {
            'cardiology': ['心脏', '心律', '血压', '心肌', '心电图', '冠心病'],
            'neurology': ['神经', '脑', '癫痫', '头痛', '瘫痪', '昏迷'],
            'oncology': ['肿瘤', '癌症', '恶性', '转移', '化疗', '放疗'],
            'endocrinology': ['糖尿病', '甲状腺', '激素', '血糖', '胰岛素'],
            'pulmonology': ['肺', '呼吸', '哮喘', '肺炎', '咳嗽', '气短'],
            'gastroenterology': ['胃', '肠', '消化', '腹痛', '腹泻', '便秘'],
            'nephrology': ['肾', '尿', '透析', '肾功能', '肾炎'],
            'rheumatology': ['关节', '风湿', '关节炎', '红斑狼疮', '自身免疫'],
            'dermatology': ['皮肤', '皮疹', '湿疹', '过敏', '荨麻疹'],
            'psychiatry': ['抑郁', '焦虑', '精神', '心理', '失眠', '幻觉']
        }
        
        for specialty, keywords in specialty_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                required_specialties.append(specialty)
        
        return required_specialties
    
    def get_agent_distribution(self, strategy: CollaborationStrategy) -> Dict[str, int]:
        """获取智能体分布统计"""
        distribution = {}
        for agent_type in strategy.team_config.agents:
            agent_name = agent_type.value
            distribution[agent_name] = distribution.get(agent_name, 0) + 1
        return distribution
    
    def estimate_resource_usage(self, strategy: CollaborationStrategy) -> Dict[str, Any]:
        """估算资源使用量"""
        agent_count = len(strategy.team_config.agents)
        
        # 根据智能体数量和类型估算资源
        cpu_usage = "low"
        memory_usage = "low"
        
        if agent_count >= 5:
            cpu_usage = "high"
            memory_usage = "high"
        elif agent_count >= 3:
            cpu_usage = "medium"
            memory_usage = "medium"
        
        return {
            "estimated_agents": agent_count,
            "cpu_requirement": cpu_usage,
            "memory_requirement": memory_usage,
            "estimated_time_seconds": strategy.team_config.estimated_time,
            "collaboration_pattern": strategy.team_config.collaboration_pattern.value,
            "consensus_method": strategy.team_config.consensus_method
        }
    
    def validate_strategy(self, strategy: CollaborationStrategy, 
                         available_agents: List[AgentType] = None) -> Tuple[bool, List[str]]:
        """验证策略可行性"""
        issues = []
        is_valid = True
        
        # 检查智能体可用性
        if available_agents:
            missing_agents = []
            for required_agent in strategy.team_config.agents:
                if required_agent not in available_agents:
                    missing_agents.append(required_agent.value)
            
            if missing_agents:
                issues.append(f"缺少必要的智能体: {', '.join(missing_agents)}")
                is_valid = False
        
        # 检查资源限制
        if len(strategy.team_config.agents) > 10:
            issues.append("智能体数量过多，可能导致协调困难")
            is_valid = False
        
        # 检查协作模式兼容性
        if strategy.complexity_level == ComplexityLevel.SIMPLE and \
           len(strategy.team_config.agents) > 1:
            issues.append("低复杂度查询不需要多智能体协作")
            is_valid = False
        
        return is_valid, issues