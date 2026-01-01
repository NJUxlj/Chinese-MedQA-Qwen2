"""
MDAgents API 数据模型
定义医疗多智能体系统的请求和响应数据结构
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ComplexityLevelEnum(str, Enum):
    """复杂度等级枚举"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class AgentTypeEnum(str, Enum):
    """智能体类型枚举"""
    MODERATOR = "moderator"
    PCC = "pcc"
    SPECIALIST = "specialist"
    RECRUITER = "recruiter"
    REVIEWER = "reviewer"
    INTEGRATOR = "integrator"


class CollaborationPatternEnum(str, Enum):
    """协作模式枚举"""
    SINGLE_AGENT = "single_agent"
    GROUP_COLLABORATION = "group_collaboration"
    HIERARCHICAL_CONSULTATION = "hierarchical_consultation"


class ConsensusMethodEnum(str, Enum):
    """共识方法枚举"""
    SIMPLE_MAJORITY = "simple_majority"
    WEIGHTED_VOTING = "weighted_voting"
    BORDA_COUNT = "borda_count"
    CONDORCET = "condorcet"
    ITERATIVE_CONSENSUS = "iterative_consensus"
    FUZZY_CONSENSUS = "fuzzy_consensus"


# ==================== 请求模型 ====================

class MedicalQueryRequest(BaseModel):
    """医疗查询请求"""
    query: str = Field(..., description="用户医疗问题描述", min_length=1, max_length=5000)
    patient_context: Optional[Dict[str, Any]] = Field(None, description="患者背景信息")
    options: Optional[Dict[str, Any]] = Field(None, description="可选配置")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "患者男，65岁，有高血压和糖尿病史，近期出现胸痛和呼吸困难",
                "patient_context": {
                    "age": 65,
                    "gender": "male",
                    "history": ["高血压", "糖尿病"]
                },
                "options": {
                    "stream": True,
                    "show_steps": True
                }
            }
        }


class AgentConsultationRequest(BaseModel):
    """智能体会诊请求"""
    query: str = Field(..., description="会诊问题", min_length=1, max_length=5000)
    agent_types: List[AgentTypeEnum] = Field(..., description="参与的智能体类型")
    consensus_method: Optional[ConsensusMethodEnum] = Field(None, description="共识方法")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "患者诊断不明确，需要多学科会诊",
                "agent_types": ["pcc", "specialist", "reviewer"],
                "consensus_method": "weighted_voting"
            }
        }


class ComplexityAnalysisRequest(BaseModel):
    """复杂度分析请求"""
    query: str = Field(..., description="待分析的医疗查询", min_length=1, max_length=5000)
    include_features: bool = Field(True, description="是否返回详细特征")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "患者有多种慢性病，需要复杂治疗方案",
                "include_features": True
            }
        }


class CollaborationStrategyRequest(BaseModel):
    """协作策略请求"""
    complexity_level: ComplexityLevelEnum = Field(..., description="查询复杂度等级")
    query: str = Field(..., description="医疗查询内容")
    context: Optional[Dict[str, Any]] = Field(None, description="额外上下文信息")
    
    class Config:
        json_schema_extra = {
            "example": {
                "complexity_level": "moderate",
                "query": "患者有胸痛和呼吸困难",
                "context": {"urgent": True}
            }
        }


class ReviewRequest(BaseModel):
    """医疗建议审查请求"""
    medical_advice: str = Field(..., description="待审查的医疗建议")
    agent_recommendations: Optional[List[Dict[str, Any]]] = Field(None, description="智能体推荐")
    complexity_level: Optional[ComplexityLevelEnum] = Field(None, description="复杂度等级")
    
    class Config:
        json_schema_extra = {
            "example": {
                "medical_advice": "建议患者服用阿司匹林并定期复查",
                "agent_recommendations": [
                    {"diagnosis": "冠心病", "confidence": 0.9}
                ],
                "complexity_level": "moderate"
            }
        }


# ==================== 响应模型 ====================

class ComplexityFeaturesResponse(BaseModel):
    """复杂度特征响应"""
    text_length: int = Field(..., description="文本长度")
    symptom_count: int = Field(..., description="症状数量")
    medical_terms_count: int = Field(..., description="医疗术语数量")
    specialty_count: int = Field(..., description="涉及专科数量")
    chronic_conditions_mentioned: bool = Field(..., description="是否提及慢性病")
    emergency_keywords: List[str] = Field(default_factory=list, description="急症关键词列表")
    diagnostic_complexity: float = Field(..., description="诊断复杂度分数")
    treatment_complexity: float = Field(..., description="治疗复杂度分数")


class ComplexityAnalysisResponse(BaseModel):
    """复杂度分析响应"""
    query: str = Field(..., description="原始查询")
    complexity_score: float = Field(..., description="复杂度分数 (0-1)")
    complexity_level: ComplexityLevelEnum = Field(..., description="复杂度等级")
    features: Optional[ComplexityFeaturesResponse] = Field(None, description="详细特征")
    confidence: float = Field(..., description="分析置信度")
    recommended_agents: List[AgentTypeEnum] = Field(default_factory=list, description="推荐的智能体类型")


class CollaborationStrategyResponse(BaseModel):
    """协作策略响应"""
    name: str = Field(..., description="策略名称")
    complexity_level: ComplexityLevelEnum = Field(..., description="复杂度等级")
    agents_needed: List[str] = Field(default_factory=list, description="需要的智能体列表")
    estimated_time: int = Field(..., description="预计处理时间（秒）")
    confidence: float = Field(..., description="策略置信度")
    adaptive_rules: List[str] = Field(default_factory=list, description="自适应规则")
    fallback_strategies: List[str] = Field(default_factory=list, description="备用策略")


class AgentResult(BaseModel):
    """单个智能体结果"""
    agent_type: str = Field(..., description="智能体类型")
    agent_name: str = Field(..., description="智能体名称")
    diagnosis: Optional[List[str]] = Field(None, description="诊断结果")
    treatment_plan: Optional[str] = Field(None, description="治疗方案")
    confidence_score: float = Field(..., description="置信度分数")
    reasoning: Optional[str] = Field(None, description="推理过程")
    status: str = Field(..., description="处理状态")


class ConsensusResultResponse(BaseModel):
    """共识结果响应"""
    consensus_level: float = Field(..., description="共识水平 (0-1)")
    final_diagnosis: Optional[Dict[str, Any]] = Field(None, description="最终诊断")
    final_treatment: Optional[Dict[str, Any]] = Field(None, description="最终治疗方案")
    supporting_agents: List[str] = Field(default_factory=list, description="支持的智能体")
    confidence_score: float = Field(..., description="置信度")
    method_used: str = Field(..., description="使用的共识方法")
    details: Optional[Dict[str, Any]] = Field(None, description="详细信息")


class AgentConsultationResponse(BaseModel):
    """智能体会诊响应"""
    query: str = Field(..., description="原始查询")
    complexity_analysis: ComplexityAnalysisResponse = Field(..., description="复杂度分析")
    collaboration_strategy: CollaborationStrategyResponse = Field(..., description="协作策略")
    agent_results: List[AgentResult] = Field(default_factory=list, description="各智能体结果")
    consensus_result: Optional[ConsensusResultResponse] = Field(None, description="共识结果")
    final_report: Optional[Dict[str, Any]] = Field(None, description="最终报告")
    processing_time: float = Field(..., description="处理时间（秒）")
    session_id: str = Field(..., description="会话ID")


class ReviewResultResponse(BaseModel):
    """审查结果响应"""
    review_id: str = Field(..., description="审查ID")
    risk_level: str = Field(..., description="风险等级")
    quality_score: float = Field(..., description="质量分数")
    recommendations: List[str] = Field(default_factory=list, description="改进建议")
    concerns: List[str] = Field(default_factory=list, description="注意事项")
    verification_details: Optional[Dict[str, Any]] = Field(None, description="验证详情")
    status: str = Field(..., description="审查状态")


class MedicalQueryResponse(BaseModel):
    """完整医疗查询响应"""
    session_id: str = Field(..., description="会话ID")
    query: str = Field(..., description="原始查询")
    complexity_analysis: ComplexityAnalysisResponse = Field(..., description="复杂度分析")
    collaboration_strategy: Optional[CollaborationStrategyResponse] = Field(None, description="协作策略")
    agent_results: Optional[List[AgentResult]] = Field(None, description="智能体结果")
    consensus_result: Optional[ConsensusResultResponse] = Field(None, description="共识结果")
    final_report: Optional[Dict[str, Any]] = Field(None, description="最终报告")
    review_result: Optional[ReviewResultResponse] = Field(None, description="审查结果")
    processing_time: float = Field(..., description="处理时间")
    timestamp: str = Field(..., description="时间戳")


class AgentStatusResponse(BaseModel):
    """智能体状态响应"""
    agent_type: AgentTypeEnum = Field(..., description="智能体类型")
    agent_name: str = Field(..., description="智能体名称")
    status: str = Field(..., description="状态")
    last_active: Optional[str] = Field(None, description="最后活跃时间")
    total_consultations: int = Field(..., description="总会诊次数")
    success_rate: float = Field(..., description="成功率")


class SystemStatusResponse(BaseModel):
    """系统状态响应"""
    status: str = Field(..., description="系统状态")
    total_agents: int = Field(..., description="智能体总数")
    active_agents: int = Field(..., description="活跃智能体数")
    total_decisions: int = Field(..., description="总决策数")
    successful_consensus: int = Field(..., description="成功共识数")
    consensus_rate: float = Field(..., description="共识成功率")
    system_load: float = Field(..., description="系统负载")
    uptime: str = Field(..., description="运行时间")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    mdagents_available: bool = Field(..., description="mdagents是否可用")
    version: str = Field(..., description="版本号")


# ==================== 流式响应模型 ====================

class StreamEventType(str, Enum):
    """流式事件类型"""
    START = "start"
    COMPLEXITY_ANALYSIS = "complexity_analysis"
    STRATEGY_ALLOCATION = "strategy_allocation"
    AGENT_START = "agent_start"
    AGENT_PROGRESS = "agent_progress"
    AGENT_COMPLETE = "agent_complete"
    CONSENSUS = "consensus"
    REVIEW = "review"
    COMPLETE = "complete"
    ERROR = "error"


class StreamEvent(BaseModel):
    """流式事件"""
    event: StreamEventType = Field(..., description="事件类型")
    data: Dict[str, Any] = Field(default_factory=dict, description="事件数据")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="时间戳")


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误信息")
    details: Optional[Dict[str, Any]] = Field(None, description="详细信息")
    request_id: Optional[str] = Field(None, description="请求ID")
