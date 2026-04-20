"""
MDAgents服务
负责医疗多智能体系统的业务逻辑处理
"""

import asyncio
import logging
import threading
import time
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from agent.mdagents.config import (
    ComplexityLevel, AgentType, CollaborationPattern, 
    AgentConfig, ConfigManager
)
from agent.mdagents.core.main_controller import MainController
from api.schemas.mdagents import (
    MedicalQueryRequest, MedicalQueryResponse,
    ComplexityAnalysisRequest, ComplexityAnalysisResponse,
    CollaborationStrategyRequest, CollaborationStrategyResponse,
    AgentConsultationRequest, AgentConsultationResponse,
    ReviewRequest, ReviewResultResponse,
    AgentStatusResponse, SystemStatusResponse,
    ComplexityFeaturesResponse, AgentResult,
    ConsensusResultResponse, HealthResponse,
    ComplexityLevelEnum, AgentTypeEnum,
    CollaborationPatternEnum, ConsensusMethodEnum,
    ErrorResponse
)

logger = logging.getLogger(__name__)


class MDAgentsService:
    """MDAgents服务类"""
    
    def __init__(self):
        """初始化MDAgents服务"""
        self._initialized = False
        self._init_lock = threading.Lock()
        
        self.controller: Optional[MainController] = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self._stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_processing_time": 0.0,
            "start_time": None
        }
        
        logger.info("MDAgents服务实例已创建")
    
    def initialize(self, llm_config=None) -> None:
        """初始化MDAgents服务
        
        Args:
            llm_config: LLM配置
        """
        if self._initialized:
            return
        
        with self._init_lock:
            if self._initialized:
                return
            
            logger.info("初始化MDAgents服务...")
            
            try:
                self.controller = MainController(llm_config=llm_config)
                self._initialized = True
                self._stats["start_time"] = datetime.now().isoformat()
                
                logger.info("MDAgents服务初始化成功")
                
            except Exception as e:
                logger.error(f"MDAgents服务初始化失败: {e}")
                raise
    
    def is_available(self) -> bool:
        """检查服务是否可用"""
        return self._initialized and self.controller is not None
    
    def _ensure_initialized(self):
        """确保服务已初始化"""
        if not self.is_available():
            raise RuntimeError("MDAgents服务未初始化，请先调用initialize()")
    
    async def process_medical_query(self, request: MedicalQueryRequest) -> MedicalQueryResponse:
        """处理医疗查询
        
        Args:
            request: 医疗查询请求
            
        Returns:
            医疗查询响应
        """
        self._ensure_initialized()
        
        start_time = time.time()
        
        try:
            self._stats["total_queries"] += 1
            
            patient_info = request.patient_context or {}
            query = request.query
            
            result = await asyncio.to_thread(
                self.controller.process_medical_query, query, patient_info
            )
            
            processing_time = time.time() - start_time
            self._stats["total_processing_time"] += processing_time
            
            if result.get("status") == "success":
                self._stats["successful_queries"] += 1
                
                complexity_result = result.get("complexity_analysis", {})
                complexity_level = complexity_result.get("complexity_level", "simple")
                if isinstance(complexity_level, str):
                    pass
                else:
                    complexity_level = complexity_level.value if hasattr(complexity_level, 'value') else str(complexity_level)
                
                collaboration_plan = result.get("collaboration_plan", {})
                agents_needed = collaboration_plan.get("agents_needed", [])
                
                agent_results = []
                expert_diagnoses = result.get("consensus_result", {}).get("details", {}).get("expert_opinions", [])
                for i, diagnosis in enumerate(expert_diagnoses):
                    agent_results.append(AgentResult(
                        agent_type=diagnosis.get("agent_type", "unknown"),
                        agent_name=diagnosis.get("agent_name", f"专家_{i}"),
                        diagnosis=[diagnosis.get("diagnosis", "")],
                        confidence_score=diagnosis.get("confidence", 0.0),
                        reasoning=diagnosis.get("reasoning", ""),
                        status="completed"
                    ))
                
                consensus_result_data = result.get("consensus_result", {})
                if consensus_result_data:
                    consensus_result = ConsensusResultResponse(
                        consensus_level=consensus_result_data.get("consensus_level", 0.0),
                        final_diagnosis=consensus_result_data.get("final_decision", {}),
                        final_treatment=consensus_result_data.get("final_treatment", {}),
                        supporting_agents=consensus_result_data.get("supporting_agents", []),
                        confidence_score=consensus_result_data.get("confidence_score", 0.0),
                        method_used=consensus_result_data.get("method_used", "simple_majority"),
                        details=consensus_result_data.get("details", {})
                    )
                else:
                    consensus_result = None
                
                complexity_features = complexity_result.get("features", {})
                if complexity_features:
                    features_response = ComplexityFeaturesResponse(
                        text_length=complexity_features.get("text_length", 0),
                        symptom_count=complexity_features.get("symptom_count", 0),
                        medical_terms_count=complexity_features.get("medical_terms_count", 0),
                        specialty_count=complexity_features.get("specialty_count", 0),
                        chronic_conditions_mentioned=complexity_features.get("chronic_conditions_mentioned", False),
                        emergency_keywords=complexity_features.get("emergency_keywords", []),
                        diagnostic_complexity=complexity_features.get("diagnostic_complexity", 0.0),
                        treatment_complexity=complexity_features.get("treatment_complexity", 0.0)
                    )
                else:
                    features_response = None
                
                complexity_response = ComplexityAnalysisResponse(
                    query=query,
                    complexity_score=complexity_result.get("complexity_score", 0.0),
                    complexity_level=ComplexityLevelEnum(complexity_level),
                    features=features_response,
                    confidence=complexity_result.get("confidence", 0.0),
                    recommended_agents=[AgentTypeEnum(a) for a in agents_needed]
                )
                
                final_report = result.get("final_report", {})
                
                response = MedicalQueryResponse(
                    session_id=result.get("session_id", ""),
                    query=query,
                    complexity_analysis=complexity_response,
                    collaboration_strategy=None,
                    agent_results=agent_results if agent_results else None,
                    consensus_result=consensus_result,
                    final_report=final_report,
                    processing_time=processing_time,
                    timestamp=datetime.now().isoformat()
                )
                
                return response
            else:
                self._stats["failed_queries"] += 1
                
                return MedicalQueryResponse(
                    session_id=result.get("session_id", ""),
                    query=query,
                    complexity_analysis=ComplexityAnalysisResponse(
                        query=query,
                        complexity_score=0.0,
                        complexity_level=ComplexityLevelEnum.SIMPLE,
                        confidence=0.0,
                        recommended_agents=[]
                    ),
                    processing_time=processing_time,
                    timestamp=datetime.now().isoformat()
                )
                
        except Exception as e:
            logger.error(f"处理医疗查询失败: {e}")
            self._stats["failed_queries"] += 1
            
            raise
    
    async def analyze_complexity(self, request: ComplexityAnalysisRequest) -> ComplexityAnalysisResponse:
        """分析查询复杂度
        
        Args:
            request: 复杂度分析请求
            
        Returns:
            复杂度分析响应
        """
        self._ensure_initialized()
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: self.controller.complexity_analyzer.analyze_complexity(
                    request.query, None
                )
            )
            
            result_dict = result.to_dict() if hasattr(result, 'to_dict') else result
            
            complexity_level = result_dict.get("complexity_level", ComplexityLevel.SIMPLE)
            if hasattr(complexity_level, 'value'):
                complexity_level = complexity_level.value
            
            features = result_dict.get("features", {})
            if features and request.include_features:
                features_response = ComplexityFeaturesResponse(
                    text_length=features.get("text_length", 0),
                    symptom_count=features.get("symptom_count", 0),
                    medical_terms_count=features.get("medical_terms_count", 0),
                    specialty_count=features.get("specialty_count", 0),
                    chronic_conditions_mentioned=features.get("chronic_conditions_mentioned", False),
                    emergency_keywords=features.get("emergency_keywords", []),
                    diagnostic_complexity=features.get("diagnostic_complexity", 0.0),
                    treatment_complexity=features.get("treatment_complexity", 0.0)
                )
            else:
                features_response = None
            
            recommended_agents = result_dict.get("recommended_agents", [])
            if isinstance(recommended_agents[0], str) if recommended_agents else False:
                recommended_agents = [AgentTypeEnum(a) for a in recommended_agents]
            elif recommended_agents and hasattr(recommended_agents[0], 'value'):
                recommended_agents = [AgentTypeEnum(a.value) for a in recommended_agents]
            
            return ComplexityAnalysisResponse(
                query=request.query,
                complexity_score=result_dict.get("complexity_score", 0.0),
                complexity_level=ComplexityLevelEnum(complexity_level),
                features=features_response,
                confidence=result_dict.get("confidence", 0.0),
                recommended_agents=recommended_agents
            )
            
        except Exception as e:
            logger.error(f"复杂度分析失败: {e}")
            raise
    
    async def get_collaboration_strategy(
        self, request: CollaborationStrategyRequest
    ) -> CollaborationStrategyResponse:
        """获取协作策略
        
        Args:
            request: 协作策略请求
            
        Returns:
            协作策略响应
        """
        self._ensure_initialized()
        
        try:
            complexity_level = request.complexity_level.value if hasattr(
                request.complexity_level, 'value'
            ) else request.complexity_level
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: self.controller.collaboration_allocator.allocate_collaboration(
                    complexity_level, request.query
                )
            )
            
            result_dict = result.to_dict() if hasattr(result, 'to_dict') else result
            
            pattern = result_dict.get("pattern", CollaborationPattern.SINGLE_AGENT)
            if hasattr(pattern, 'value'):
                pattern = pattern.value
            
            agents_needed = result_dict.get("agents_needed", [])
            agents_needed_str = [a.value if hasattr(a, 'value') else str(a) for a in agents_needed]
            
            return CollaborationStrategyResponse(
                name=result_dict.get("name", "默认策略"),
                complexity_level=request.complexity_level,
                agents_needed=agents_needed_str,
                estimated_time=result_dict.get("estimated_time", 300),
                confidence=result_dict.get("confidence", 0.8),
                adaptive_rules=result_dict.get("adaptive_rules", []),
                fallback_strategies=result_dict.get("fallback_strategies", [])
            )
            
        except Exception as e:
            logger.error(f"获取协作策略失败: {e}")
            raise
    
    async def process_agent_consultation(
        self, request: AgentConsultationRequest
    ) -> AgentConsultationResponse:
        """处理智能体会诊
        
        Args:
            request: 智能体会诊请求
            
        Returns:
            智能体会诊响应
        """
        self._ensure_initialized()
        
        start_time = time.time()
        
        try:
            complexity_result = await self.analyze_complexity(
                ComplexityAnalysisRequest(query=request.query, include_features=True)
            )
            
            collaboration_result = await self.get_collaboration_strategy(
                CollaborationStrategyRequest(
                    complexity_level=complexity_result.complexity_level,
                    query=request.query
                )
            )
            
            diagnosis_result = await asyncio.to_thread(
                self.controller._execute_collaborative_diagnosis,
                request.query, None,
                [self.controller.agents.get(a.value) for a in request.agent_types],
                complexity_result.complexity_level.value
            )

            consensus_result = await asyncio.to_thread(
                self.controller._achieve_consensus,
                diagnosis_result, complexity_result.complexity_level.value
            )
            
            agent_results = []
            for expert_diagnosis in diagnosis_result.get("expert_diagnoses", []):
                diagnosis = expert_diagnosis.get("diagnosis", {})
                agent_results.append(AgentResult(
                    agent_type=expert_diagnosis.get("agent_type", "unknown"),
                    agent_name=expert_diagnosis.get("agent_name", "专家"),
                    diagnosis=[diagnosis.get("primary_diagnosis", "")],
                    treatment_plan=diagnosis.get("recommendations", [""]),
                    confidence_score=diagnosis.get("confidence", 0.0),
                    reasoning=diagnosis.get("reasoning", ""),
                    status="completed"
                ))
            
            consensus_response = ConsensusResultResponse(
                consensus_level=consensus_result.consensus_level if hasattr(consensus_result, 'consensus_level') else 0.0,
                final_diagnosis=consensus_result.final_decision if hasattr(consensus_result, 'final_decision') else {},
                final_treatment=consensus_result.final_treatment if hasattr(consensus_result, 'final_treatment') else {},
                supporting_agents=consensus_result.supporting_agents if hasattr(consensus_result, 'supporting_agents') else [],
                confidence_score=consensus_result.confidence_score if hasattr(consensus_result, 'confidence_score') else 0.0,
                method_used=request.consensus_method.value if request.consensus_method else "simple_majority"
            )
            
            processing_time = time.time() - start_time
            
            session_id = str(uuid.uuid4())
            
            return AgentConsultationResponse(
                query=request.query,
                complexity_analysis=complexity_result,
                collaboration_strategy=collaboration_result,
                agent_results=agent_results,
                consensus_result=consensus_response,
                final_report=None,
                processing_time=processing_time,
                session_id=session_id
            )
            
        except Exception as e:
            logger.error(f"智能体会诊处理失败: {e}")
            raise
    
    async def process_review(self, request: ReviewRequest) -> ReviewResultResponse:
        """处理医疗建议审查
        
        Args:
            request: 审查请求
            
        Returns:
            审查结果响应
        """
        self._ensure_initialized()
        
        try:
            reviewer = self.controller.agents.get("reviewer")
            if not reviewer:
                raise RuntimeError("审查员智能体不可用")
            
            context = {
                "agent_recommendations": request.agent_recommendations or [],
                "complexity_level": request.complexity_level.value if request.complexity_level else "moderate"
            }
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: reviewer.process_query(request.medical_advice, context)
            )

            risk_level = result.get("risk_level", "medium")
            if risk_level not in ["low", "medium", "high", "critical"]:
                risk_level = "medium"
            
            return ReviewResultResponse(
                review_id=str(uuid.uuid4()),
                risk_level=risk_level,
                quality_score=result.get("quality_score", 0.7),
                recommendations=result.get("recommendations", []),
                concerns=result.get("concerns", []),
                verification_details=result.get("details", {}),
                status="completed"
            )
            
        except Exception as e:
            logger.error(f"医疗建议审查失败: {e}")
            raise
    
    def get_agent_status(self, agent_type: str) -> AgentStatusResponse:
        """获取智能体状态
        
        Args:
            agent_type: 智能体类型
            
        Returns:
            智能体状态响应
        """
        self._ensure_initialized()
        
        try:
            agent = self.controller.agents.get(agent_type)
            if not agent:
                raise ValueError(f"未找到类型为 {agent_type} 的智能体")
            
            return AgentStatusResponse(
                agent_type=AgentTypeEnum(agent_type),
                agent_name=agent.name,
                status=agent.status,
                last_active=None,
                total_consultations=0,
                success_rate=0.9
            )
            
        except Exception as e:
            logger.error(f"获取智能体状态失败: {e}")
            raise
    
    def get_all_agent_status(self) -> List[AgentStatusResponse]:
        """获取所有智能体状态"""
        self._ensure_initialized()
        
        agents_status = []
        for agent_type, agent in self.controller.agents.items():
            agents_status.append(AgentStatusResponse(
                agent_type=AgentTypeEnum(agent_type),
                agent_name=agent.name,
                status=agent.status,
                last_active=None,
                total_consultations=0,
                success_rate=0.9
            ))
        
        return agents_status
    
    def get_system_status(self) -> SystemStatusResponse:
        """获取系统状态"""
        self._ensure_initialized()
        
        try:
            status = self.controller.get_system_status()
            
            success_rate = (
                self._stats["successful_queries"] / 
                max(self._stats["total_queries"], 1)
            )
            
            uptime_seconds = 0
            if self._stats["start_time"]:
                start = datetime.fromisoformat(self._stats["start_time"])
                uptime_seconds = (datetime.now() - start).total_seconds()
            
            hours = int(uptime_seconds // 3600)
            minutes = int((uptime_seconds % 3600) // 60)
            uptime_str = f"{hours}h {minutes}m"
            
            return SystemStatusResponse(
                status=status.get("system_status", "running"),
                total_agents=len(self.controller.agents),
                active_agents=len([a for a in self.controller.agents.values() if a.status == "active"]),
                total_decisions=status.get("total_decisions", 0),
                successful_consensus=status.get("successful_consensus", 0),
                consensus_rate=success_rate,
                system_load=0.5,
                uptime=uptime_str
            )
            
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            raise
    
    def health_check(self) -> HealthResponse:
        """健康检查"""
        return HealthResponse(
            status="healthy" if self.is_available() else "unhealthy",
            mdagents_available=self.is_available(),
            version="1.0.0"
        )
    
    async def shutdown(self):
        """关闭服务"""
        if self.controller:
            await self.controller.shutdown()
        
        self.executor.shutdown(wait=True)
        self._initialized = False
        
        logger.info("MDAgents服务已关闭")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        return {
            "total_queries": self._stats["total_queries"],
            "successful_queries": self._stats["successful_queries"],
            "failed_queries": self._stats["failed_queries"],
            "success_rate": (
                self._stats["successful_queries"] / 
                max(self._stats["total_queries"], 1)
            ),
            "average_processing_time": (
                self._stats["total_processing_time"] / 
                max(self._stats["total_queries"], 1)
            ),
            "uptime_seconds": (
                (datetime.now() - datetime.fromisoformat(self._stats["start_time"])).total_seconds()
                if self._stats["start_time"] else 0
            )
        }


_mdagents_service = None
_service_lock = threading.Lock()


def get_mdagents_service() -> MDAgentsService:
    """获取MDAgents服务单例
    
    Returns:
        MDAgents服务实例
    """
    global _mdagents_service
    
    if _mdagents_service is None:
        with _service_lock:
            if _mdagents_service is None:
                _mdagents_service = MDAgentsService()
                _mdagents_service.initialize()
    
    return _mdagents_service


def reset_mdagents_service():
    """重置MDAgents服务单例（用于测试）"""
    global _mdagents_service
    
    if _mdagents_service is not None:
        import asyncio
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_mdagents_service.shutdown())
        except Exception:
            pass
        finally:
            loop.close()
    
    _mdagents_service = None
