"""
主控制器 - MDAgents系统的核心协调器
负责管理整个多智能体医疗决策系统的工作流程
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import uuid

from .complexity_analyzer import ComplexityAnalyzer
from .collaboration_allocator import CollaborationAllocator
from .consensus_mechanism import ConsensusMechanism, ConsensusResult
from ..config import AgentConfig, ComplexityLevel, CollaborationPattern, DecisionType
from ..agents.base_agent import BaseAgent
from ..agents.pcc_agent import PCCAgent
from ..agents.specialist_agent import SpecialistAgent
from ..agents.moderator_agent import ModeratorAgent
from ..agents.recruiter_agent import RecruiterAgent
from ..agents.reviewer_agent import ReviewerAgent
from ..agents.integrator_agent import IntegratorAgent
from ..tools.logger import setup_logger


class MainController:
    """MDAgents系统主控制器"""
    
    def __init__(self, llm_config=None):
        """初始化主控制器
        
        Args:
            llm_config: LLM配置实例
        """
        self.logger = setup_logger(self.__class__.__name__)
        self.llm_config = llm_config
        
        # 初始化核心组件
        self.complexity_analyzer = ComplexityAnalyzer()
        self.collaboration_allocator = CollaborationAllocator()
        self.consensus_mechanism = ConsensusMechanism()
        
        # 智能体管理
        self.agents: Dict[str, BaseAgent] = {}
        self.active_collaboration = None
        
        # 系统状态
        self.system_status = "initialized"
        self.total_decisions = 0
        self.successful_consensus = 0
        
        # 初始化默认智能体
        self._initialize_default_agents()
        
        self.logger.info("主控制器初始化完成")
    
    def _initialize_default_agents(self):
        """初始化默认智能体"""
        try:
            # 创建全科医生智能体
            pcc_config = AgentConfig(
                agent_type="pcc",
                name="全科医生_01",
                description="初级保健医生，负责初步诊断和分诊"
            )
            self.agents["pcc"] = PCCAgent(pcc_config)
            
            # 创建专科医生智能体（内科）
            specialist_config = AgentConfig(
                agent_type="specialist",
                name="内科专家_01",
                description="内科专家医生",
                specialty="内科"
            )
            self.agents["specialist"] = SpecialistAgent(specialist_config)
            
            # 创建主持人智能体
            moderator_config = AgentConfig(
                agent_type="moderator",
                name="主持人_01",
                description="负责协调多智能体协作和讨论"
            )
            self.agents["moderator"] = ModeratorAgent(moderator_config)
            
            # 创建招募者智能体
            recruiter_config = AgentConfig(
                agent_type="recruiter",
                name="招募者_01",
                description="负责根据复杂度招募合适的专家智能体"
            )
            self.agents["recruiter"] = RecruiterAgent(recruiter_config)
            
            # 创建审查员智能体
            reviewer_config = AgentConfig(
                agent_type="reviewer",
                name="审查员_01",
                description="负责质量审查和风险评估"
            )
            self.agents["reviewer"] = ReviewerAgent(reviewer_config)
            
            # 创建整合者智能体
            integrator_config = AgentConfig(
                agent_type="integrator",
                name="整合者_01",
                description="负责整合专家意见并形成最终决策"
            )
            self.agents["integrator"] = IntegratorAgent(integrator_config)
            
            self.logger.info("默认智能体初始化完成")
            
        except Exception as e:
            self.logger.error(f"智能体初始化失败: {e}")
            raise
    
    async def process_medical_query(self, query: str, patient_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理医疗查询的主要方法
        
        Args:
            query: 医疗查询文本
            patient_info: 患者信息（可选）
            
        Returns:
            处理结果字典
        """
        session_id = str(uuid.uuid4())
        self.logger.info(f"开始处理医疗查询 [会话ID: {session_id}]")
        
        try:
            self.system_status = "processing"
            
            # 步骤1: 复杂度分析
            complexity_result = await self._analyze_complexity(query, patient_info)
            complexity_level = complexity_result["complexity_level"]
            complexity_details = {
                "score": complexity_result["complexity_score"],
                "explanation": complexity_result["explanation"],
                "features": complexity_result["features"],
                "recommendation": complexity_result["recommendation"]
            }
            
            self.logger.info(f"复杂度分析完成: {complexity_level}")
            
            # 步骤2: 协作模式分配
            collaboration_plan = await self._allocate_collaboration(complexity_level, query)
            self.active_collaboration = collaboration_plan
            
            pattern = collaboration_plan['pattern'] if isinstance(collaboration_plan['pattern'], str) else collaboration_plan['pattern'].value
            self.logger.info(f"协作模式分配完成: {pattern}")
            
            # 步骤3: 招募专家智能体
            expert_agents = await self._recruit_experts(collaboration_plan, complexity_level)
            
            # 步骤4: 执行协作诊断
            diagnosis_result = await self._execute_collaborative_diagnosis(
                query, patient_info, expert_agents, complexity_level
            )
            
            # 步骤5: 达成共识
            consensus_result = await self._achieve_consensus(diagnosis_result, complexity_level)
            
            # 步骤6: 生成最终报告
            final_report = await self._generate_final_report(
                query, complexity_details, collaboration_plan, consensus_result
            )
            
            # 更新统计信息
            self.total_decisions += 1
            if consensus_result.success:
                self.successful_consensus += 1
            
            self.system_status = "completed"
            
            return {
                "session_id": session_id,
                "status": "success",
                "query": query,
                "complexity_analysis": complexity_result,
                "collaboration_plan": collaboration_plan,
                "expert_agents": [agent.name for agent in expert_agents],
                "consensus_result": consensus_result.to_dict(),
                "final_report": final_report,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"医疗查询处理失败: {e}")
            self.system_status = "error"
            
            return {
                "session_id": session_id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _analyze_complexity(self, query: str, patient_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """分析查询复杂度"""
        try:
            result = self.complexity_analyzer.analyze_complexity(query, patient_info)
            return result.to_dict() if hasattr(result, 'to_dict') else result
        except Exception as e:
            self.logger.error(f"复杂度分析失败: {e}")
            return {
                "complexity_level": ComplexityLevel.SIMPLE,
                "confidence": 0.5,
                "details": {"error": str(e)},
                "factors": {}
            }
    
    async def _allocate_collaboration(self, complexity_level: ComplexityLevel, query: str) -> Dict[str, Any]:
        """分配协作模式"""
        try:
            result = self.collaboration_allocator.allocate_collaboration(complexity_level, query)
            return result.to_dict() if hasattr(result, 'to_dict') else result
        except Exception as e:
            self.logger.error(f"协作模式分配失败: {e}")
            return {
                "pattern": CollaborationPattern.SINGLE_AGENT,
                "agents_needed": ["pcc"],
                "reasoning": f"默认分配，错误: {str(e)}",
                "estimated_time": 300,
                "confidence": 0.5
            }
    
    async def _recruit_experts(self, collaboration_plan: Dict[str, Any], complexity_level: ComplexityLevel) -> List[BaseAgent]:
        """招募专家智能体"""
        try:
            recruiter = self.agents.get("recruiter")
            if not recruiter:
                self.logger.warning("招募者智能体未找到，使用默认智能体")
                return [self.agents["pcc"]]
            
            # 根据协作计划招募合适的智能体
            agents_needed = collaboration_plan.get("agents_needed", ["pcc"])
            recruited_agents = []
            
            for agent_type in agents_needed:
                if agent_type in self.agents:
                    recruited_agents.append(self.agents[agent_type])
                else:
                    self.logger.warning(f"未找到类型为 {agent_type} 的智能体")
            
            # 如果没有招募到任何智能体，至少返回全科医生
            if not recruited_agents:
                recruited_agents = [self.agents["pcc"]]
            
            return recruited_agents
            
        except Exception as e:
            self.logger.error(f"专家招募失败: {e}")
            return [self.agents["pcc"]]
    
    async def _execute_collaborative_diagnosis(
        self, 
        query: str, 
        patient_info: Optional[Dict[str, Any]], 
        expert_agents: List[BaseAgent],
        complexity_level: ComplexityLevel
    ) -> Dict[str, Any]:
        """执行协作诊断"""
        try:
            self.logger.info(f"开始协作诊断，参与专家: {[agent.name for agent in expert_agents]}")
            
            # 并行执行各专家的诊断
            tasks = []
            for agent in expert_agents:
                task = asyncio.create_task(self._get_agent_diagnosis(agent, query, patient_info))
                tasks.append(task)
            
            # 等待所有专家完成诊断
            agent_diagnoses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理诊断结果
            valid_diagnoses = []
            for i, diagnosis in enumerate(agent_diagnoses):
                if isinstance(diagnosis, Exception):
                    self.logger.error(f"专家 {expert_agents[i].name} 诊断失败: {diagnosis}")
                else:
                    valid_diagnoses.append({
                        "agent_name": expert_agents[i].name,
                        "agent_type": expert_agents[i].agent_type,
                        "diagnosis": diagnosis
                    })
            
            return {
                "expert_diagnoses": valid_diagnoses,
                "total_experts": len(expert_agents),
                "successful_experts": len(valid_diagnoses),
                "complexity_level": complexity_level.value if hasattr(complexity_level, 'value') else complexity_level
            }
            
        except Exception as e:
            self.logger.error(f"协作诊断执行失败: {e}")
            return {
                "expert_diagnoses": [],
                "total_experts": len(expert_agents),
                "successful_experts": 0,
                "error": str(e)
            }
    
    async def _get_agent_diagnosis(
        self, 
        agent: BaseAgent, 
        query: str, 
        patient_info: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """获取单个智能体的诊断结果"""
        try:
            # 模拟智能体诊断过程
            # 实际实现中这里会调用智能体的具体诊断方法
            
            diagnosis_data = {
                "primary_diagnosis": f"{agent.name}的初步诊断",
                "confidence": 0.8,
                "reasoning": f"基于{agent.agent_type}的专业知识进行分析",
                "recommendations": ["建议进一步检查", "密切观察症状变化"],
                "risk_level": "中等",
                "urgency": "常规"
            }
            
            # 模拟处理时间
            await asyncio.sleep(0.1)
            
            return diagnosis_data
            
        except Exception as e:
            self.logger.error(f"智能体 {agent.name} 诊断过程失败: {e}")
            raise
    
    async def _achieve_consensus(
        self, 
        diagnosis_result: Dict[str, Any], 
        complexity_level: ComplexityLevel
    ) -> ConsensusResult:
        """达成共识"""
        try:
            # 准备专家意见
            expert_opinions = []
            for expert_diagnosis in diagnosis_result.get("expert_diagnoses", []):
                opinion = {
                    "agent_name": expert_diagnosis["agent_name"],
                    "agent_type": expert_diagnosis["agent_type"],
                    "diagnosis": expert_diagnosis["diagnosis"]["primary_diagnosis"],
                    "confidence": expert_diagnosis["diagnosis"]["confidence"],
                    "recommendations": expert_diagnosis["diagnosis"]["recommendations"],
                    "risk_level": expert_diagnosis["diagnosis"]["risk_level"]
                }
                expert_opinions.append(opinion)
            
            # 选择合适的共识算法
            if complexity_level == ComplexityLevel.SIMPLE:
                algorithm = "simple_majority"
            elif complexity_level == ComplexityLevel.MODERATE:
                algorithm = "weighted_voting"
            else:
                algorithm = "iterative_consensus"
            
            # 达成共识
            consensus_result = self.consensus_mechanism.achieve_consensus(
                expert_opinions=expert_opinions,
                algorithm=algorithm,
                decision_type="diagnosis"
            )
            
            return consensus_result
            
        except Exception as e:
            self.logger.error(f"共识达成失败: {e}")
            return ConsensusResult(
                consensus_level=0.0,
                final_decision={"error": "需要进一步诊断", "message": str(e)},
                supporting_agents=[],
                confidence_score=0.3,
                success=False
            )
    
    async def _generate_final_report(
        self,
        query: str,
        complexity_details: Dict[str, Any],
        collaboration_plan: Dict[str, Any],
        consensus_result: ConsensusResult
    ) -> Dict[str, Any]:
        """生成最终报告"""
        try:
            integrator = self.agents.get("integrator")
            if integrator:
                final_report = {
                    "summary": "基于多专家协作的医疗诊断报告",
                    "key_findings": consensus_result.final_decision,
                    "recommendations": consensus_result.details.get("recommendations", []) if isinstance(consensus_result.details, dict) else [],
                    "confidence_level": consensus_result.confidence_score,
                    "collaboration_summary": f"使用了{len(collaboration_plan.get('agents_needed', []))}位专家进行协作诊断",
                    "generated_at": datetime.now().isoformat()
                }
                
                return final_report
            else:
                return {
                    "summary": "医疗诊断报告",
                    "key_findings": consensus_result.final_decision,
                    "confidence_level": consensus_result.confidence_score,
                    "recommendations": ["建议咨询专业医生"],
                    "generated_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"报告生成失败: {e}")
            return {
                "summary": "诊断报告生成失败",
                "error": str(e),
                "generated_at": datetime.now().isoformat()
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "system_status": self.system_status,
            "total_decisions": self.total_decisions,
            "successful_consensus": self.successful_consensus,
            "success_rate": self.successful_consensus / max(self.total_decisions, 1),
            "active_agents": len(self.agents),
            "agent_types": list(self.agents.keys())
        }
    
    def add_custom_agent(self, agent: BaseAgent):
        """添加自定义智能体"""
        self.agents[agent.agent_type] = agent
        self.logger.info(f"添加自定义智能体: {agent.name}")
    
    def remove_agent(self, agent_type: str):
        """移除智能体"""
        if agent_type in self.agents:
            del self.agents[agent_type]
            self.logger.info(f"移除智能体类型: {agent_type}")
    
    async def shutdown(self):
        """关闭系统"""
        self.logger.info("正在关闭MDAgents系统...")
        self.system_status = "shutdown"
        
        # 清理资源
        for agent in self.agents.values():
            if hasattr(agent, 'cleanup'):
                try:
                    await agent.cleanup()
                except Exception as e:
                    self.logger.error(f"智能体 {agent.name} 清理失败: {e}")
        
        self.logger.info("MDAgents系统已关闭")