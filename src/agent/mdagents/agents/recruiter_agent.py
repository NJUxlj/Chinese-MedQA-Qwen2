"""
招募智能体 - 负责组建专家团队
"""

from typing import Dict, List, Any, Optional
import json

from .base_agent import BaseAgent
from ..config import AgentConfig, AgentType, ComplexityLevel
from ..core.collaboration_allocator import CollaborationAllocator

class RecruiterAgent(BaseAgent):
    """招募智能体"""
    
    def __init__(self, config: AgentConfig, api_model=None, llm_config=None):
        """
        初始化招募智能体
        
        Args:
            config: 智能体配置
            api_model: API模型实例
            llm_config: LLM配置实例
        """
        super().__init__(config, api_model, llm_config)
        
        # 协作分配器
        self.collaboration_allocator = CollaborationAllocator()
        
        # 专家库
        self.expert_registry = {
            "心血管科": ["心脏内科专家", "心脏外科专家", "心电图专家"],
            "神经科": ["神经内科专家", "神经外科专家", "脑血管专家"],
            "呼吸科": ["呼吸内科专家", "胸外科专家", "重症医学专家"],
            "消化科": ["消化内科专家", "肝胆外科专家", "胃肠外科专家"],
            "内分泌科": ["糖尿病专家", "甲状腺专家", "内分泌外科专家"],
            "肿瘤科": ["肿瘤内科专家", "肿瘤外科专家", "放疗科专家"],
            "肾病科": ["肾内科专家", "泌尿外科专家", "透析专家"],
            "骨科": ["骨外科专家", "脊柱外科专家", "关节外科专家"]
        }
        
        # 团队模板
        self.team_templates = {
            "simple": {
                "agents": [AgentType.PCC],
                "description": "简单团队：单专家处理"
            },
            "mdt": {
                "agents": [AgentType.PCC, AgentType.SPECIALIST, AgentType.REVIEWER],
                "description": "多学科团队：协作处理"
            },
            "ict": {
                "agents": [AgentType.RECRUITER, AgentType.PCC, AgentType.SPECIALIST, 
                          AgentType.SPECIALIST, AgentType.SPECIALIST, AgentType.REVIEWER, AgentType.INTEGRATOR],
                "description": "综合护理团队：分层处理"
            }
        }
    
    def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        处理团队招募请求
        
        Args:
            query: 医疗查询
            context: 上下文信息
            
        Returns:
            招募结果
        """
        self.log_interaction("recruitment_started", f"开始为查询招募团队: {query[:100]}...")
        
        try:
            # 分析需求
            requirements = self._analyze_team_requirements(query, context)
            
            # 选择团队配置
            team_selection = self._select_team_configuration(requirements)
            
            # 招募专家
            expert_recruitment = self._recruit_experts(team_selection, query, context)
            
            # 资源分配
            resource_allocation = self._allocate_resources(team_selection, expert_recruitment)
            
            # 协作计划
            collaboration_plan = self._create_collaboration_plan(team_selection, context)
            
            result = {
                "agent_type": self.agent_type.value if hasattr(self.agent_type, 'value') else self.agent_type,
                "agent_name": self.name,
                "query": query,
                "requirements_analysis": requirements,
                "team_selection": team_selection,
                "expert_recruitment": expert_recruitment,
                "resource_allocation": resource_allocation,
                "collaboration_plan": collaboration_plan,
                "recruitment_summary": self._generate_recruitment_summary(team_selection, expert_recruitment),
                "status": "completed"
            }
            
            self.log_interaction("recruitment_completed", 
                               f"团队招募完成: {team_selection['team_type']}团队")
            return result
            
        except Exception as e:
            self.logger.error(f"招募过程出错: {e}")
            self.status = "error"
            return {
                "agent_type": self.agent_type.value if hasattr(self.agent_type, 'value') else self.agent_type,
                "agent_name": self.name,
                "query": query,
                "error": str(e),
                "status": "error"
            }
    
    def _analyze_team_requirements(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """分析团队需求"""
        requirements = {
            "complexity_level": ComplexityLevel.MODERATE,  # 默认中等复杂度
            "required_specialties": [],
            "urgency_level": "routine",
            "estimated_duration": "90分钟",
            "resource_requirements": "medium",
            "special_considerations": []
        }
        
        # 复杂度评估
        if any(word in query for word in ["简单", "常见", "轻微"]):
            requirements["complexity_level"] = ComplexityLevel.SIMPLE
            requirements["estimated_duration"] = "15分钟"
            requirements["resource_requirements"] = "low"
        
        elif any(word in query for word in ["复杂", "严重", "多种症状"]):
            requirements["complexity_level"] = ComplexityLevel.COMPLEX
            requirements["estimated_duration"] = "200分钟"
            requirements["resource_requirements"] = "high"
        
        # 专科需求分析
        specialty_mapping = {
            "心血管": "心血管科",
            "神经": "神经科", 
            "呼吸": "呼吸科",
            "消化": "消化科",
            "内分泌": "内分泌科",
            "肿瘤": "肿瘤科",
            "肾": "肾病科",
            "骨": "骨科"
        }
        
        for keyword, specialty in specialty_mapping.items():
            if keyword in query:
                requirements["required_specialties"].append(specialty)
        
        # 紧急程度评估
        emergency_symptoms = ["胸痛", "呼吸困难", "昏迷", "严重出血"]
        if any(symptom in query for symptom in emergency_symptoms):
            requirements["urgency_level"] = "emergency"
            requirements["estimated_duration"] = "立即处理"
        
        # 特殊考虑
        if context:
            if context.get("elderly", False):
                requirements["special_considerations"].append("老年患者需要额外关注")
            
            if context.get("multiple_chronic_diseases", False):
                requirements["special_considerations"].append("多慢性病需要综合管理")
                if len(requirements["required_specialties"]) < 2:
                    requirements["required_specialties"].append("全科")
        
        return requirements
    
    def _select_team_configuration(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """选择团队配置"""
        complexity_level = requirements["complexity_level"]
        specialty_count = len(requirements["required_specialties"])
        urgency = requirements["urgency_level"]
        
        team_config = {
            "team_type": "",
            "agents": [],
            "description": "",
            "justification": ""
        }
        
        # 根据复杂度选择团队类型
        if complexity_level == ComplexityLevel.SIMPLE:
            team_config["team_type"] = "simple"
            team_config["agents"] = [AgentType.PCC]
            team_config["description"] = "单专家快速处理团队"
            team_config["justification"] = "低复杂度问题适合单专家处理"
        
        elif complexity_level == ComplexityLevel.MODERATE:
            if specialty_count <= 1:
                team_config["team_type"] = "mdt_simple"
                team_config["agents"] = [AgentType.PCC, AgentType.SPECIALIST, AgentType.REVIEWER]
                team_config["description"] = "标准多学科团队"
                team_config["justification"] = "中等复杂度需要多学科协作"
            else:
                team_config["team_type"] = "mdt_extended"
                team_config["agents"] = [AgentType.PCC] + [AgentType.SPECIALIST] * min(specialty_count, 3) + [AgentType.REVIEWER]
                team_config["description"] = "扩展多学科团队"
                team_config["justification"] = f"涉及{specialty_count}个专科，需要扩展团队"
        
        else:  # HIGH complexity
            team_config["team_type"] = "ict"
            base_agents = [AgentType.RECRUITER, AgentType.PCC]
            specialist_count = min(max(specialty_count, 3), 5)
            base_agents.extend([AgentType.SPECIALIST] * specialist_count)
            base_agents.extend([AgentType.REVIEWER, AgentType.INTEGRATOR])
            
            team_config["agents"] = base_agents
            team_config["description"] = "综合护理团队"
            team_config["justification"] = "高复杂度问题需要综合护理团队处理"
        
        # 紧急情况调整
        if urgency == "emergency":
            team_config["agents"] = [agent for agent in team_config["agents"] if agent != AgentType.REVIEWER]
            team_config["description"] += "（紧急版本）"
        
        return team_config
    
    def _recruit_experts(self, team_selection: Dict[str, Any], 
                        query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """招募专家"""
        recruitment = {
            "recruited_experts": [],
            "expert_details": {},
            "availability_status": {},
            "special_assignments": {}
        }
        
        agents = team_selection["agents"]
        
        for agent_type in agents:
            if agent_type == AgentType.PCC:
                expert_info = {
                    "type": "pcc",
                    "name": "初级保健临床医生",
                    "specialty": "全科",
                    "expertise": ["常见病诊断", "基础治疗", "健康咨询"],
                    "availability": "available",
                    "estimated_response_time": "5分钟"
                }
            
            elif agent_type == AgentType.SPECIALIST:
                # 根据查询内容确定专科类型
                specialty = self._determine_specialty_from_query(query)
                expert_info = {
                    "type": "specialist",
                    "name": f"{specialty}专家",
                    "specialty": specialty,
                    "expertise": self.expert_registry.get(specialty, ["专业诊断", "专科治疗"]),
                    "availability": "available",
                    "estimated_response_time": "10分钟"
                }
            
            elif agent_type == AgentType.REVIEWER:
                expert_info = {
                    "type": "reviewer",
                    "name": "医疗质量审查员",
                    "specialty": "医疗质量",
                    "expertise": ["质量控制", "风险评估", "决策审查"],
                    "availability": "available",
                    "estimated_response_time": "8分钟"
                }
            
            elif agent_type == AgentType.INTEGRATOR:
                expert_info = {
                    "type": "integrator",
                    "name": "医疗决策整合师",
                    "specialty": "决策整合",
                    "expertise": ["意见整合", "决策综合", "最终输出"],
                    "availability": "available",
                    "estimated_response_time": "15分钟"
                }
            
            elif agent_type == AgentType.RECRUITER:
                expert_info = {
                    "type": "recruiter",
                    "name": "团队协调员",
                    "specialty": "团队协调",
                    "expertise": ["团队组建", "资源协调", "流程管理"],
                    "availability": "available",
                    "estimated_response_time": "3分钟"
                }
            
            else:
                expert_info = {
                    "type": agent_type.value,
                    "name": f"{agent_type.value}专家",
                    "specialty": "通用",
                    "expertise": ["专业服务"],
                    "availability": "available",
                    "estimated_response_time": "10分钟"
                }
            
            recruitment["recruited_experts"].append(expert_info)
            recruitment["expert_details"][agent_type.value] = expert_info
            recruitment["availability_status"][agent_type.value] = "available"
        
        return recruitment
    
    def _determine_specialty_from_query(self, query: str) -> str:
        """从查询中确定专科"""
        specialty_keywords = {
            "心血管科": ["心脏", "心律", "血压", "胸痛"],
            "神经科": ["脑", "神经", "头痛", "癫痫"],
            "呼吸科": ["肺", "呼吸", "咳嗽", "气短"],
            "消化科": ["胃", "肠", "腹痛", "消化"],
            "内分泌科": ["糖尿病", "甲状腺", "激素"],
            "肿瘤科": ["肿瘤", "癌症", "肿块"],
            "肾病科": ["肾", "尿", "透析"],
            "骨科": ["关节", "骨", "骨折"]
        }
        
        for specialty, keywords in specialty_keywords.items():
            if any(keyword in query for keyword in keywords):
                return specialty
        
        return "全科"  # 默认专科
    
    def _allocate_resources(self, team_selection: Dict[str, Any], 
                           expert_recruitment: Dict[str, Any]) -> Dict[str, Any]:
        """分配资源"""
        agent_count = len(team_selection["agents"])
        
        resource_allocation = {
            "compute_resources": {},
            "time_allocation": {},
            "priority_levels": {},
            "cost_estimation": {}
        }
        
        # 计算资源分配
        if agent_count <= 2:
            resource_allocation["compute_resources"] = {
                "cpu_cores": 2,
                "memory_gb": 4,
                "storage_gb": 10
            }
        elif agent_count <= 5:
            resource_allocation["compute_resources"] = {
                "cpu_cores": 4,
                "memory_gb": 8,
                "storage_gb": 20
            }
        else:
            resource_allocation["compute_resources"] = {
                "cpu_cores": 8,
                "memory_gb": 16,
                "storage_gb": 50
            }
        
        # 时间分配
        total_time = 0
        for expert in expert_recruitment["recruited_experts"]:
            time_str = expert["estimated_response_time"]
            if "分钟" in time_str:
                time_minutes = int(time_str.replace("分钟", ""))
                total_time += time_minutes
        
        resource_allocation["time_allocation"] = {
            "total_estimated_time": f"{total_time}分钟",
            "parallel_processing": agent_count > 2,
            "bottleneck_agent": "integrator" if agent_count > 3 else None
        }
        
        # 优先级分配
        for i, expert in enumerate(expert_recruitment["recruited_experts"]):
            agent_type = expert["type"]
            if agent_type == "pcc":
                priority = "high"
            elif agent_type == "specialist":
                priority = "high"
            elif agent_type == "reviewer":
                priority = "medium"
            else:
                priority = "medium"
            
            resource_allocation["priority_levels"][agent_type] = priority
        
        # 成本估算
        cost_per_agent_per_hour = 10  # 假设成本
        estimated_hours = total_time / 60
        total_cost = agent_count * cost_per_agent_per_hour * estimated_hours
        
        resource_allocation["cost_estimation"] = {
            "cost_per_agent_hour": cost_per_agent_per_hour,
            "estimated_hours": round(estimated_hours, 2),
            "total_estimated_cost": round(total_cost, 2),
            "cost_category": "medium" if total_cost < 100 else "high"
        }
        
        return resource_allocation
    
    def _create_collaboration_plan(self, team_selection: Dict[str, Any], 
                                  context: Dict[str, Any] = None) -> Dict[str, Any]:
        """创建协作计划"""
        collaboration_plan = {
            "collaboration_pattern": "",
            "communication_flow": [],
            "decision_making_process": "",
            "quality_control_steps": [],
            "contingency_plans": []
        }
        
        team_type = team_selection["team_type"]
        
        if team_type == "simple":
            collaboration_plan["collaboration_pattern"] = "sequential"
            collaboration_plan["communication_flow"] = ["pcc -> final_output"]
            collaboration_plan["decision_making_process"] = "direct_decision"
            collaboration_plan["quality_control_steps"] = ["basic_validation"]
        
        elif "mdt" in team_type:
            collaboration_plan["collaboration_pattern"] = "parallel_group"
            collaboration_plan["communication_flow"] = ["pcc -> specialists -> reviewer -> consensus"]
            collaboration_plan["decision_making_process"] = "group_voting"
            collaboration_plan["quality_control_steps"] = ["peer_review", "consensus_check"]
        
        elif team_type == "ict":
            collaboration_plan["collaboration_pattern"] = "hierarchical_parallel"
            collaboration_plan["communication_flow"] = [
                "recruiter -> team_coordination",
                "pcc -> initial_assessment", 
                "specialists -> parallel_assessment",
                "reviewer -> quality_check",
                "integrator -> final_synthesis"
            ]
            collaboration_plan["decision_making_process"] = "staged_consensus"
            collaboration_plan["quality_control_steps"] = [
                "individual_assessment",
                "peer_review", 
                "consensus_building",
                "final_review"
            ]
        
        # 应急计划
        collaboration_plan["contingency_plans"] = [
            "如果专家不可用，使用备用专家",
            "如果协作失败，升级为更高复杂度处理",
            "如果决策分歧，进行额外讨论轮次"
        ]
        
        return collaboration_plan
    
    def _generate_recruitment_summary(self, team_selection: Dict[str, Any], 
                                     expert_recruitment: Dict[str, Any]) -> Dict[str, Any]:
        """生成招募总结"""
        return {
            "team_type": team_selection["team_type"],
            "team_size": len(team_selection["agents"]),
            "recruited_experts": [expert["name"] for expert in expert_recruitment["recruited_experts"]],
            "specialties_covered": list(set(expert["specialty"] for expert in expert_recruitment["recruited_experts"])),
            "ready_for_deployment": True,
            "estimated_completion_time": "90分钟",
            "quality_assurance_level": "high" if len(team_selection["agents"]) > 3 else "medium"
        }
    
    def get_recruitment_capabilities(self) -> Dict[str, Any]:
        """获取招募能力"""
        return {
            "supported_team_types": list(self.team_templates.keys()),
            "expert_specialties": list(self.expert_registry.keys()),
            "recruitment_speed": "快速（3-5分钟）",
            "scalability": "支持大规模团队组建"
        }