"""
调节器智能体 - 负责复杂度评估和任务路由
"""

from typing import Dict, List, Any, Optional
import json

from .base_agent import BaseAgent
from ..config import AgentConfig, AgentType, ComplexityLevel
from ..core.complexity_analyzer import ComplexityAnalyzer

class ModeratorAgent(BaseAgent):
    """调节器智能体"""
    
    def __init__(self, config: AgentConfig, api_model=None, llm_config=None):
        """
        初始化调节器
        
        Args:
            config: 智能体配置
            api_model: API模型实例
            llm_config: LLM配置实例
        """
        super().__init__(config, api_model, llm_config)
        
        # 复杂度分析器
        self.complexity_analyzer = ComplexityAnalyzer()
        
        # 调节规则
        self.moderation_rules = {
            "complexity_thresholds": {
                ComplexityLevel.SIMPLE: 0.3,
                ComplexityLevel.MODERATE: 0.6,
                ComplexityLevel.COMPLEX: 1.0
            },
            "routing_criteria": {
                "emergency_symptoms": ["胸痛", "呼吸困难", "昏迷", "严重出血"],
                "specialty_indicators": {
                    "心血管": "心血管科",
                    "神经": "神经科", 
                    "肿瘤": "肿瘤科",
                    "内分泌": "内分泌科"
                }
            }
        }
    
    def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        处理医疗查询的调节
        
        Args:
            query: 医疗查询
            context: 上下文信息
            
        Returns:
            调节结果
        """
        self.log_interaction("moderation_started", f"开始调节查询: {query[:100]}...")
        
        try:
            # 复杂度分析
            complexity_analysis = self.complexity_analyzer.analyze_complexity(query, context)
            
            # 生成调节决策
            moderation_decision = self._generate_moderation_decision(query, complexity_analysis, context)
            
            # 路由建议
            routing_plan = self._generate_routing_plan(query, complexity_analysis, context)
            
            # 风险评估
            risk_assessment = self._assess_risk_level(query, context)
            
            result = {
                "agent_type": self.agent_type.value,
                "agent_name": self.name,
                "query": query,
                "complexity_analysis": complexity_analysis,
                "moderation_decision": moderation_decision,
                "routing_plan": routing_plan,
                "risk_assessment": risk_assessment,
                "confidence_score": complexity_analysis.get("complexity_score", 0.5),
                "status": "completed"
            }
            
            self.log_interaction("moderation_completed", 
                               f"调节完成: {moderation_decision['recommended_approach']}")
            return result
            
        except Exception as e:
            self.logger.error(f"调节过程出错: {e}")
            self.status = "error"
            return {
                "agent_type": self.agent_type.value,
                "agent_name": self.name,
                "query": query,
                "error": str(e),
                "status": "error"
            }
    
    def _generate_moderation_decision(self, query: str, 
                                    complexity_analysis: Dict[str, Any], 
                                    context: Dict[str, Any] = None) -> Dict[str, Any]:
        """生成调节决策"""
        complexity_score = complexity_analysis.get("complexity_score", 0.5)
        complexity_level = ComplexityLevel(complexity_analysis.get("complexity_level", "medium"))
        
        # 紧急情况检测
        emergency_detected = self._detect_emergency_symptoms(query)
        
        decision = {
            "complexity_level": complexity_level.value,
            "complexity_score": complexity_score,
            "recommended_approach": "",
            "processing_strategy": "",
            "urgency_level": "routine",
            "special_handling": []
        }
        
        if emergency_detected:
            decision["urgency_level"] = "emergency"
            decision["recommended_approach"] = "立即急诊处理"
            decision["processing_strategy"] = "emergency_protocol"
            decision["special_handling"].append("绕过正常流程，直接急诊")
        else:
            # 根据复杂度等级决定处理方式
            if complexity_level == ComplexityLevel.SIMPLE:
                decision["recommended_approach"] = "单专家快速处理"
                decision["processing_strategy"] = "pcc_direct"
            elif complexity_level == ComplexityLevel.MODERATE:
                decision["recommended_approach"] = "多学科团队协作"
                decision["processing_strategy"] = "mdt_collaboration"
            else:  # HIGH
                decision["recommended_approach"] = "综合护理团队"
                decision["processing_strategy"] = "ict_hierarchical"
        
        # 根据置信度调整
        confidence = complexity_analysis.get("confidence_score", 0.5)
        if confidence < 0.4:
            decision["special_handling"].append("复杂度评估置信度低，需要人工确认")
        
        return decision
    
    def _generate_routing_plan(self, query: str, 
                             complexity_analysis: Dict[str, Any], 
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """生成路由计划"""
        complexity_level = ComplexityLevel(complexity_analysis.get("complexity_level", "medium"))
        
        # 识别需要的专科
        required_specialties = self._identify_specialties(query)
        
        routing_plan = {
            "primary_route": "",
            "specialty_routing": [],
            "fallback_routes": [],
            "resource_allocation": {}
        }
        
        # 根据复杂度制定路由策略
        if complexity_level == ComplexityLevel.SIMPLE:
            routing_plan["primary_route"] = "pcc_agent"
            routing_plan["resource_allocation"] = {
                "agents_needed": 1,
                "estimated_time": "15分钟",
                "complexity": "低"
            }
        
        elif complexity_level == ComplexityLevel.MODERATE:
            routing_plan["primary_route"] = "mdt_team"
            routing_plan["specialty_routing"] = required_specialties[:2]  # 最多2个专科
            routing_plan["resource_allocation"] = {
                "agents_needed": 3,
                "estimated_time": "90分钟",
                "complexity": "中"
            }
        
        else:  # HIGH
            routing_plan["primary_route"] = "ict_team"
            routing_plan["specialty_routing"] = required_specialties
            routing_plan["resource_allocation"] = {
                "agents_needed": len(required_specialties) + 3,
                "estimated_time": "200分钟",
                "complexity": "高"
            }
        
        return routing_plan
    
    def _assess_risk_level(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """评估风险等级"""
        risk_assessment = {
            "risk_level": "低风险",
            "risk_factors": [],
            "monitoring_requirements": [],
            "immediate_actions": []
        }
        
        # 高风险症状检测
        high_risk_symptoms = [
            "胸痛", "呼吸困难", "昏迷", "严重出血", 
            "剧烈头痛", "意识障碍", "瘫痪", "癫痫持续状态"
        ]
        
        for symptom in high_risk_symptoms:
            if symptom in query:
                risk_assessment["risk_level"] = "高风险"
                risk_assessment["risk_factors"].append(f"出现{symptom}症状")
                risk_assessment["immediate_actions"].append("立即就医")
                break
        
        # 中等风险症状
        medium_risk_symptoms = [
            "持续发热", "体重快速下降", "严重疼痛", 
            "视力模糊", "吞咽困难"
        ]
        
        if risk_assessment["risk_level"] == "低风险":
            for symptom in medium_risk_symptoms:
                if symptom in query:
                    risk_assessment["risk_level"] = "中风险"
                    risk_assessment["risk_factors"].append(f"出现{symptom}症状")
                    risk_assessment["monitoring_requirements"].append("密切观察症状变化")
                    break
        
        # 根据上下文调整风险评估
        if context:
            if context.get("elderly", False):
                risk_assessment["risk_factors"].append("老年患者")
                if risk_assessment["risk_level"] == "低风险":
                    risk_assessment["risk_level"] = "中风险"
            
            if context.get("chronic_diseases", False):
                risk_assessment["risk_factors"].append("慢性病史")
                risk_assessment["monitoring_requirements"].append("注意药物相互作用")
        
        return risk_assessment
    
    def _detect_emergency_symptoms(self, query: str) -> bool:
        """检测紧急症状"""
        emergency_keywords = [
            "胸痛", "呼吸困难", "昏迷", "严重出血", "剧烈头痛",
            "意识障碍", "抽搐", "瘫痪", "言语不清", "视力丧失"
        ]
        
        return any(keyword in query for keyword in emergency_keywords)
    
    def _identify_specialties(self, query: str) -> List[str]:
        """识别需要的专科"""
        specialties = []
        
        specialty_keywords = {
            "心血管科": ["胸痛", "心悸", "血压", "心脏病"],
            "神经科": ["头痛", "头晕", "肢体无力", "癫痫", "脑卒中"],
            "呼吸科": ["咳嗽", "气短", "哮喘", "肺炎"],
            "消化科": ["腹痛", "消化不良", "胃痛", "腹泻"],
            "内分泌科": ["糖尿病", "甲状腺", "体重变化"],
            "肿瘤科": ["肿块", "体重下降", "癌症", "肿瘤标志物"]
        }
        
        for specialty, keywords in specialty_keywords.items():
            if any(keyword in query for keyword in keywords):
                specialties.append(specialty)
        
        return specialties
    
    def get_moderation_summary(self) -> Dict[str, Any]:
        """获取调节总结"""
        return {
            "agent_name": self.name,
            "capabilities": [
                "复杂度评估",
                "智能路由",
                "风险评估",
                "紧急情况检测"
            ],
            "supported_complexity_levels": [level.value for level in ComplexityLevel],
            "specialty_coverage": [
                "心血管科", "神经科", "呼吸科", "消化科", 
                "内分泌科", "肿瘤科", "全科"
            ]
        }