"""
审查代理模块
负责对医疗建议和决策进行质量审查和验证
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import re

from ..config import AgentConfig, AgentType, ComplexityLevel
from .base_agent import BaseAgent
from ..core.complexity_analyzer import ComplexityAnalyzer
from ..tools.logger import setup_logger


class ReviewerAgent(BaseAgent):
    """审查智能体"""
    
    def __init__(self, config: AgentConfig, api_model=None, llm_config=None):
        super().__init__(config, api_model, llm_config)
        
        self.logger = setup_logger(self.__class__.__name__)
        
        # 质量审查标准
        self.quality_standards = self._initialize_quality_standards()
        
        # 审查规则
        self.review_rules = self._initialize_review_rules()
        
        # 风险评估矩阵
        self.risk_matrix = self._initialize_risk_matrix()
        
        # 审查历史记录
        self.review_history = []
    
    def _initialize_quality_standards(self) -> Dict[str, Any]:
        """初始化质量审查标准"""
        return {
            "medical_accuracy": {
                "weight": 0.3,
                "criteria": [
                    "诊断准确性",
                    "治疗方案科学性",
                    "药物选择合理性",
                    "检查建议适当性"
                ]
            },
            "safety": {
                "weight": 0.25,
                "criteria": [
                    "用药安全性",
                    "检查风险评估",
                    "禁忌症识别",
                    "副作用预防"
                ]
            },
            "evidence_based": {
                "weight": 0.2,
                "criteria": [
                    "循证医学依据",
                    "临床指南遵循",
                    "最新研究支持",
                    "专家共识"
                ]
            },
            "completeness": {
                "weight": 0.15,
                "criteria": [
                    "信息完整性",
                    "分析全面性",
                    "建议具体性",
                    "注意事项详细性"
                ]
            },
            "communication": {
                "weight": 0.1,
                "criteria": [
                    "表达清晰性",
                    "逻辑连贯性",
                    "患者友好性",
                    "专业术语使用"
                ]
            }
        }
    
    def _initialize_review_rules(self) -> Dict[str, Any]:
        """初始化审查规则"""
        return {
            "critical_errors": [
                r"完全错误.*诊断",
                r"危险.*治疗.*建议",
                r"绝对.*禁忌.*药物",
                r"致命.*错误"
            ],
            "high_risk_keywords": [
                "手术", "化疗", "放疗", "侵入性检查",
                "强效药物", "长期用药", "多种药物联用"
            ],
            "red_flags": [
                "延误诊断",
                "漏诊",
                "误诊",
                "用药错误",
                "检查遗漏"
            ],
            "review_depth": {
                ComplexityLevel.SIMPLE: "快速审查",
                ComplexityLevel.MODERATE: "标准审查", 
                ComplexityLevel.COMPLEX: "深度审查"
            }
        }
    
    def _initialize_risk_matrix(self) -> Dict[str, Any]:
        """初始化风险评估矩阵"""
        return {
            "risk_levels": {
                "低风险": {
                    "score_range": (0, 2),
                    "action": "通过审查",
                    "color": "green"
                },
                "中等风险": {
                    "score_range": (3, 5),
                    "action": "建议修改",
                    "color": "yellow"
                },
                "高风险": {
                    "score_range": (6, 8),
                    "action": "需要重新评估",
                    "color": "orange"
                },
                "极高风险": {
                    "score_range": (9, 10),
                    "action": "拒绝通过",
                    "color": "red"
                }
            },
            "risk_factors": {
                "诊断不确定": 2,
                "治疗风险高": 3,
                "药物相互作用": 2,
                "检查风险": 1,
                "患者特殊状况": 2,
                "多系统疾病": 3,
                "紧急情况": 4
            }
        }
    
    def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理审查请求"""
        try:
            self.logger.info(f"开始审查查询: {query[:100]}...")
            
            if not context or 'medical_advice' not in context:
                return self._create_error_response("缺少医疗建议内容进行审查")
            
            medical_advice = context['medical_advice']
            agent_recommendations = context.get('agent_recommendations', [])
            complexity_level = context.get('complexity_level', ComplexityLevel.MODERATE)
            
            # 执行审查
            review_result = self._perform_comprehensive_review(
                query, medical_advice, agent_recommendations, complexity_level
            )
            
            # 记录审查历史
            self._record_review_history(query, review_result)
            
            self.logger.info(f"审查完成，风险等级: {review_result['risk_assessment']['risk_level']}")
            
            return {
                "status": "success",
                "review_result": review_result,
                "review_id": self._generate_review_id(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"审查过程中发生错误: {str(e)}")
            return self._create_error_response(f"审查失败: {str(e)}")
    
    def _perform_comprehensive_review(self, query: str, medical_advice: str, 
                                    agent_recommendations: List[Dict[str, Any]], 
                                    complexity_level: ComplexityLevel) -> Dict[str, Any]:
        """执行综合审查"""
        
        # 1. 质量评估
        quality_assessment = self._assess_quality(medical_advice, agent_recommendations)
        
        # 2. 风险评估
        risk_assessment = self._assess_risk(medical_advice, query, complexity_level)
        
        # 3. 安全审查
        safety_review = self._review_safety(medical_advice, agent_recommendations)
        
        # 4. 证据审查
        evidence_review = self._review_evidence(medical_advice, agent_recommendations)
        
        # 5. 完整性检查
        completeness_check = self._check_completeness(medical_advice, query)
        
        # 6. 综合评分
        overall_score = self._calculate_overall_score(
            quality_assessment, risk_assessment, safety_review, 
            evidence_review, completeness_check
        )
        
        # 7. 生成审查建议
        recommendations = self._generate_review_recommendations(
            quality_assessment, risk_assessment, safety_review, 
            evidence_review, completeness_check, overall_score
        )
        
        # 8. 决定审查结果
        review_decision = self._make_review_decision(overall_score, risk_assessment)
        
        return {
            "quality_assessment": quality_assessment,
            "risk_assessment": risk_assessment,
            "safety_review": safety_review,
            "evidence_review": evidence_review,
            "completeness_check": completeness_check,
            "overall_score": overall_score,
            "recommendations": recommendations,
            "review_decision": review_decision,
            "review_timestamp": datetime.now().isoformat()
        }
    
    def _assess_quality(self, medical_advice: str, agent_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估医疗建议质量"""
        scores = {}
        details = {}
        
        for dimension, standard in self.quality_standards.items():
            score = self._evaluate_dimension(medical_advice, agent_recommendations, dimension)
            scores[dimension] = score
            details[dimension] = {
                "score": score,
                "weight": standard["weight"],
                "weighted_score": score * standard["weight"],
                "criteria_met": self._check_criteria(medical_advice, standard["criteria"])
            }
        
        total_score = sum(details[dim]["weighted_score"] for dim in details)
        
        return {
            "dimension_scores": scores,
            "dimension_details": details,
            "total_score": total_score,
            "quality_level": self._determine_quality_level(total_score)
        }
    
    def _evaluate_dimension(self, medical_advice: str, agent_recommendations: List[Dict[str, Any]], 
                          dimension: str) -> float:
        """评估特定维度的质量"""
        
        if dimension == "medical_accuracy":
            return self._evaluate_medical_accuracy(medical_advice, agent_recommendations)
        elif dimension == "safety":
            return self._evaluate_safety(medical_advice)
        elif dimension == "evidence_based":
            return self._evaluate_evidence_based(medical_advice)
        elif dimension == "completeness":
            return self._evaluate_completeness(medical_advice, agent_recommendations)
        elif dimension == "communication":
            return self._evaluate_communication(medical_advice)
        else:
            return 5.0  # 默认中等分数
    
    def _evaluate_medical_accuracy(self, medical_advice: str, agent_recommendations: List[Dict[str, Any]]) -> float:
        """评估医学准确性"""
        score = 5.0
        
        # 检查诊断一致性
        if len(agent_recommendations) > 1:
            diagnoses = [rec.get('diagnosis', '') for rec in agent_recommendations]
            if len(set(diagnoses)) == 1:  # 诊断一致
                score += 1.0
        
        # 检查治疗方案合理性
        if "治疗方案" in medical_advice and "药物" in medical_advice:
            score += 0.5
        
        # 检查检查建议适当性
        if any(keyword in medical_advice for keyword in ["检查", "化验", "影像"]):
            score += 0.5
        
        return min(score, 10.0)
    
    def _evaluate_safety(self, medical_advice: str) -> float:
        """评估安全性"""
        score = 7.0  # 基础安全分数
        
        # 检查禁忌症提及
        if any(keyword in medical_advice for keyword in ["禁忌", "避免", "慎用"]):
            score += 1.0
        
        # 检查副作用提醒
        if any(keyword in medical_advice for keyword in ["副作用", "不良反应", "注意"]):
            score += 1.0
        
        # 检查风险警告
        high_risk_count = sum(1 for keyword in self.review_rules["high_risk_keywords"] 
                            if keyword in medical_advice)
        if high_risk_count > 0:
            score -= high_risk_count * 0.5
        
        return max(min(score, 10.0), 0.0)
    
    def _evaluate_evidence_based(self, medical_advice: str) -> float:
        """评估循证医学依据"""
        score = 4.0
        
        # 检查指南提及
        if any(keyword in medical_advice for keyword in ["指南", "共识", "标准"]):
            score += 2.0
        
        # 检查循证表述
        evidence_terms = ["循证", "研究显示", "临床试验", "Meta分析"]
        if any(term in medical_advice for term in evidence_terms):
            score += 2.0
        
        # 检查专家建议
        if "专家" in medical_advice:
            score += 1.0
        
        return min(score, 10.0)
    
    def _evaluate_completeness(self, medical_advice: str, agent_recommendations: List[Dict[str, Any]]) -> float:
        """评估完整性"""
        score = 5.0
        
        # 检查必要组成部分
        components = ["诊断", "治疗", "用药", "检查", "随访"]
        for component in components:
            if component in medical_advice:
                score += 0.8
        
        # 检查注意事项
        if any(keyword in medical_advice for keyword in ["注意", "提醒", "警告"]):
            score += 1.0
        
        # 检查随访计划
        if "随访" in medical_advice or "复查" in medical_advice:
            score += 1.0
        
        return min(score, 10.0)
    
    def _evaluate_communication(self, medical_advice: str) -> float:
        """评估沟通效果"""
        score = 6.0
        
        # 检查逻辑性
        sentences = medical_advice.split('。')
        if len(sentences) > 3:  # 有一定长度
            score += 1.0
        
        # 检查清晰度
        if not any(char in medical_advice for char in ["？", "？", "？"]):  # 无疑问句
            score += 1.0
        
        # 检查专业术语使用
        technical_terms = ["诊断", "治疗", "药物", "检查", "症状"]
        if sum(1 for term in technical_terms if term in medical_advice) >= 3:
            score += 1.0
        
        return min(score, 10.0)
    
    def _check_criteria(self, medical_advice: str, criteria: List[str]) -> List[str]:
        """检查标准是否满足"""
        met_criteria = []
        for criterion in criteria:
            if any(keyword in medical_advice for keyword in criterion.split()):
                met_criteria.append(criterion)
        return met_criteria
    
    def _determine_quality_level(self, score: float) -> str:
        """确定质量等级"""
        if score >= 8.5:
            return "优秀"
        elif score >= 7.0:
            return "良好"
        elif score >= 5.5:
            return "中等"
        elif score >= 4.0:
            return "较差"
        else:
            return "差"
    
    def _assess_risk(self, medical_advice: str, query: str, complexity_level: ComplexityLevel) -> Dict[str, Any]:
        """评估风险"""
        risk_factors = []
        risk_score = 0
        
        # 分析风险因素
        for factor, score in self.risk_matrix["risk_factors"].items():
            if self._check_risk_factor(medical_advice, query, factor):
                risk_factors.append(factor)
                risk_score += score
        
        # 复杂度调整
        complexity_multiplier = {
            ComplexityLevel.SIMPLE: 0.8,
            ComplexityLevel.MODERATE: 1.0,
            ComplexityLevel.COMPLEX: 1.3
        }
        adjusted_score = risk_score * complexity_multiplier.get(complexity_level, 1.0)
        
        # 确定风险等级
        risk_level = self._determine_risk_level(adjusted_score)
        
        return {
            "risk_factors": risk_factors,
            "base_risk_score": risk_score,
            "adjusted_risk_score": adjusted_score,
            "risk_level": risk_level,
            "risk_description": self._get_risk_description(risk_level),
            "mitigation_suggestions": self._generate_mitigation_suggestions(risk_factors)
        }
    
    def _check_risk_factor(self, medical_advice: str, query: str, factor: str) -> bool:
        """检查特定风险因素"""
        factor_patterns = {
            "诊断不确定": ["可能", "疑似", "考虑", "不排除"],
            "治疗风险高": ["手术", "化疗", "放疗", "侵入性"],
            "药物相互作用": ["药物", "联用", "相互作用"],
            "检查风险": ["穿刺", "活检", "造影", "有创"],
            "患者特殊状况": ["老年", "儿童", "孕妇", "合并症"],
            "多系统疾病": ["多个系统", "复杂", "多器官"],
            "紧急情况": ["紧急", "立即", "急诊", "危重"]
        }
        
        patterns = factor_patterns.get(factor, [])
        text = medical_advice + " " + query
        return any(pattern in text for pattern in patterns)
    
    def _determine_risk_level(self, score: float) -> str:
        """确定风险等级"""
        for level, info in self.risk_matrix["risk_levels"].items():
            min_score, max_score = info["score_range"]
            if min_score <= score <= max_score:
                return level
        return "极高风险"
    
    def _get_risk_description(self, risk_level: str) -> str:
        """获取风险描述"""
        descriptions = {
            "低风险": "该医疗建议风险较低，可以安全实施",
            "中等风险": "该医疗建议存在一定风险，需要谨慎实施",
            "高风险": "该医疗建议风险较高，需要严格监控",
            "极高风险": "该医疗建议风险极高，不建议实施"
        }
        return descriptions.get(risk_level, "风险评估失败")
    
    def _generate_mitigation_suggestions(self, risk_factors: List[str]) -> List[str]:
        """生成风险缓解建议"""
        suggestions = []
        
        for factor in risk_factors:
            if factor == "诊断不确定":
                suggestions.append("建议进一步完善检查以明确诊断")
            elif factor == "治疗风险高":
                suggestions.append("建议评估风险效益比，必要时咨询专家")
            elif factor == "药物相互作用":
                suggestions.append("建议仔细评估药物相互作用，调整用药方案")
            elif factor == "检查风险":
                suggestions.append("建议充分告知患者检查风险，取得知情同意")
            elif factor == "患者特殊状况":
                suggestions.append("建议考虑患者特殊状况，制定个性化方案")
            elif factor == "多系统疾病":
                suggestions.append("建议多学科会诊，制定综合治疗方案")
            elif factor == "紧急情况":
                suggestions.append("建议密切监测病情变化，及时调整治疗")
        
        return suggestions
    
    def _review_safety(self, medical_advice: str, agent_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """安全审查"""
        safety_issues = []
        safety_score = 10.0
        
        # 检查危险建议
        for pattern in self.review_rules["critical_errors"]:
            if re.search(pattern, medical_advice):
                safety_issues.append(f"发现危险模式: {pattern}")
                safety_score -= 3.0
        
        # 检查用药安全
        if "药物" in medical_advice:
            medication_safety = self._check_medication_safety(medical_advice)
            safety_issues.extend(medication_safety["issues"])
            safety_score -= medication_safety["penalty"]
        
        # 检查操作风险
        if any(keyword in medical_advice for keyword in ["手术", "操作", "侵入性"]):
            procedure_safety = self._check_procedure_safety(medical_advice)
            safety_issues.extend(procedure_safety["issues"])
            safety_score -= procedure_safety["penalty"]
        
        return {
            "safety_score": max(safety_score, 0.0),
            "safety_issues": safety_issues,
            "safety_level": self._determine_safety_level(safety_score),
            "recommendations": self._generate_safety_recommendations(safety_issues)
        }
    
    def _check_medication_safety(self, medical_advice: str) -> Dict[str, Any]:
        """检查用药安全"""
        issues = []
        penalty = 0
        
        # 检查剂量提及
        if "药物" in medical_advice and "剂量" not in medical_advice:
            issues.append("未明确药物剂量")
            penalty += 1.0
        
        # 检查禁忌症
        if any(word in medical_advice for word in ["过敏", "禁忌"]):
            issues.append("涉及药物过敏或禁忌症，需要特别关注")
            penalty += 2.0
        
        return {"issues": issues, "penalty": penalty}
    
    def _check_procedure_safety(self, medical_advice: str) -> Dict[str, Any]:
        """检查操作安全"""
        issues = []
        penalty = 0
        
        # 检查风险提示
        if "手术" in medical_advice and "风险" not in medical_advice:
            issues.append("手术建议未提及风险")
            penalty += 1.5
        
        # 检查知情同意
        if any(word in medical_advice for word in ["手术", "侵入性"]):
            issues.append("建议获取患者知情同意")
            penalty += 0.5
        
        return {"issues": issues, "penalty": penalty}
    
    def _determine_safety_level(self, score: float) -> str:
        """确定安全等级"""
        if score >= 8.0:
            return "安全"
        elif score >= 6.0:
            return "较安全"
        elif score >= 4.0:
            return "一般"
        else:
            return "不安全"
    
    def _generate_safety_recommendations(self, safety_issues: List[str]) -> List[str]:
        """生成安全建议"""
        recommendations = []
        
        for issue in safety_issues:
            if "剂量" in issue:
                recommendations.append("明确药物剂量和使用方法")
            if "风险" in issue:
                recommendations.append("详细说明治疗风险和注意事项")
            if "知情同意" in issue:
                recommendations.append("获取患者充分的知情同意")
        
        return list(set(recommendations))  # 去重
    
    def _review_evidence(self, medical_advice: str, agent_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """证据审查"""
        evidence_strength = 0
        evidence_sources = []
        gaps = []
        
        # 检查循证表述
        evidence_terms = ["研究显示", "临床试验", "Meta分析", "系统评价", "指南推荐"]
        for term in evidence_terms:
            if term in medical_advice:
                evidence_strength += 2
                evidence_sources.append(term)
        
        # 检查专家建议
        if "专家" in medical_advice:
            evidence_strength += 1
            evidence_sources.append("专家建议")
        
        # 检查循证医学等级
        if any(level in medical_advice for level in ["A级证据", "B级证据", "C级证据"]):
            evidence_strength += 2
            evidence_sources.append("证据等级")
        
        # 识别证据缺口
        if evidence_strength < 3:
            gaps.append("缺乏循证医学支持")
        if "指南" not in medical_advice:
            gaps.append("未引用临床指南")
        if len(agent_recommendations) > 1 and not evidence_sources:
            gaps.append("多方建议缺乏一致性证据")
        
        return {
            "evidence_strength": evidence_strength,
            "evidence_sources": evidence_sources,
            "evidence_gaps": gaps,
            "evidence_level": self._determine_evidence_level(evidence_strength),
            "recommendations": self._generate_evidence_recommendations(gaps)
        }
    
    def _determine_evidence_level(self, strength: int) -> str:
        """确定证据等级"""
        if strength >= 6:
            return "强证据"
        elif strength >= 4:
            return "中等证据"
        elif strength >= 2:
            return "弱证据"
        else:
            return "无明显证据"
    
    def _generate_evidence_recommendations(self, gaps: List[str]) -> List[str]:
        """生成证据建议"""
        recommendations = []
        
        for gap in gaps:
            if "循证医学支持" in gap:
                recommendations.append("补充循证医学证据支持")
            if "临床指南" in gap:
                recommendations.append("引用相关临床指南")
            if "一致性证据" in gap:
                recommendations.append("提供多方建议的一致性证据")
        
        return recommendations
    
    def _check_completeness(self, medical_advice: str, query: str) -> Dict[str, Any]:
        """完整性检查"""
        required_elements = {
            "诊断": "诊断结论",
            "治疗": "治疗方案",
            "用药": "用药建议",
            "检查": "检查建议",
            "随访": "随访计划"
        }
        
        missing_elements = []
        present_elements = []
        
        for element, description in required_elements.items():
            if element in medical_advice:
                present_elements.append(description)
            else:
                missing_elements.append(description)
        
        # 检查特殊情况
        if any(word in query for word in ["紧急", "急诊", "危重"]) and "紧急处理" not in medical_advice:
            missing_elements.append("紧急处理方案")
        
        if any(word in query for word in ["手术", "操作"]) and "术前准备" not in medical_advice:
            missing_elements.append("术前准备和风险评估")
        
        completeness_score = len(present_elements) / len(required_elements) * 10
        
        return {
            "present_elements": present_elements,
            "missing_elements": missing_elements,
            "completeness_score": completeness_score,
            "completeness_level": self._determine_completeness_level(completeness_score),
            "recommendations": [f"补充缺失内容: {element}" for element in missing_elements]
        }
    
    def _determine_completeness_level(self, score: float) -> str:
        """确定完整性等级"""
        if score >= 9.0:
            return "完整"
        elif score >= 7.0:
            return "较完整"
        elif score >= 5.0:
            return "一般"
        else:
            return "不完整"
    
    def _calculate_overall_score(self, quality_assessment: Dict, risk_assessment: Dict, 
                               safety_review: Dict, evidence_review: Dict, 
                               completeness_check: Dict) -> float:
        """计算综合评分"""
        
        # 质量评分 (40%)
        quality_score = quality_assessment["total_score"] * 0.4
        
        # 安全性评分 (30%)
        safety_score = safety_review["safety_score"] * 0.3
        
        # 证据评分 (15%)
        evidence_score = min(evidence_review["evidence_strength"] * 1.25, 10) * 0.15
        
        # 完整性评分 (10%)
        completeness_score = completeness_check["completeness_score"] * 0.1
        
        # 风险调整 (-5%)
        risk_penalty = risk_assessment["adjusted_risk_score"] * 0.05
        
        overall_score = quality_score + safety_score + evidence_score + completeness_score - risk_penalty
        
        return max(min(overall_score, 10.0), 0.0)
    
    def _generate_review_recommendations(self, quality_assessment: Dict, risk_assessment: Dict,
                                       safety_review: Dict, evidence_review: Dict,
                                       completeness_check: Dict, overall_score: float) -> Dict[str, Any]:
        """生成审查建议"""
        
        recommendations = {
            "quality_improvements": [],
            "safety_improvements": [],
            "evidence_improvements": [],
            "completeness_improvements": [],
            "risk_mitigation": risk_assessment["mitigation_suggestions"]
        }
        
        # 质量改进建议
        for dimension, detail in quality_assessment["dimension_details"].items():
            if detail["score"] < 7.0:
                recommendations["quality_improvements"].append(
                    f"改进{dimension}: 当前得分 {detail['score']:.1f}/10"
                )
        
        # 安全改进建议
        recommendations["safety_improvements"] = safety_review["recommendations"]
        
        # 证据改进建议
        recommendations["evidence_improvements"] = evidence_review["recommendations"]
        
        # 完整性改进建议
        recommendations["completeness_improvements"] = completeness_check["recommendations"]
        
        # 总体建议
        if overall_score < 7.0:
            recommendations["overall"] = "建议大幅改进医疗建议的质量和安全性"
        elif overall_score < 8.5:
            recommendations["overall"] = "建议进行适度改进以提高质量"
        else:
            recommendations["overall"] = "医疗建议质量良好，可以考虑小幅优化"
        
        return recommendations
    
    def _make_review_decision(self, overall_score: float, risk_assessment: Dict) -> Dict[str, Any]:
        """做出审查决定"""
        
        risk_level = risk_assessment["risk_level"]
        decision = {"status": "", "reason": "", "actions": []}
        
        # 基于分数和风险等级决定
        if risk_level in ["极高风险", "高风险"] or overall_score < 4.0:
            decision["status"] = "拒绝"
            decision["reason"] = "风险过高或质量严重不足"
            decision["actions"] = ["重新评估", "专家会诊", "修改建议"]
        elif risk_level == "中等风险" or overall_score < 7.0:
            decision["status"] = "有条件通过"
            decision["reason"] = "需要改进后通过"
            decision["actions"] = ["按建议改进", "重新提交审查"]
        elif overall_score >= 8.5:
            decision["status"] = "优秀通过"
            decision["reason"] = "质量优秀，风险可控"
            decision["actions"] = ["直接通过", "可作为范例"]
        else:
            decision["status"] = "通过"
            decision["reason"] = "质量合格，风险可控"
            decision["actions"] = ["直接通过"]
        
        return decision
    
    def _record_review_history(self, query: str, review_result: Dict[str, Any]):
        """记录审查历史"""
        record = {
            "query": query[:100],  # 截取前100字符
            "review_result": review_result,
            "timestamp": datetime.now().isoformat()
        }
        self.review_history.append(record)
        
        # 保持历史记录在合理范围内
        if len(self.review_history) > 1000:
            self.review_history = self.review_history[-500:]  # 保留最近500条
    
    def _generate_review_id(self) -> str:
        """生成审查ID"""
        import uuid
        return f"REV_{datetime.now().strftime('%Y%m%d')}_{str(uuid.uuid4())[:8]}"
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """创建错误响应"""
        return {
            "status": "error",
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_review_statistics(self) -> Dict[str, Any]:
        """获取审查统计信息"""
        if not self.review_history:
            return {"message": "暂无审查历史记录"}
        
        total_reviews = len(self.review_history)
        recent_reviews = self.review_history[-100:]  # 最近100条
        
        # 统计各类决策
        decisions = {}
        risk_levels = {}
        
        for record in recent_reviews:
            decision = record["review_result"]["review_decision"]["status"]
            risk = record["review_result"]["risk_assessment"]["risk_level"]
            
            decisions[decision] = decisions.get(decision, 0) + 1
            risk_levels[risk] = risk_levels.get(risk, 0) + 1
        
        return {
            "total_reviews": total_reviews,
            "recent_reviews_count": len(recent_reviews),
            "decision_distribution": decisions,
            "risk_level_distribution": risk_levels,
            "average_score": sum(r["review_result"]["overall_score"] for r in recent_reviews) / len(recent_reviews)
        }