"""
整合代理模块
负责整合多个专家代理的建议，形成统一的医疗决策
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import statistics
from collections import Counter

from ..config import AgentConfig, AgentType, ComplexityLevel
from .base_agent import BaseAgent
from ..core.complexity_analyzer import ComplexityAnalyzer
from ..tools.logger import setup_logger


class IntegratorAgent(BaseAgent):
    """整合智能体"""
    
    def __init__(self, config: AgentConfig, api_model=None, llm_config=None):
        super().__init__(config, api_model, llm_config)
        
        self.logger = setup_logger(self.__class__.__name__)
        
        # 整合策略
        self.integration_strategies = self._initialize_integration_strategies()
        
        # 共识算法
        self.consensus_algorithms = self._initialize_consensus_algorithms()
        
        # 冲突解决规则
        self.conflict_resolution_rules = self._initialize_conflict_resolution_rules()
        
        # 质量权重
        self.quality_weights = self._initialize_quality_weights()
        
        # 整合历史
        self.integration_history = []
    
    def _initialize_integration_strategies(self) -> Dict[str, Any]:
        """初始化整合策略"""
        return {
            "weighted_average": {
                "name": "加权平均",
                "description": "根据代理质量权重计算加权平均",
                "applicable_scenarios": ["multiple_agents", "similar_recommendations"],
                "complexity_threshold": ComplexityLevel.MODERATE
            },
            "majority_vote": {
                "name": "多数投票",
                "description": "选择多数代理支持的选项",
                "applicable_scenarios": ["diagnostic_consensus", "treatment_choice"],
                "complexity_threshold": ComplexityLevel.SIMPLE
            },
            "expert_override": {
                "name": "专家override",
                "description": "高等级专家意见覆盖其他意见",
                "applicable_scenarios": ["conflicting_opinions", "critical_decisions"],
                "complexity_threshold": ComplexityLevel.COMPLEX
            },
            "hierarchical_consensus": {
                "name": "层次共识",
                "description": "按专家等级逐步达成共识",
                "applicable_scenarios": ["complex_cases", "multi_specialty"],
                "complexity_threshold": ComplexityLevel.COMPLEX
            },
            "hybrid_approach": {
                "name": "混合方法",
                "description": "结合多种整合策略",
                "applicable_scenarios": ["very_complex", "uncertain_cases"],
                "complexity_threshold": ComplexityLevel.COMPLEX
            }
        }
    
    def _initialize_consensus_algorithms(self) -> Dict[str, Any]:
        """初始化共识算法"""
        return {
            "simple_majority": {
                "threshold": 0.5,
                "description": "简单多数决"
            },
            "supermajority": {
                "threshold": 0.67,
                "description": "超级多数决（2/3）"
            },
            "unanimous": {
                "threshold": 1.0,
                "description": "全体一致"
            },
            "weighted_consensus": {
                "threshold": 0.6,
                "description": "加权共识"
            },
            "iterative_consensus": {
                "threshold": 0.8,
                "description": "迭代共识"
            }
        }
    
    def _initialize_conflict_resolution_rules(self) -> Dict[str, Any]:
        """初始化冲突解决规则"""
        return {
            "diagnostic_conflicts": {
                "priority": "specialist_expertise",
                "resolution_method": "evidence_based",
                "escalation": "panel_discussion"
            },
            "treatment_conflicts": {
                "priority": "patient_safety",
                "resolution_method": "risk_benefit_analysis",
                "escalation": "multidisciplinary_team"
            },
            "medication_conflicts": {
                "priority": "drug_interactions",
                "resolution_method": "pharmacist_review",
                "escalation": "clinical_pharmacist"
            },
            "procedure_conflicts": {
                "priority": "invasiveness",
                "resolution_method": "risk_assessment",
                "escalation": "surgical_committee"
            }
        }
    
    def _initialize_quality_weights(self) -> Dict[str, Any]:
        """初始化质量权重"""
        return {
            "agent_type_weights": {
                AgentType.SPECIALIST: 1.0,
                AgentType.PCC: 0.7,
                AgentType.MODERATOR: 0.8,
                AgentType.RECRUITER: 0.6,
                AgentType.REVIEWER: 0.9,
                AgentType.INTEGRATOR: 1.0
            },
            "specialty_weights": {
                "心血管科": 1.0,
                "神经科": 1.0,
                "肿瘤科": 1.0,
                "急诊科": 1.0,
                "内科": 0.8,
                "外科": 0.9,
                "儿科": 0.9,
                "妇科": 0.9,
                "其他": 0.7
            },
            "confidence_weights": {
                "high": 1.0,
                "medium": 0.8,
                "low": 0.6
            }
        }
    
    def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理整合请求"""
        try:
            self.logger.info(f"开始整合查询: {query[:100]}...")
            
            if not context or 'agent_recommendations' not in context:
                return self._create_error_response("缺少代理建议进行整合")
            
            agent_recommendations = context['agent_recommendations']
            query_complexity = context.get('complexity_level', ComplexityLevel.MODERATE)
            patient_context = context.get('patient_context', {})
            
            # 执行整合
            integration_result = self._perform_integration(
                query, agent_recommendations, query_complexity, patient_context
            )
            
            # 记录整合历史
            self._record_integration_history(query, integration_result)
            
            self.logger.info(f"整合完成，使用策略: {integration_result['integration_strategy']['name']}")
            
            return {
                "status": "success",
                "integration_result": integration_result,
                "integration_id": self._generate_integration_id(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"整合过程中发生错误: {str(e)}")
            return self._create_error_response(f"整合失败: {str(e)}")
    
    def _perform_integration(self, query: str, agent_recommendations: List[Dict[str, Any]], 
                           complexity_level: ComplexityLevel, patient_context: Dict[str, Any]) -> Dict[str, Any]:
        """执行整合过程"""
        
        # 1. 预处理建议
        processed_recommendations = self._preprocess_recommendations(agent_recommendations)
        
        # 2. 选择整合策略
        integration_strategy = self._select_integration_strategy(
            processed_recommendations, complexity_level, query
        )
        
        # 3. 执行整合
        if integration_strategy["name"] == "加权平均":
            integrated_result = self._weighted_average_integration(processed_recommendations)
        elif integration_strategy["name"] == "多数投票":
            integrated_result = self._majority_vote_integration(processed_recommendations)
        elif integration_strategy["name"] == "专家override":
            integrated_result = self._expert_override_integration(processed_recommendations)
        elif integration_strategy["name"] == "层次共识":
            integrated_result = self._hierarchical_consensus_integration(processed_recommendations)
        else:  # 混合方法
            integrated_result = self._hybrid_integration(processed_recommendations, complexity_level)
        
        # 4. 冲突解决
        resolved_conflicts = self._resolve_conflicts(processed_recommendations, integrated_result)
        
        # 5. 质量评估
        quality_assessment = self._assess_integration_quality(
            processed_recommendations, integrated_result, resolved_conflicts
        )
        
        # 6. 生成最终建议
        final_recommendation = self._generate_final_recommendation(
            integrated_result, resolved_conflicts, quality_assessment
        )
        
        return {
            "integration_strategy": integration_strategy,
            "processed_recommendations": processed_recommendations,
            "integrated_result": integrated_result,
            "resolved_conflicts": resolved_conflicts,
            "quality_assessment": quality_assessment,
            "final_recommendation": final_recommendation,
            "consensus_level": self._calculate_consensus_level(processed_recommendations),
            "confidence_score": self._calculate_confidence_score(processed_recommendations, quality_assessment),
            "integration_timestamp": datetime.now().isoformat()
        }
    
    def _preprocess_recommendations(self, agent_recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """预处理代理建议"""
        processed = []
        
        for i, rec in enumerate(agent_recommendations):
            processed_rec = {
                "id": rec.get("id", f"rec_{i}"),
                "agent_type": rec.get("agent_type", "unknown"),
                "specialty": rec.get("specialty", "通用"),
                "recommendation": rec.get("recommendation", ""),
                "confidence": rec.get("confidence", "medium"),
                "diagnosis": rec.get("diagnosis", []),
                "treatment": rec.get("treatment", []),
                "medications": rec.get("medications", []),
                "tests": rec.get("tests", []),
                "reasoning": rec.get("reasoning", ""),
                "quality_score": rec.get("quality_score", 7.0),
                "evidence_level": rec.get("evidence_level", "中等"),
                "weight": self._calculate_agent_weight(rec)
            }
            processed.append(processed_rec)
        
        return processed
    
    def _calculate_agent_weight(self, recommendation: Dict[str, Any]) -> float:
        """计算代理权重"""
        base_weight = self.quality_weights["agent_type_weights"].get(
            recommendation.get("agent_type"), 0.5
        )
        
        # 专科权重
        specialty_weight = self.quality_weights["specialty_weights"].get(
            recommendation.get("specialty", "其他"), 0.7
        )
        
        # 信心权重
        confidence_weight = self.quality_weights["confidence_weights"].get(
            recommendation.get("confidence", "medium"), 0.8
        )
        
        # 质量分数权重
        quality_weight = recommendation.get("quality_score", 7.0) / 10.0
        
        final_weight = base_weight * specialty_weight * confidence_weight * quality_weight
        
        return min(final_weight, 1.0)
    
    def _select_integration_strategy(self, processed_recommendations: List[Dict[str, Any]], 
                                   complexity_level: ComplexityLevel, query: str) -> Dict[str, Any]:
        """选择整合策略"""
        
        num_agents = len(processed_recommendations)
        confidence_levels = [rec["confidence"] for rec in processed_recommendations]
        agent_types = [rec["agent_type"] for rec in processed_recommendations]
        
        # 策略选择逻辑
        if complexity_level == ComplexityLevel.SIMPLE and num_agents <= 2:
            return self.integration_strategies["majority_vote"]
        elif complexity_level == ComplexityLevel.COMPLEX or num_agents >= 4:
            if len(set(agent_types)) > 2:  # 多专科
                return self.integration_strategies["hierarchical_consensus"]
            else:
                return self.integration_strategies["expert_override"]
        elif self._has_high_confidence_variation(confidence_levels):
            return self.integration_strategies["weighted_average"]
        elif self._detect_potential_conflicts(processed_recommendations):
            return self.integration_strategies["hybrid_approach"]
        else:
            return self.integration_strategies["weighted_average"]
    
    def _has_high_confidence_variation(self, confidence_levels: List[str]) -> bool:
        """检测信心水平差异"""
        confidence_counts = Counter(confidence_levels)
        if len(confidence_counts) > 1:
            max_count = max(confidence_counts.values())
            total_count = len(confidence_levels)
            return max_count / total_count < 0.7  # 没有明显多数
        return False
    
    def _detect_potential_conflicts(self, processed_recommendations: List[Dict[str, Any]]) -> bool:
        """检测潜在冲突"""
        diagnoses = []
        treatments = []
        
        for rec in processed_recommendations:
            diagnoses.extend(rec.get("diagnosis", []))
            treatments.extend(rec.get("treatment", []))
        
        # 检查诊断冲突
        diag_conflicts = len(set(diagnoses)) > len(diagnoses) * 0.7
        
        # 检查治疗冲突
        treat_conflicts = len(set(treatments)) > len(treatments) * 0.7
        
        return diag_conflicts or treat_conflicts
    
    def _weighted_average_integration(self, processed_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """加权平均整合"""
        
        # 计算加权平均诊断
        diagnosis_scores = {}
        for rec in processed_recommendations:
            weight = rec["weight"]
            for diag in rec.get("diagnosis", []):
                if diag not in diagnosis_scores:
                    diagnosis_scores[diag] = []
                diagnosis_scores[diag].append(weight)
        
        # 计算诊断最终分数
        final_diagnoses = {}
        for diag, weights in diagnosis_scores.items():
            final_diagnoses[diag] = {
                "score": sum(weights),
                "confidence": "high" if sum(weights) > 0.8 else "medium" if sum(weights) > 0.5 else "low",
                "supporting_agents": len(weights)
            }
        
        # 计算加权平均治疗方案
        treatment_scores = {}
        for rec in processed_recommendations:
            weight = rec["weight"]
            for treat in rec.get("treatment", []):
                if treat not in treatment_scores:
                    treatment_scores[treat] = []
                treatment_scores[treat].append(weight)
        
        final_treatments = {}
        for treat, weights in treatment_scores.items():
            final_treatments[treat] = {
                "score": sum(weights),
                "confidence": "high" if sum(weights) > 0.8 else "medium" if sum(weights) > 0.5 else "low",
                "supporting_agents": len(weights)
            }
        
        return {
            "diagnoses": final_diagnoses,
            "treatments": final_treatments,
            "medications": self._integrate_medications(processed_recommendations),
            "tests": self._integrate_tests(processed_recommendations),
            "reasoning": self._synthesize_reasoning(processed_recommendations)
        }
    
    def _majority_vote_integration(self, processed_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """多数投票整合"""
        
        # 收集所有诊断和治疗
        all_diagnoses = []
        all_treatments = []
        
        for rec in processed_recommendations:
            all_diagnoses.extend(rec.get("diagnosis", []))
            all_treatments.extend(rec.get("treatment", []))
        
        # 多数投票选择
        diag_votes = Counter(all_diagnoses)
        treat_votes = Counter(all_treatments)
        
        # 选择得票最多的选项
        top_diagnoses = diag_votes.most_common(3)
        top_treatments = treat_votes.most_common(5)
        
        final_diagnoses = {
            diag: {
                "votes": votes,
                "percentage": votes / len(processed_recommendations),
                "confidence": "high" if votes > len(processed_recommendations) * 0.6 else "medium"
            }
            for diag, votes in top_diagnoses if votes >= 2
        }
        
        final_treatments = {
            treat: {
                "votes": votes,
                "percentage": votes / len(processed_recommendations),
                "confidence": "high" if votes > len(processed_recommendations) * 0.6 else "medium"
            }
            for treat, votes in top_treatments if votes >= 2
        }
        
        return {
            "diagnoses": final_diagnoses,
            "treatments": final_treatments,
            "medications": self._integrate_medications(processed_recommendations),
            "tests": self._integrate_tests(processed_recommendations),
            "reasoning": f"基于{len(processed_recommendations)}位专家的多数投票结果"
        }
    
    def _expert_override_integration(self, processed_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """专家override整合"""
        
        # 找到最高权重专家
        highest_weight_rec = max(processed_recommendations, key=lambda x: x["weight"])
        
        # 如果最高权重专家的信心是high，则采用其建议
        if highest_weight_rec["confidence"] == "high" and highest_weight_rec["weight"] > 0.8:
            return {
                "diagnoses": {diag: {"confidence": "high", "source": "expert_override"} 
                            for diag in highest_weight_rec.get("diagnosis", [])},
                "treatments": {treat: {"confidence": "high", "source": "expert_override"} 
                             for treat in highest_weight_rec.get("treatment", [])},
                "medications": highest_weight_rec.get("medications", []),
                "tests": highest_weight_rec.get("tests", []),
                "reasoning": f"采用{highest_weight_rec['agent_type']}专家的高信心建议",
                "override_expert": highest_weight_rec["agent_type"]
            }
        else:
            # 如果最高权重专家信心不够，则回退到加权平均
            return self._weighted_average_integration(processed_recommendations)
    
    def _hierarchical_consensus_integration(self, processed_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """层次共识整合"""
        
        # 按权重排序
        sorted_recs = sorted(processed_recommendations, key=lambda x: x["weight"], reverse=True)
        
        # 从最高权重开始，逐步建立共识
        consensus_diagnoses = {}
        consensus_treatments = {}
        
        # 收集所有建议
        all_diagnoses = []
        all_treatments = []
        
        for rec in sorted_recs:
            all_diagnoses.extend(rec.get("diagnosis", []))
            all_treatments.extend(rec.get("treatment", []))
        
        # 寻找共识
        diag_counts = Counter(all_diagnoses)
        treat_counts = Counter(all_treatments)
        
        # 设定共识阈值
        consensus_threshold = max(2, len(processed_recommendations) * 0.6)
        
        final_diagnoses = {
            diag: {
                "votes": votes,
                "consensus_level": votes / len(processed_recommendations),
                "confidence": "high" if votes >= consensus_threshold else "medium"
            }
            for diag, votes in diag_counts.items() if votes >= 2
        }
        
        final_treatments = {
            treat: {
                "votes": votes,
                "consensus_level": votes / len(processed_recommendations),
                "confidence": "high" if votes >= consensus_threshold else "medium"
            }
            for treat, votes in treat_counts.items() if votes >= 2
        }
        
        return {
            "diagnoses": final_diagnoses,
            "treatments": final_treatments,
            "medications": self._integrate_medications(processed_recommendations),
            "tests": self._integrate_tests(processed_recommendations),
            "reasoning": f"通过层次共识达成，涉及{len(processed_recommendations)}位专家",
            "consensus_threshold": consensus_threshold
        }
    
    def _hybrid_integration(self, processed_recommendations: List[Dict[str, Any]], 
                          complexity_level: ComplexityLevel) -> Dict[str, Any]:
        """混合方法整合"""
        
        # 首先尝试多数投票
        majority_result = self._majority_vote_integration(processed_recommendations)
        
        # 检查是否有足够的一致性
        total_diagnoses = sum(len(diag.keys()) for diag in majority_result["diagnoses"].values())
        total_treatments = sum(len(treat.keys()) for treat in majority_result["treatments"].values())
        
        # 如果一致性不够，则使用加权平均
        if (total_diagnoses > 3 or total_treatments > 5 or 
            complexity_level == ComplexityLevel.COMPLEX):
            
            weighted_result = self._weighted_average_integration(processed_recommendations)
            
            # 合并结果
            final_diagnoses = {**majority_result["diagnoses"], **weighted_result["diagnoses"]}
            final_treatments = {**majority_result["treatments"], **weighted_result["treatments"]}
            
            return {
                "diagnoses": final_diagnoses,
                "treatments": final_treatments,
                "medications": self._integrate_medications(processed_recommendations),
                "tests": self._integrate_tests(processed_recommendations),
                "reasoning": "综合多数投票和加权平均的结果",
                "methods_used": ["majority_vote", "weighted_average"]
            }
        
        return majority_result
    
    def _integrate_medications(self, processed_recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """整合药物建议"""
        all_medications = []
        
        for rec in processed_recommendations:
            for med in rec.get("medications", []):
                if isinstance(med, dict):
                    all_medications.append(med)
                else:
                    all_medications.append({"name": med, "dosage": "", "frequency": ""})
        
        # 去重并合并相似药物
        unique_medications = []
        for med in all_medications:
            med_name = med.get("name", "")
            if not any(existing_med.get("name") == med_name for existing_med in unique_medications):
                unique_medications.append(med)
        
        return unique_medications
    
    def _integrate_tests(self, processed_recommendations: List[Dict[str, Any]]) -> List[str]:
        """整合检查建议"""
        all_tests = []
        
        for rec in processed_recommendations:
            all_tests.extend(rec.get("tests", []))
        
        # 去重
        return list(set(all_tests))
    
    def _synthesize_reasoning(self, processed_recommendations: List[Dict[str, Any]]) -> str:
        """综合推理过程"""
        reasonings = [rec.get("reasoning", "") for rec in processed_recommendations if rec.get("reasoning")]
        
        if not reasonings:
            return "基于多位专家的综合分析"
        
        # 简单连接所有推理
        combined_reasoning = "；".join(reasonings)
        
        # 如果太长，则截取关键部分
        if len(combined_reasoning) > 500:
            combined_reasoning = combined_reasoning[:500] + "..."
        
        return combined_reasoning
    
    def _resolve_conflicts(self, processed_recommendations: List[Dict[str, Any]], 
                         integrated_result: Dict[str, Any]) -> Dict[str, Any]:
        """解决冲突"""
        
        conflicts = {
            "diagnostic_conflicts": [],
            "treatment_conflicts": [],
            "medication_conflicts": [],
            "procedure_conflicts": []
        }
        
        resolutions = {}
        
        # 检测诊断冲突
        diagnoses = integrated_result.get("diagnoses", {})
        if len(diagnoses) > 3:
            conflicts["diagnostic_conflicts"].append("诊断建议过多，可能存在分歧")
            resolutions["diagnostic_conflicts"] = "建议进一步检查明确诊断"
        
        # 检测治疗冲突
        treatments = integrated_result.get("treatments", {})
        if len(treatments) > 5:
            conflicts["treatment_conflicts"].append("治疗方案较多，需要统一")
            resolutions["treatment_conflicts"] = "建议根据患者具体情况选择最适合的治疗方案"
        
        return {
            "conflicts": conflicts,
            "resolutions": resolutions,
            "conflict_severity": self._assess_conflict_severity(conflicts)
        }
    
    def _assess_conflict_severity(self, conflicts: Dict[str, List[str]]) -> str:
        """评估冲突严重程度"""
        total_conflicts = sum(len(conflict_list) for conflict_list in conflicts.values())
        
        if total_conflicts == 0:
            return "无冲突"
        elif total_conflicts <= 2:
            return "轻微冲突"
        elif total_conflicts <= 4:
            return "中等冲突"
        else:
            return "严重冲突"
    
    def _assess_integration_quality(self, processed_recommendations: List[Dict[str, Any]], 
                                  integrated_result: Dict[str, Any], 
                                  resolved_conflicts: Dict[str, Any]) -> Dict[str, Any]:
        """评估整合质量"""
        
        quality_scores = {}
        
        # 一致性评分
        consistency_score = self._calculate_consistency_score(processed_recommendations, integrated_result)
        quality_scores["consistency"] = consistency_score
        
        # 完整性评分
        completeness_score = self._calculate_completeness_score(integrated_result)
        quality_scores["completeness"] = completeness_score
        
        # 信心评分
        confidence_score = self._calculate_integration_confidence_score(integrated_result)
        quality_scores["confidence"] = confidence_score
        
        # 可行性评分
        feasibility_score = self._calculate_feasibility_score(integrated_result, processed_recommendations)
        quality_scores["feasibility"] = feasibility_score
        
        # 综合评分
        overall_score = sum(quality_scores.values()) / len(quality_scores)
        
        return {
            "dimension_scores": quality_scores,
            "overall_score": overall_score,
            "quality_level": self._determine_quality_level(overall_score),
            "improvement_suggestions": self._generate_improvement_suggestions(quality_scores)
        }
    
    def _calculate_consistency_score(self, processed_recommendations: List[Dict[str, Any]], 
                                   integrated_result: Dict[str, Any]) -> float:
        """计算一致性评分"""
        if not processed_recommendations:
            return 0.0
        
        # 计算诊断一致性
        all_diagnoses = []
        for rec in processed_recommendations:
            all_diagnoses.extend(rec.get("diagnosis", []))
        
        if not all_diagnoses:
            return 5.0
        
        diag_consistency = 1.0 - (len(set(all_diagnoses)) - 1) / max(len(all_diagnoses), 1)
        
        # 计算治疗一致性
        all_treatments = []
        for rec in processed_recommendations:
            all_treatments.extend(rec.get("treatment", []))
        
        if not all_treatments:
            return 5.0
        
        treat_consistency = 1.0 - (len(set(all_treatments)) - 1) / max(len(all_treatments), 1)
        
        return (diag_consistency + treat_consistency) * 5.0
    
    def _calculate_completeness_score(self, integrated_result: Dict[str, Any]) -> float:
        """计算完整性评分"""
        score = 0.0
        
        # 检查诊断完整性
        if integrated_result.get("diagnoses"):
            score += 2.5
        
        # 检查治疗完整性
        if integrated_result.get("treatments"):
            score += 2.5
        
        # 检查用药完整性
        if integrated_result.get("medications"):
            score += 2.5
        
        # 检查检查建议完整性
        if integrated_result.get("tests"):
            score += 2.5
        
        return score
    
    def _calculate_integration_confidence_score(self, integrated_result: Dict[str, Any]) -> float:
        """计算整合信心评分"""
        confidence_scores = []
        
        # 诊断信心
        diagnoses = integrated_result.get("diagnoses", {})
        for diag_info in diagnoses.values():
            if isinstance(diag_info, dict):
                if diag_info.get("confidence") == "high":
                    confidence_scores.append(10.0)
                elif diag_info.get("confidence") == "medium":
                    confidence_scores.append(7.0)
                else:
                    confidence_scores.append(4.0)
        
        # 治疗信心
        treatments = integrated_result.get("treatments", {})
        for treat_info in treatments.values():
            if isinstance(treat_info, dict):
                if treat_info.get("confidence") == "high":
                    confidence_scores.append(10.0)
                elif treat_info.get("confidence") == "medium":
                    confidence_scores.append(7.0)
                else:
                    confidence_scores.append(4.0)
        
        return statistics.mean(confidence_scores) if confidence_scores else 5.0
    
    def _calculate_feasibility_score(self, integrated_result: Dict[str, Any], 
                                   processed_recommendations: List[Dict[str, Any]]) -> float:
        """计算可行性评分"""
        score = 7.0  # 基础可行性分数
        
        # 检查建议数量是否合理
        diag_count = len(integrated_result.get("diagnoses", {}))
        treat_count = len(integrated_result.get("treatments", {}))
        
        if diag_count > 5:  # 诊断太多
            score -= 1.0
        
        if treat_count > 8:  # 治疗太多
            score -= 1.0
        
        # 检查药物相互作用
        medications = integrated_result.get("medications", [])
        if len(medications) > 5:
            score -= 0.5
        
        # 检查专家数量是否足够
        if len(processed_recommendations) < 2:
            score -= 1.0
        
        return max(score, 0.0)
    
    def _determine_quality_level(self, score: float) -> str:
        """确定质量等级"""
        if score >= 8.5:
            return "优秀"
        elif score >= 7.0:
            return "良好"
        elif score >= 5.5:
            return "中等"
        else:
            return "较差"
    
    def _generate_improvement_suggestions(self, quality_scores: Dict[str, float]) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        for dimension, score in quality_scores.items():
            if score < 6.0:
                if dimension == "consistency":
                    suggestions.append("提高专家意见一致性")
                elif dimension == "completeness":
                    suggestions.append("补充完整的医疗建议")
                elif dimension == "confidence":
                    suggestions.append("提高建议的信心水平")
                elif dimension == "feasibility":
                    suggestions.append("优化建议的可行性")
        
        return suggestions
    
    def _calculate_consensus_level(self, processed_recommendations: List[Dict[str, Any]]) -> float:
        """计算共识水平"""
        if not processed_recommendations:
            return 0.0
        
        # 计算诊断共识
        all_diagnoses = []
        for rec in processed_recommendations:
            all_diagnoses.extend(rec.get("diagnosis", []))
        
        if not all_diagnoses:
            return 0.5
        
        diag_counter = Counter(all_diagnoses)
        max_diag_votes = max(diag_counter.values())
        diag_consensus = max_diag_votes / len(all_diagnoses)
        
        # 计算治疗共识
        all_treatments = []
        for rec in processed_recommendations:
            all_treatments.extend(rec.get("treatment", []))
        
        if not all_treatments:
            return diag_consensus
        
        treat_counter = Counter(all_treatments)
        max_treat_votes = max(treat_counter.values())
        treat_consensus = max_treat_votes / len(all_treatments)
        
        return (diag_consensus + treat_consensus) / 2.0
    
    def _calculate_confidence_score(self, processed_recommendations: List[Dict[str, Any]], 
                                  quality_assessment: Dict[str, Any]) -> float:
        """计算信心分数"""
        
        # 基础信心
        base_confidence = quality_assessment["overall_score"] / 10.0
        
        # 专家数量调整
        expert_count_factor = min(len(processed_recommendations) / 3.0, 1.0)
        
        # 共识水平调整
        consensus_level = self._calculate_consensus_level(processed_recommendations)
        
        final_confidence = base_confidence * 0.6 + expert_count_factor * 0.2 + consensus_level * 0.2
        
        return min(final_confidence, 1.0)
    
    def _generate_final_recommendation(self, integrated_result: Dict[str, Any], 
                                     resolved_conflicts: Dict[str, Any], 
                                     quality_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """生成最终建议"""
        
        # 格式化诊断建议
        formatted_diagnoses = []
        for diag, info in integrated_result.get("diagnoses", {}).items():
            if isinstance(info, dict):
                confidence = info.get("confidence", "medium")
                formatted_diagnoses.append({
                    "diagnosis": diag,
                    "confidence": confidence,
                    "supporting_evidence": info.get("supporting_agents", 1)
                })
        
        # 格式化治疗建议
        formatted_treatments = []
        for treat, info in integrated_result.get("treatments", {}).items():
            if isinstance(info, dict):
                confidence = info.get("confidence", "medium")
                formatted_treatments.append({
                    "treatment": treat,
                    "confidence": confidence,
                    "supporting_evidence": info.get("supporting_agents", 1)
                })
        
        # 综合推理
        overall_reasoning = integrated_result.get("reasoning", "基于多位专家的综合分析")
        
        # 风险提醒
        risk_warnings = []
        if resolved_conflicts["conflict_severity"] in ["中等冲突", "严重冲突"]:
            risk_warnings.append("存在专家意见分歧，需要谨慎决策")
        
        if quality_assessment["overall_score"] < 7.0:
            risk_warnings.append("整合质量有待提高，建议进一步确认")
        
        return {
            "primary_diagnosis": formatted_diagnoses[0] if formatted_diagnoses else None,
            "differential_diagnosis": formatted_diagnoses[1:] if len(formatted_diagnoses) > 1 else [],
            "primary_treatment": formatted_treatments[0] if formatted_treatments else None,
            "alternative_treatments": formatted_treatments[1:] if len(formatted_treatments) > 1 else [],
            "medications": integrated_result.get("medications", []),
            "recommended_tests": integrated_result.get("tests", []),
            "reasoning": overall_reasoning,
            "confidence_level": self._determine_overall_confidence(quality_assessment),
            "quality_assessment": quality_assessment,
            "risk_warnings": risk_warnings,
            "next_steps": self._generate_next_steps(integrated_result, resolved_conflicts)
        }
    
    def _determine_overall_confidence(self, quality_assessment: Dict[str, Any]) -> str:
        """确定总体信心水平"""
        overall_score = quality_assessment["overall_score"]
        
        if overall_score >= 8.5:
            return "很高"
        elif overall_score >= 7.5:
            return "高"
        elif overall_score >= 6.0:
            return "中等"
        elif overall_score >= 4.0:
            return "低"
        else:
            return "很低"
    
    def _generate_next_steps(self, integrated_result: Dict[str, Any], 
                           resolved_conflicts: Dict[str, Any]) -> List[str]:
        """生成下一步行动建议"""
        next_steps = []
        
        # 基于冲突情况
        if resolved_conflicts["conflict_severity"] == "严重冲突":
            next_steps.append("建议多学科会诊")
            next_steps.append("考虑寻求第三方专家意见")
        elif resolved_conflicts["conflict_severity"] == "中等冲突":
            next_steps.append("建议进一步检查明确诊断")
        
        # 基于质量评估
        if integrated_result.get("reasoning", "") == "基于多位专家的综合分析":
            next_steps.append("完善推理过程和证据支持")
        
        # 基于建议数量
        diag_count = len(integrated_result.get("diagnoses", {}))
        if diag_count > 3:
            next_steps.append("进一步检查以缩小诊断范围")
        
        treat_count = len(integrated_result.get("treatments", {}))
        if treat_count > 5:
            next_steps.append("根据患者具体情况选择最适合的治疗方案")
        
        return next_steps
    
    def _record_integration_history(self, query: str, integration_result: Dict[str, Any]):
        """记录整合历史"""
        record = {
            "query": query[:100],  # 截取前100字符
            "integration_result": integration_result,
            "timestamp": datetime.now().isoformat()
        }
        self.integration_history.append(record)
        
        # 保持历史记录在合理范围内
        if len(self.integration_history) > 1000:
            self.integration_history = self.integration_history[-500:]  # 保留最近500条
    
    def _generate_integration_id(self) -> str:
        """生成整合ID"""
        import uuid
        return f"INT_{datetime.now().strftime('%Y%m%d')}_{str(uuid.uuid4())[:8]}"
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """创建错误响应"""
        return {
            "status": "error",
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """获取整合统计信息"""
        if not self.integration_history:
            return {"message": "暂无整合历史记录"}
        
        total_integrations = len(self.integration_history)
        recent_integrations = self.integration_history[-100:]  # 最近100条
        
        # 统计各类策略使用
        strategy_usage = {}
        quality_distribution = {}
        
        for record in recent_integrations:
            strategy_name = record["integration_result"]["integration_strategy"]["name"]
            strategy_usage[strategy_name] = strategy_usage.get(strategy_name, 0) + 1
            
            quality_level = record["integration_result"]["quality_assessment"]["quality_level"]
            quality_distribution[quality_level] = quality_distribution.get(quality_level, 0) + 1
        
        return {
            "total_integrations": total_integrations,
            "recent_integrations_count": len(recent_integrations),
            "strategy_usage": strategy_usage,
            "quality_distribution": quality_distribution,
            "average_quality_score": sum(
                r["integration_result"]["quality_assessment"]["overall_score"] 
                for r in recent_integrations
            ) / len(recent_integrations),
            "average_consensus_level": sum(
                r["integration_result"]["consensus_level"] 
                for r in recent_integrations
            ) / len(recent_integrations)
        }