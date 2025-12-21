"""
共识机制模块
实现多种共识算法来处理多专家代理之间的意见分歧
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import numpy as np
from collections import Counter, defaultdict
import statistics
import math
from enum import Enum

from ..tools.logger import setup_logger
from ..config import ComplexityLevel


class ConsensusMethod(Enum):
    """共识方法枚举"""
    SIMPLE_MAJORITY = "simple_majority"
    WEIGHTED_VOTING = "weighted_voting"
    BORDA_COUNT = "borda_count"
    CONDORCET = "condorcet"
    ITERATIVE_CONSENSUS = "iterative_consensus"
    FUZZY_CONSENSUS = "fuzzy_consensus"


class ConsensusResult:
    """共识结果类"""
    
    def __init__(self, consensus_level: float, final_decision: Dict[str, Any], 
                 supporting_agents: List[str], confidence_score: float, success: bool = True):
        self.consensus_level = consensus_level
        self.final_decision = final_decision
        self.supporting_agents = supporting_agents
        self.confidence_score = confidence_score
        self.success = success
        self.details = {
            "consensus_level": consensus_level,
            "supporting_agents_count": len(supporting_agents),
            "confidence_score": confidence_score
        }
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "consensus_level": self.consensus_level,
            "final_decision": self.final_decision,
            "supporting_agents": self.supporting_agents,
            "confidence_score": self.confidence_score,
            "success": self.success,
            "details": self.details,
            "timestamp": self.timestamp
        }


class ConsensusMechanism:
    """共识机制核心类"""
    
    def __init__(self):
        """初始化共识机制"""
        self.logger = setup_logger(self.__class__.__name__)
        
        # 共识阈值配置（必须在初始化算法之前定义）
        self.consensus_thresholds = {
            "unanimous": 1.0,
            "super_majority": 0.8,
            "majority": 0.6,
            "plurality": 0.4,
            "minimum": 0.3
        }
        
        # 专家权重配置
        self.expert_weights = {
            "specialist": 1.0,
            "pcc": 0.7,
            "moderator": 0.8,
            "recruiter": 0.6,
            "reviewer": 0.9,
            "integrator": 1.0
        }
        
        # 共识算法配置（现在可以安全地访问阈值了）
        self.algorithms = self._initialize_algorithms()
    
    def _initialize_algorithms(self) -> Dict[str, Any]:
        """初始化共识算法"""
        return {
            "simple_majority": {
                "name": "简单多数决",
                "description": "选择得票最多的选项",
                "threshold": self.consensus_thresholds["majority"],
                "weight_consideration": False
            },
            "weighted_voting": {
                "name": "加权投票",
                "description": "根据专家权重计算加权投票结果",
                "threshold": self.consensus_thresholds["majority"],
                "weight_consideration": True
            },
            "borda_count": {
                "name": "Borda计数",
                "description": "根据排序位置分配分数",
                "threshold": self.consensus_thresholds["majority"],
                "weight_consideration": True
            },
            "condorcet": {
                "name": "Condorcet方法",
                "description": "两两比较选择最受欢迎的选项",
                "threshold": self.consensus_thresholds["majority"],
                "weight_consideration": True
            },
            "iterative_consensus": {
                "name": "迭代共识",
                "description": "多轮讨论逐步达成共识",
                "threshold": self.consensus_thresholds["super_majority"],
                "weight_consideration": True
            },
            "fuzzy_consensus": {
                "name": "模糊共识",
                "description": "处理模糊和不确定意见的共识",
                "threshold": self.consensus_thresholds["majority"],
                "weight_consideration": True
            }
        }
    
    def achieve_consensus(self, expert_opinions: List[Dict[str, Any]], 
                         algorithm: str = "weighted_voting", 
                         decision_type: str = "diagnosis") -> ConsensusResult:
        """
        达成共识
        
        Args:
            expert_opinions: 专家意见列表
            algorithm: 共识算法
            decision_type: 决策类型（diagnosis/treatment/medication）
            
        Returns:
            共识结果
        """
        try:
            self.logger.info(f"开始使用{algorithm}算法达成{decision_type}共识")
            
            if not expert_opinions:
                raise ValueError("专家意见列表为空")
            
            # 预处理专家意见
            processed_opinions = self._preprocess_opinions(expert_opinions, decision_type)
            
            # 根据算法类型执行共识
            if algorithm == "simple_majority":
                result = self._simple_majority_consensus(processed_opinions)
            elif algorithm == "weighted_voting":
                result = self._weighted_voting_consensus(processed_opinions)
            elif algorithm == "borda_count":
                result = self._borda_count_consensus(processed_opinions)
            elif algorithm == "condorcet":
                result = self._condorcet_consensus(processed_opinions)
            elif algorithm == "iterative_consensus":
                result = self._iterative_consensus(processed_opinions)
            elif algorithm == "fuzzy_consensus":
                result = self._fuzzy_consensus(processed_opinions)
            else:
                raise ValueError(f"未知的共识算法: {algorithm}")
            
            # 验证共识质量
            quality_check = self._validate_consensus_quality(result, processed_opinions)
            result.consensus_level *= quality_check["quality_multiplier"]
            
            self.logger.info(f"共识达成，水平: {result.consensus_level:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"共识达成失败: {str(e)}")
            raise
    
    def _preprocess_opinions(self, expert_opinions: List[Dict[str, Any]], 
                           decision_type: str) -> List[Dict[str, Any]]:
        """预处理专家意见"""
        processed = []
        
        for opinion in expert_opinions:
            processed_opinion = {
                "expert_id": opinion.get("expert_id", "unknown"),
                "expert_type": opinion.get("expert_type", "specialist"),
                "weight": self.expert_weights.get(opinion.get("expert_type", "specialist"), 0.7),
                "confidence": opinion.get("confidence", 0.7),
                "reasoning": opinion.get("reasoning", ""),
                "timestamp": opinion.get("timestamp", datetime.now().isoformat())
            }
            
            # 根据决策类型提取意见
            if decision_type == "diagnosis":
                processed_opinion["diagnoses"] = opinion.get("diagnosis", [])
                processed_opinion["diagnosis_confidence"] = opinion.get("diagnosis_confidence", {})
            elif decision_type == "treatment":
                processed_opinion["treatments"] = opinion.get("treatment", [])
                processed_opinion["treatment_confidence"] = opinion.get("treatment_confidence", {})
            elif decision_type == "medication":
                processed_opinion["medications"] = opinion.get("medications", [])
                processed_opinion["medication_confidence"] = opinion.get("medication_confidence", {})
            
            processed.append(processed_opinion)
        
        return processed
    
    def _simple_majority_consensus(self, processed_opinions: List[Dict[str, Any]]) -> ConsensusResult:
        """简单多数决共识"""
        
        # 收集所有选项
        all_options = []
        for opinion in processed_opinions:
            if "diagnoses" in opinion:
                all_options.extend(opinion["diagnoses"])
            elif "treatments" in opinion:
                all_options.extend(opinion["treatments"])
            elif "medications" in opinion:
                all_options.extend(opinion["medications"])
        
        if not all_options:
            return ConsensusResult(0.0, {}, [], 0.0, success=False)
        
        # 计算得票
        option_counts = Counter(all_options)
        total_votes = len(processed_opinions)
        
        # 选择得票最多的选项
        winning_option = option_counts.most_common(1)[0]
        option, votes = winning_option
        
        consensus_level = votes / total_votes
        supporting_agents = [op["expert_id"] for op in processed_opinions 
                           if option in op.get("diagnoses", []) or 
                              option in op.get("treatments", []) or 
                              option in op.get("medications", [])]
        
        confidence_score = votes / total_votes
        
        final_decision = {option: {"votes": votes, "percentage": consensus_level}}
        
        return ConsensusResult(consensus_level, final_decision, supporting_agents, confidence_score, success=True)
    
    def _weighted_voting_consensus(self, processed_opinions: List[Dict[str, Any]]) -> ConsensusResult:
        """加权投票共识"""
        
        # 收集所有选项和权重
        option_scores = defaultdict(float)
        option_votes = defaultdict(int)
        total_weight = sum(opinion["weight"] for opinion in processed_opinions)
        
        for opinion in processed_opinions:
            weight = opinion["weight"]
            confidence = opinion["confidence"]
            
            # 获取选项列表
            options = []
            if "diagnoses" in opinion:
                options = opinion["diagnoses"]
            elif "treatments" in opinion:
                options = opinion["treatments"]
            elif "medications" in opinion:
                options = opinion["medications"]
            
            for option in options:
                option_scores[option] += weight * confidence
                option_votes[option] += 1
        
        if not option_scores:
            return ConsensusResult(0.0, {}, [], 0.0, success=False)
        
        # 选择得分最高的选项
        winning_option = max(option_scores.items(), key=lambda x: x[1])
        option, score = winning_option
        
        consensus_level = score / total_weight
        supporting_agents = [op["expert_id"] for op in processed_opinions 
                           if option in op.get("diagnoses", []) or 
                              option in op.get("treatments", []) or 
                              option in op.get("medications", [])]
        
        confidence_score = option_votes[option] / len(processed_opinions)
        
        final_decision = {
            option: {
                "weighted_score": score,
                "raw_votes": option_votes[option],
                "weight_percentage": consensus_level
            }
        }
        
        return ConsensusResult(consensus_level, final_decision, supporting_agents, confidence_score, success=True)
    
    def _borda_count_consensus(self, processed_opinions: List[Dict[str, Any]]) -> ConsensusResult:
        """Borda计数共识"""
        
        # 收集所有选项
        all_options = set()
        for opinion in processed_opinions:
            if "diagnoses" in opinion:
                all_options.update(opinion["diagnoses"])
            elif "treatments" in opinion:
                all_options.update(opinion["treatments"])
            elif "medications" in opinion:
                all_options.update(opinion["medications"])
        
        if not all_options:
            return ConsensusResult(0.0, {}, [], 0.0, success=False)
        
        option_borda_scores = defaultdict(float)
        
        for opinion in processed_opinions:
            weight = opinion["weight"]
            confidence = opinion["confidence"]
            
            # 获取选项列表并排序
            options = []
            if "diagnoses" in opinion:
                options = opinion["diagnoses"]
            elif "treatments" in opinion:
                options = opinion["treatments"]
            elif "medications" in opinion:
                options = opinion["medications"]
            
            # 为每个选项分配Borda分数
            for i, option in enumerate(options):
                borda_score = len(options) - i - 1  # 排名越高分数越高
                option_borda_scores[option] += borda_score * weight * confidence
        
        # 选择得分最高的选项
        winning_option = max(option_borda_scores.items(), key=lambda x: x[1])
        option, score = winning_option
        
        max_possible_score = sum(opinion["weight"] * opinion["confidence"] 
                               for opinion in processed_opinions) * max(len(options) for options in [
                                   opinion.get("diagnoses", []) or opinion.get("treatments", []) or opinion.get("medications", [])
                                   for opinion in processed_opinions
                               ])
        
        consensus_level = score / max_possible_score if max_possible_score > 0 else 0.0
        
        supporting_agents = [op["expert_id"] for op in processed_opinions 
                           if option in op.get("diagnoses", []) or 
                              option in op.get("treatments", []) or 
                              option in op.get("medications", [])]
        
        confidence_score = len(supporting_agents) / len(processed_opinions)
        
        final_decision = {
            option: {
                "borda_score": score,
                "supporting_count": len(supporting_agents),
                "supporting_percentage": confidence_score
            }
        }
        
        return ConsensusResult(consensus_level, final_decision, supporting_agents, confidence_score, success=True)
    
    def _condorcet_consensus(self, processed_opinions: List[Dict[str, Any]]) -> ConsensusResult:
        """Condorcet共识"""
        
        # 收集所有选项
        all_options = set()
        for opinion in processed_opinions:
            if "diagnoses" in opinion:
                all_options.update(opinion["diagnoses"])
            elif "treatments" in opinion:
                all_options.update(opinion["treatments"])
            elif "medications" in opinion:
                all_options.update(opinion["medications"])
        
        if len(all_options) < 2:
            # 如果只有一个选项，直接选择
            option = list(all_options)[0] if all_options else None
            if option:
                supporting_agents = [op["expert_id"] for op in processed_opinions 
                                   if option in op.get("diagnoses", []) or 
                                      option in op.get("treatments", []) or 
                                      option in op.get("medications", [])]
                return ConsensusResult(1.0, {option: {"votes": len(supporting_agents)}}, 
                                     supporting_agents, 1.0, success=True)
            else:
                return ConsensusResult(0.0, {}, [], 0.0, success=False)
        
        # 构建成对比较矩阵
        pairwise_wins = defaultdict(int)
        total_weight = sum(opinion["weight"] for opinion in processed_opinions)
        
        option_list = list(all_options)
        
        for i, option1 in enumerate(option_list):
            for j, option2 in enumerate(option_list):
                if i >= j:  # 避免重复比较
                    continue
                
                option1_score = 0
                option2_score = 0
                
                for opinion in processed_opinions:
                    weight = opinion["weight"]
                    confidence = opinion["confidence"]
                    
                    # 检查专家对两个选项的偏好
                    options = []
                    if "diagnoses" in opinion:
                        options = opinion["diagnoses"]
                    elif "treatments" in opinion:
                        options = opinion["treatments"]
                    elif "medications" in opinion:
                        options = opinion["medications"]
                    
                    if option1 in options and option2 not in options:
                        option1_score += weight * confidence
                    elif option2 in options and option1 not in options:
                        option2_score += weight * confidence
                    elif option1 in options and option2 in options:
                        # 如果两个选项都在列表中，根据位置决定偏好
                        if options.index(option1) < options.index(option2):
                            option1_score += weight * confidence
                        else:
                            option2_score += weight * confidence
                
                # 记录胜者
                if option1_score > option2_score:
                    pairwise_wins[option1] += 1
                elif option2_score > option1_score:
                    pairwise_wins[option2] += 1
                # 平局不记录
        
        if not pairwise_wins:
            # 没有明确的胜者，回退到加权投票
            return self._weighted_voting_consensus(processed_opinions)
        
        # 选择赢得最多成对比较的选项
        winning_option = max(pairwise_wins.items(), key=lambda x: x[1])
        option, wins = winning_option
        
        consensus_level = wins / (len(all_options) - 1)
        supporting_agents = [op["expert_id"] for op in processed_opinions 
                           if option in op.get("diagnoses", []) or 
                              option in op.get("treatments", []) or 
                              option in op.get("medications", [])]
        
        confidence_score = len(supporting_agents) / len(processed_opinions)
        
        final_decision = {
            option: {
                "pairwise_wins": wins,
                "total_opponents": len(all_options) - 1,
                "supporting_count": len(supporting_agents)
            }
        }
        
        return ConsensusResult(consensus_level, final_decision, supporting_agents, confidence_score, success=True)
    
    def _iterative_consensus(self, processed_opinions: List[Dict[str, Any]]) -> ConsensusResult:
        """迭代共识"""
        
        max_iterations = 3
        current_opinions = processed_opinions.copy()
        
        for iteration in range(max_iterations):
            self.logger.info(f"迭代共识第{iteration + 1}轮")
            
            # 执行加权投票
            result = self._weighted_voting_consensus(current_opinions)
            
            # 检查是否达成足够的共识
            if result.consensus_level >= self.consensus_thresholds["super_majority"]:
                self.logger.info(f"在第{iteration + 1}轮达成充分共识")
                return result
            
            # 识别分歧较大的专家
            outliers = self._identify_outliers(current_opinions, result)
            
            if outliers:
                # 移除异常值，重新计算
                current_opinions = [op for op in current_opinions 
                                  if op["expert_id"] not in outliers]
                self.logger.info(f"移除{len(outliers)}个异常专家意见")
            else:
                # 没有明显的异常值，但共识不够，尝试模糊共识
                return self._fuzzy_consensus(current_opinions)
            
            # 如果剩余专家太少，停止迭代
            if len(current_opinions) < 2:
                break
        
        # 最后一轮尝试
        if current_opinions:
            return self._weighted_voting_consensus(current_opinions)
        else:
            return ConsensusResult(0.0, {}, [], 0.0, success=False)
    
    def _identify_outliers(self, opinions: List[Dict[str, Any]], 
                          consensus_result: ConsensusResult) -> List[str]:
        """识别异常专家意见"""
        if not opinions:
            return []
        
        # 计算每个专家与共识的距离
        expert_distances = []
        winning_option = list(consensus_result.final_decision.keys())[0] if consensus_result.final_decision else None
        
        if not winning_option:
            return []
        
        for opinion in opinions:
            distance = 0
            if winning_option not in opinion.get("diagnoses", []) and \
               winning_option not in opinion.get("treatments", []) and \
               winning_option not in opinion.get("medications", []):
                distance = 1.0  # 完全不支持共识选项
            
            expert_distances.append((opinion["expert_id"], distance))
        
        # 识别距离大于阈值的异常值
        distances = [d[1] for d in expert_distances]
        if not distances:
            return []
        
        threshold = statistics.mean(distances) + statistics.stdev(distances) if len(distances) > 1 else 0.5
        
        outliers = [expert_id for expert_id, distance in expert_distances 
                   if distance > threshold]
        
        return outliers
    
    def _fuzzy_consensus(self, processed_opinions: List[Dict[str, Any]]) -> ConsensusResult:
        """模糊共识"""
        
        # 收集所有选项
        all_options = set()
        for opinion in processed_opinions:
            if "diagnoses" in opinion:
                all_options.update(opinion["diagnoses"])
            elif "treatments" in opinion:
                all_options.update(opinion["treatments"])
            elif "medications" in opinion:
                all_options.update(opinion["medications"])
        
        if not all_options:
            return ConsensusResult(0.0, {}, [], 0.0, success=False)
        
        # 计算模糊支持度
        fuzzy_scores = defaultdict(float)
        
        for opinion in processed_opinions:
            weight = opinion["weight"]
            confidence = opinion["confidence"]
            
            # 获取选项列表
            options = []
            if "diagnoses" in opinion:
                options = opinion["diagnoses"]
            elif "treatments" in opinion:
                options = opinion["treatments"]
            elif "medications" in opinion:
                options = opinion["medications"]
            
            # 计算每个选项的模糊支持度
            for i, option in enumerate(options):
                # 位置权重：越靠前权重越高
                position_weight = (len(options) - i) / len(options)
                fuzzy_scores[option] += weight * confidence * position_weight
        
        # 选择模糊支持度最高的选项
        if not fuzzy_scores:
            return ConsensusResult(0.0, {}, [], 0.0, success=False)
        
        winning_option = max(fuzzy_scores.items(), key=lambda x: x[1])
        option, fuzzy_score = winning_option
        
        # 归一化模糊分数
        total_fuzzy_score = sum(fuzzy_scores.values())
        consensus_level = fuzzy_score / total_fuzzy_score if total_fuzzy_score > 0 else 0.0
        
        supporting_agents = [op["expert_id"] for op in processed_opinions 
                           if option in op.get("diagnoses", []) or 
                              option in op.get("treatments", []) or 
                              option in op.get("medications", [])]
        
        confidence_score = len(supporting_agents) / len(processed_opinions)
        
        final_decision = {
            option: {
                "fuzzy_score": fuzzy_score,
                "normalized_score": consensus_level,
                "supporting_count": len(supporting_agents)
            }
        }
        
        return ConsensusResult(consensus_level, final_decision, supporting_agents, confidence_score)
    
    def _validate_consensus_quality(self, result: ConsensusResult, 
                                  processed_opinions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """验证共识质量"""
        
        quality_factors = {
            "support_diversity": 0.0,
            "confidence_consistency": 0.0,
            "reasoning_quality": 0.0
        }
        
        # 支持多样性评估
        if result.supporting_agents:
            expert_types = [op["expert_type"] for op in processed_opinions 
                          if op["expert_id"] in result.supporting_agents]
            unique_types = len(set(expert_types))
            total_types = len(set(op["expert_type"] for op in processed_opinions))
            quality_factors["support_diversity"] = unique_types / total_types if total_types > 0 else 0.0
        
        # 信心一致性评估
        supporting_confidences = [op["confidence"] for op in processed_opinions 
                                if op["expert_id"] in result.supporting_agents]
        if supporting_confidences:
            avg_confidence = statistics.mean(supporting_confidences)
            confidence_variance = statistics.variance(supporting_confidences) if len(supporting_confidences) > 1 else 0
            quality_factors["confidence_consistency"] = avg_confidence * (1 - confidence_variance)
        
        # 推理质量评估（基于推理文本长度和结构）
        reasoning_texts = [op["reasoning"] for op in processed_opinions 
                         if op["expert_id"] in result.supporting_agents and op["reasoning"]]
        if reasoning_texts:
            avg_length = statistics.mean([len(text) for text in reasoning_texts])
            quality_factors["reasoning_quality"] = min(avg_length / 200, 1.0)  # 标准化到0-1
        
        # 计算综合质量分数
        overall_quality = statistics.mean(list(quality_factors.values()))
        
        # 质量调整因子
        quality_multiplier = 0.8 + (overall_quality * 0.4)  # 范围：0.8-1.2
        
        return {
            "quality_factors": quality_factors,
            "overall_quality": overall_quality,
            "quality_multiplier": quality_multiplier
        }
    
    def get_consensus_statistics(self, historical_results: List[ConsensusResult]) -> Dict[str, Any]:
        """获取共识统计信息"""
        if not historical_results:
            return {"message": "暂无共识历史记录"}
        
        consensus_levels = [result.consensus_level for result in historical_results]
        confidence_scores = [result.confidence_score for result in historical_results]
        
        # 统计不同共识水平的分布
        consensus_distribution = {
            "unanimous": len([level for level in consensus_levels if level >= 0.95]),
            "super_majority": len([level for level in consensus_levels if 0.8 <= level < 0.95]),
            "majority": len([level for level in consensus_levels if 0.6 <= level < 0.8]),
            "plurality": len([level for level in consensus_levels if 0.4 <= level < 0.6]),
            "weak": len([level for level in consensus_levels if level < 0.4])
        }
        
        return {
            "total_consensus": len(historical_results),
            "average_consensus_level": statistics.mean(consensus_levels),
            "average_confidence": statistics.mean(confidence_scores),
            "consensus_distribution": consensus_distribution,
            "consistency_metrics": {
                "consensus_std": statistics.stdev(consensus_levels) if len(consensus_levels) > 1 else 0,
                "confidence_std": statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0
            }
        }


class ConsensusOrchestrator:
    """共识协调器"""
    
    def __init__(self):
        """初始化共识协调器"""
        self.logger = setup_logger(self.__class__.__name__)
        self.consensus_mechanism = ConsensusMechanism()
        self.consensus_history = []
    
    def coordinate_multi_level_consensus(self, expert_opinions: List[Dict[str, Any]], 
                                       complexity_level: ComplexityLevel) -> Dict[str, Any]:
        """
        协调多层次共识
        
        Args:
            expert_opinions: 专家意见
            complexity_level: 复杂度等级
            
        Returns:
            多层次共识结果
        """
        try:
            self.logger.info(f"开始多层次共识协调，复杂度: {complexity_level.value}")
            
            # 根据复杂度选择共识策略
            if complexity_level == ComplexityLevel.SIMPLE:
                strategies = ["simple_majority", "weighted_voting"]
            elif complexity_level == ComplexityLevel.MODERATE:
                strategies = ["weighted_voting", "borda_count", "condorcet"]
            else:  # HIGH
                strategies = ["iterative_consensus", "fuzzy_consensus", "condorcet"]
            
            consensus_results = {}
            
            # 对每种决策类型执行共识
            decision_types = ["diagnosis", "treatment", "medication"]
            
            for decision_type in decision_types:
                self.logger.info(f"执行{decision_type}共识")
                
                # 选择最适合的算法
                algorithm = self._select_algorithm_for_type(strategies, decision_type)
                
                result = self.consensus_mechanism.achieve_consensus(
                    expert_opinions, algorithm, decision_type
                )
                
                consensus_results[decision_type] = result
            
            # 计算整体共识水平
            overall_consensus = self._calculate_overall_consensus(consensus_results)
            
            # 生成综合建议
            integrated_recommendation = self._generate_integrated_recommendation(consensus_results)
            
            final_result = {
                "individual_consensus": {k: v.to_dict() for k, v in consensus_results.items()},
                "overall_consensus": overall_consensus,
                "integrated_recommendation": integrated_recommendation,
                "consensus_strategy": strategies,
                "complexity_level": complexity_level.value,
                "timestamp": datetime.now().isoformat()
            }
            
            # 记录共识历史
            self.consensus_history.append(final_result)
            
            self.logger.info("多层次共识协调完成")
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"多层次共识协调失败: {str(e)}")
            raise
    
    def _select_algorithm_for_type(self, strategies: List[str], decision_type: str) -> str:
        """为特定决策类型选择算法"""
        
        algorithm_preferences = {
            "diagnosis": ["condorcet", "weighted_voting", "iterative_consensus"],
            "treatment": ["weighted_voting", "borda_count", "fuzzy_consensus"],
            "medication": ["weighted_voting", "simple_majority", "borda_count"]
        }
        
        preferred_algorithms = algorithm_preferences.get(decision_type, strategies)
        
        for algorithm in preferred_algorithms:
            if algorithm in strategies:
                return algorithm
        
        # 如果没有匹配的偏好算法，返回第一个可用策略
        return strategies[0] if strategies else "weighted_voting"
    
    def _calculate_overall_consensus(self, consensus_results: Dict[str, ConsensusResult]) -> Dict[str, Any]:
        """计算整体共识水平"""
        
        consensus_levels = [result.consensus_level for result in consensus_results.values()]
        confidence_scores = [result.confidence_score for result in consensus_results.values()]
        
        # 加权平均共识水平
        weighted_consensus = statistics.mean(consensus_levels)
        
        # 整体信心分数
        overall_confidence = statistics.mean(confidence_scores)
        
        # 共识一致性
        consensus_consistency = 1.0 - (statistics.stdev(consensus_levels) if len(consensus_levels) > 1 else 0)
        
        return {
            "overall_consensus_level": weighted_consensus,
            "overall_confidence": overall_confidence,
            "consensus_consistency": consensus_consistency,
            "consensus_quality": (weighted_consensus + overall_confidence + consensus_consistency) / 3.0
        }
    
    def _generate_integrated_recommendation(self, consensus_results: Dict[str, ConsensusResult]) -> Dict[str, Any]:
        """生成综合建议"""
        
        integrated = {
            "primary_diagnosis": None,
            "differential_diagnoses": [],
            "primary_treatment": None,
            "alternative_treatments": [],
            "medications": [],
            "confidence_assessment": {},
            "risk_warnings": [],
            "follow_up_plan": ""
        }
        
        # 整合诊断结果
        if "diagnosis" in consensus_results and consensus_results["diagnosis"].final_decision:
            diagnosis_result = consensus_results["diagnosis"]
            primary_diag = list(diagnosis_result.final_decision.keys())[0]
            integrated["primary_diagnosis"] = {
                "diagnosis": primary_diag,
                "confidence": diagnosis_result.consensus_level,
                "supporting_experts": len(diagnosis_result.supporting_agents)
            }
            
            # 收集其他诊断选项
            for diag, info in diagnosis_result.final_decision.items():
                if diag != primary_diag:
                    integrated["differential_diagnoses"].append({
                        "diagnosis": diag,
                        "support_level": info.get("percentage", 0)
                    })
        
        # 整合治疗结果
        if "treatment" in consensus_results and consensus_results["treatment"].final_decision:
            treatment_result = consensus_results["treatment"]
            primary_treat = list(treatment_result.final_decision.keys())[0]
            integrated["primary_treatment"] = {
                "treatment": primary_treat,
                "confidence": treatment_result.consensus_level,
                "supporting_experts": len(treatment_result.supporting_agents)
            }
        
        # 整合用药结果
        if "medication" in consensus_results and consensus_results["medication"].final_decision:
            medication_result = consensus_results["medication"]
            for med, info in medication_result.final_decision.items():
                integrated["medications"].append({
                    "medication": med,
                    "confidence": info.get("weight_percentage", 0),
                    "supporting_experts": medication_result.supporting_agents
                })
        
        # 信心评估
        integrated["confidence_assessment"] = {
            "diagnostic_confidence": consensus_results.get("diagnosis", ConsensusResult(0, {}, [], 0)).confidence_score,
            "treatment_confidence": consensus_results.get("treatment", ConsensusResult(0, {}, [], 0)).confidence_score,
            "medication_confidence": consensus_results.get("medication", ConsensusResult(0, {}, [], 0)).confidence_score
        }
        
        # 风险警告
        if any(result.consensus_level < 0.6 for result in consensus_results.values()):
            integrated["risk_warnings"].append("专家意见存在分歧，建议进一步确认")
        
        if any(result.confidence_score < 0.5 for result in consensus_results.values()):
            integrated["risk_warnings"].append("部分建议信心不足，需要额外验证")
        
        return integrated
    
    def get_orchestrator_statistics(self) -> Dict[str, Any]:
        """获取协调器统计信息"""
        if not self.consensus_history:
            return {"message": "暂无协调历史记录"}
        
        # 统计各种复杂度等级的共识情况
        complexity_stats = defaultdict(list)
        for record in self.consensus_history:
            complexity = record["complexity_level"]
            overall_consensus = record["overall_consensus"]["overall_consensus_level"]
            complexity_stats[complexity].append(overall_consensus)
        
        return {
            "total_orchestrations": len(self.consensus_history),
            "complexity_distribution": dict(complexity_stats),
            "average_overall_consensus": statistics.mean([
                record["overall_consensus"]["overall_consensus_level"] 
                for record in self.consensus_history
            ]),
            "success_rate": len([
                record for record in self.consensus_history 
                if record["overall_consensus"]["overall_consensus_level"] >= 0.6
            ]) / len(self.consensus_history) * 100
        }