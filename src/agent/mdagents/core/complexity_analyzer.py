"""
复杂度评估模块 - 根据医疗查询的特征评估复杂度
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from ..config import ComplexityLevel, ComplexityThresholds
from ..tools.logger import setup_logger

@dataclass
class ComplexityFeatures:
    """复杂度特征数据结构"""
    text_length: int
    symptom_count: int
    medical_terms_count: int
    specialty_count: int
    chronic_conditions_mentioned: bool
    emergency_keywords: List[str]
    diagnostic_complexity: float
    treatment_complexity: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "text_length": self.text_length,
            "symptom_count": self.symptom_count,
            "medical_terms_count": self.medical_terms_count,
            "specialty_count": self.specialty_count,
            "chronic_conditions_mentioned": self.chronic_conditions_mentioned,
            "emergency_keywords": self.emergency_keywords,
            "diagnostic_complexity": self.diagnostic_complexity,
            "treatment_complexity": self.treatment_complexity
        }

class ComplexityAnalyzer:
    """医疗查询复杂度分析器"""
    
    def __init__(self, thresholds: ComplexityThresholds = None):
        """
        初始化复杂度分析器
        
        Args:
            thresholds: 复杂度阈值配置
        """
        self.thresholds = thresholds or ComplexityThresholds()
        self.logger = setup_logger(self.__class__.__name__)
        
        # 医疗专科关键词
        self.specialty_keywords = {
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
        
        # 急症关键词
        self.emergency_keywords = [
            '紧急', '急救', '急诊', '危重', '休克', '昏迷', '呼吸困难',
            '胸痛', '腹痛', '头痛', '出血', '中毒', '过敏反应'
        ]
        
        # 慢性病关键词
        self.chronic_keywords = [
            '糖尿病', '高血压', '心脏病', '肾病', '肝病', '肺病',
            '关节炎', '风湿', '哮喘', '癫痫', '精神病'
        ]
        
        # 复杂医疗术语
        self.complex_medical_terms = [
            '病理', '生理', '免疫', '基因', '分子', '细胞', '组织',
            '器官', '系统', '综合征', '并发症', '继发性', '原发性'
        ]
    
    def extract_features(self, query: str, context: Dict[str, Any] = None) -> ComplexityFeatures:
        """
        提取复杂度特征
        
        Args:
            query: 医疗查询文本
            context: 上下文信息
            
        Returns:
            复杂度特征对象
        """
        query = query.lower().strip()
        
        # 基础文本特征
        text_length = len(query)
        
        # 症状计数
        symptom_count = self._count_symptoms(query)
        
        # 医疗术语计数
        medical_terms_count = self._count_medical_terms(query)
        
        # 专科计数
        specialty_count = self._count_specialties(query)
        
        # 慢性病提及
        chronic_conditions_mentioned = self._check_chronic_conditions(query)
        
        # 急症关键词
        emergency_keywords = [kw for kw in self.emergency_keywords if kw in query]
        
        # 诊断复杂度评估
        diagnostic_complexity = self._assess_diagnostic_complexity(query, context)
        
        # 治疗复杂度评估
        treatment_complexity = self._assess_treatment_complexity(query, context)
        
        return ComplexityFeatures(
            text_length=text_length,
            symptom_count=symptom_count,
            medical_terms_count=medical_terms_count,
            specialty_count=specialty_count,
            chronic_conditions_mentioned=chronic_conditions_mentioned,
            emergency_keywords=emergency_keywords,
            diagnostic_complexity=diagnostic_complexity,
            treatment_complexity=treatment_complexity
        )
    
    def _count_symptoms(self, query: str) -> int:
        """计算症状相关词汇数量"""
        symptom_keywords = [
            '疼痛', '发热', '咳嗽', '头痛', '头晕', '恶心', '呕吐',
            '腹泻', '便秘', '失眠', '疲劳', '气短', '胸痛', '腹痛'
        ]
        return sum(1 for keyword in symptom_keywords if keyword in query)
    
    def _count_medical_terms(self, query: str) -> int:
        """计算医疗术语数量"""
        count = 0
        for term in self.complex_medical_terms:
            if term in query:
                count += 1
        return count
    
    def _count_specialties(self, query: str) -> int:
        """计算涉及的医疗专科数量"""
        count = 0
        for specialty, keywords in self.specialty_keywords.items():
            if any(keyword in query for keyword in keywords):
                count += 1
        return count
    
    def _check_chronic_conditions(self, query: str) -> bool:
        """检查是否提及慢性病"""
        return any(keyword in query for keyword in self.chronic_keywords)
    
    def _assess_diagnostic_complexity(self, query: str, context: Dict[str, Any] = None) -> float:
        """评估诊断复杂度 (0-1)"""
        complexity_score = 0.0
        
        # 多个症状增加复杂度
        symptom_count = self._count_symptoms(query)
        if symptom_count > 3:
            complexity_score += 0.3
        elif symptom_count > 1:
            complexity_score += 0.15
        
        # 多个专科增加复杂度
        specialty_count = self._count_specialties(query)
        if specialty_count > 2:
            complexity_score += 0.3
        elif specialty_count > 1:
            complexity_score += 0.15
        
        # 复杂医疗术语
        medical_terms = self._count_medical_terms(query)
        complexity_score += min(medical_terms * 0.1, 0.2)
        
        # 模糊描述增加复杂度
        if any(word in query for word in ['不明', '不清楚', '可能', '疑似', '大概']):
            complexity_score += 0.1
        
        return min(complexity_score, 1.0)
    
    def _assess_treatment_complexity(self, query: str, context: Dict[str, Any] = None) -> float:
        """评估治疗复杂度 (0-1)"""
        complexity_score = 0.0
        
        # 药物治疗复杂度
        if any(word in query for word in ['药物', '治疗', '手术', '疗法']):
            complexity_score += 0.2
        
        # 多药物相互作用
        if any(word in query for word in ['药物', '药物相互作用', '副作用']):
            complexity_score += 0.2
        
        # 慢性病管理
        if self._check_chronic_conditions(query):
            complexity_score += 0.3
        
        # 急症处理 - 调整分值以避免过度增加复杂度
        if any(keyword in query for keyword in self.emergency_keywords):
            complexity_score += 0.25
        
        # 利用上下文信息调整复杂度分数
        if context:
            # 如果有患者历史信息，增加复杂度
            if context.get('has_medical_history', False):
                complexity_score += 0.1
            
            # 如果有多种药物，增加复杂度
            if context.get('current_medications_count', 0) > 3:
                complexity_score += 0.15
            
            # 如果有并发症，增加复杂度
            if context.get('has_complications', False):
                complexity_score += 0.2
        
        return min(complexity_score, 1.0)
    
    def calculate_complexity_score(self, features: ComplexityFeatures) -> float:
        """
        计算综合复杂度分数
        
        Args:
            features: 复杂度特征
            
        Returns:
            复杂度分数 (0-1)
        """
        # 权重配置
        weights = {
            'text_length': 0.1,
            'symptom_count': 0.2,
            'medical_terms_count': 0.15,
            'specialty_count': 0.25,
            'chronic_conditions': 0.15,
            'diagnostic_complexity': 0.1,
            'treatment_complexity': 0.05
        }
        
        # 归一化文本长度分数
        text_length_score = min(features.text_length / 1000, 1.0)
        
        # 归一化症状数量分数
        symptom_score = min(features.symptom_count / 5, 1.0)
        
        # 归一化医疗术语分数
        medical_terms_score = min(features.medical_terms_count / 10, 1.0)
        
        # 归一化专科数量分数
        specialty_score = min(features.specialty_count / 5, 1.0)
        
        # 慢性病分数
        chronic_score = 1.0 if features.chronic_conditions_mentioned else 0.0
        
        # 计算加权总分
        total_score = (
            weights['text_length'] * text_length_score +
            weights['symptom_count'] * symptom_score +
            weights['medical_terms_count'] * medical_terms_score +
            weights['specialty_count'] * specialty_score +
            weights['chronic_conditions'] * chronic_score +
            weights['diagnostic_complexity'] * features.diagnostic_complexity +
            weights['treatment_complexity'] * features.treatment_complexity
        )
        
        return min(total_score, 1.0)
    
    def analyze_complexity(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        完整的复杂度分析
        
        Args:
            query: 医疗查询文本
            context: 上下文信息
            
        Returns:
            复杂度分析结果
        """
        self.logger.info(f"开始分析查询复杂度: {query[:100]}...")
        
        # 提取特征
        features = self.extract_features(query, context)
        
        # 计算复杂度分数
        complexity_score = self.calculate_complexity_score(features)
        
        # 确定复杂度等级
        level = self.score_to_level(complexity_score)
        
        # 生成解释
        explanation = self._generate_explanation(features, complexity_score, level)
        
        # 估计处理时间
        estimated_time = self._estimate_processing_time(level)
        
        result = {
            'complexity_score': complexity_score,
            'complexity_level': level,
            'features': features.to_dict(),
            'explanation': explanation,
            'estimated_time_seconds': estimated_time,
            'recommendation': self._get_recommendation(level)
        }
        
        self.logger.info(f"复杂度分析完成: {level.value}, 分数: {complexity_score:.3f}")
        return result
    
    def score_to_level(self, score: float) -> ComplexityLevel:
        """将分数转换为复杂度等级"""
        if score < self.thresholds.simple_max_score:
            return ComplexityLevel.SIMPLE
        elif score < self.thresholds.moderate_max_score:
            return ComplexityLevel.MODERATE
        else:
            return ComplexityLevel.COMPLEX
    
    def _generate_explanation(self, features: ComplexityFeatures, score: float, level: ComplexityLevel) -> List[str]:
        """生成复杂度解释"""
        explanations = []
        
        if features.text_length > 500:
            explanations.append(f"查询信息详细（{features.text_length}字符），复杂度较高")
        
        if features.symptom_count > 3:
            explanations.append(f"涉及多个症状（{features.symptom_count}个），需要综合分析")
        
        if features.specialty_count > 1:
            explanations.append(f"涉及多个医疗专科（{features.specialty_count}个）")
        
        if features.chronic_conditions_mentioned:
            explanations.append("涉及慢性病管理")
        
        if features.diagnostic_complexity > 0.5:
            explanations.append("诊断过程较为复杂")
        
        if features.treatment_complexity > 0.5:
            explanations.append("治疗方案较为复杂")
        
        if level == ComplexityLevel.COMPLEX:
            explanations.append("建议采用多学科团队协作模式")
        elif level == ComplexityLevel.MODERATE:
            explanations.append("建议采用中等协作模式")
        else:
            explanations.append("适合快速单一专家处理")
        
        return explanations
    
    def _estimate_processing_time(self, level: ComplexityLevel) -> float:
        """估计处理时间（秒）"""
        time_mapping = {
            ComplexityLevel.SIMPLE: 14.7,
            ComplexityLevel.MODERATE: 95.5,
            ComplexityLevel.COMPLEX: 226.0
        }
        return time_mapping[level]
    
    def _get_recommendation(self, level: ComplexityLevel) -> str:
        """获取处理建议"""
        recommendations = {
            ComplexityLevel.SIMPLE: "采用单专家快速处理模式，使用基础诊断和常见治疗方案",
            ComplexityLevel.MODERATE: "采用多学科团队协作模式，集体讨论并多数投票决策",
            ComplexityLevel.COMPLEX: "采用分层并行处理模式，分阶段报告并最终综合审查"
        }
        return recommendations[level]