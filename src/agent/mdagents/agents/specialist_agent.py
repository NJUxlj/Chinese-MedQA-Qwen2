"""
专科医生智能体
"""

from typing import Dict, List, Any, Optional
import json

from .base_agent import BaseAgent
from ..config import AgentConfig, AgentType

class SpecialistAgent(BaseAgent):
    """专科医生智能体"""
    
    def __init__(self, config: AgentConfig, specialty: str = None, api_model=None, llm_config=None):
        """
        初始化专科医生
        
        Args:
            config: 智能体配置
            specialty: 专科领域
            api_model: API模型实例
            llm_config: LLM配置实例
        """
        super().__init__(config, api_model, llm_config)
        
        self.specialty = specialty or "通用专科"
        
        # 专科知识库
        self.specialty_knowledge = self._initialize_specialty_knowledge()
        
        # 专科特定的能力
        self.specialty_capabilities = [
            f"{self.specialty}专业诊断",
            "复杂病例分析", 
            "专业治疗建议",
            "专科检查解读"
        ]
        
        self.logger.info(f"专科医生 {self.name} 初始化完成，专业领域: {self.specialty}")
    
    def _initialize_specialty_knowledge(self) -> Dict[str, Any]:
        """初始化专科知识库"""
        knowledge_bases = {
            "心血管科": {
                "common_conditions": ["冠心病", "高血压", "心律失常", "心力衰竭"],
                "key_symptoms": ["胸痛", "气短", "心悸", "水肿", "晕厥"],
                "diagnostic_tests": ["心电图", "超声心动图", "冠脉造影", "动态心电图"],
                "treatment_approaches": ["药物治疗", "介入治疗", "外科手术", "康复治疗"]
            },
            "神经科": {
                "common_conditions": ["脑卒中", "癫痫", "帕金森病", "阿尔茨海默病"],
                "key_symptoms": ["头痛", "头晕", "肢体无力", "语言障碍", "意识障碍"],
                "diagnostic_tests": ["头颅CT", "头颅MRI", "脑电图", "神经电生理"],
                "treatment_approaches": ["药物治疗", "康复训练", "手术治疗", "神经调控"]
            },
            "肿瘤科": {
                "common_conditions": ["肺癌", "乳腺癌", "胃癌", "肝癌", "结直肠癌"],
                "key_symptoms": ["肿块", "疼痛", "体重下降", "疲劳", "食欲不振"],
                "diagnostic_tests": ["肿瘤标志物", "影像学检查", "病理活检", "基因检测"],
                "treatment_approaches": ["手术", "化疗", "放疗", "靶向治疗", "免疫治疗"]
            },
            "内分泌科": {
                "common_conditions": ["糖尿病", "甲状腺疾病", "骨质疏松", "肾上腺疾病"],
                "key_symptoms": ["多饮多尿", "体重变化", "情绪异常", "生长发育异常"],
                "diagnostic_tests": ["血糖", "糖化血红蛋白", "甲状腺功能", "骨密度"],
                "treatment_approaches": ["药物治疗", "胰岛素治疗", "生活方式干预", "手术治疗"]
            }
        }
        
        return knowledge_bases.get(self.specialty, {
            "common_conditions": ["常见疾病"],
            "key_symptoms": ["典型症状"],
            "diagnostic_tests": ["相关检查"],
            "treatment_approaches": ["标准治疗"]
        })
    
    def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        处理专科医疗查询
        
        Args:
            query: 医疗查询
            context: 上下文信息
            
        Returns:
            处理结果
        """
        self.log_interaction("specialist_query_received", f"收到{self.specialty}专科查询: {query[:100]}...")
        
        try:
            # 构建专科诊断提示词
            specialty_prompt = self._build_specialty_prompt(query, context)
            
            # 生成专科建议
            specialty_response = self.generate_response(specialty_prompt)
            
            # 解析响应
            specialty_result = self._parse_specialty_response(specialty_response)
            
            # 生成专科治疗方案
            treatment_plan = self._build_specialty_treatment_plan(specialty_result, context)
            
            # 评估预后
            prognosis = self._assess_prognosis(specialty_result, query)
            
            result = {
                "agent_type": self.agent_type.value if hasattr(self.agent_type, 'value') else self.agent_type,
                "agent_name": self.name,
                "specialty": self.specialty,
                "query": query,
                "specialist_assessment": specialty_result,
                "treatment_plan": treatment_plan,
                "prognosis": prognosis,
                "follow_up_recommendations": self._generate_follow_up_recommendations(specialty_result),
                "confidence_score": specialty_result.get("confidence", 0.7),
                "status": "completed"
            }
            
            self.log_interaction("specialist_assessment_completed", f"{self.specialty}专科评估完成")
            return result
            
        except Exception as e:
            self.logger.error(f"专科评估过程出错: {e}")
            self.status = "error"
            return {
                "agent_type": self.agent_type.value if hasattr(self.agent_type, 'value') else self.agent_type,
                "agent_name": self.name,
                "specialty": self.specialty,
                "query": query,
                "error": str(e),
                "status": "error"
            }
    
    def _build_specialty_prompt(self, query: str, context: Dict[str, Any] = None) -> str:
        """构建专科诊断提示词"""
        system_prompt = self.get_system_prompt()
        
        knowledge_context = f"""
{self.specialty}专业知识要点：
- 常见疾病：{', '.join(self.specialty_knowledge['common_conditions'])}
- 关键症状：{', '.join(self.specialty_knowledge['key_symptoms'])}
- 诊断检查：{', '.join(self.specialty_knowledge['diagnostic_tests'])}
- 治疗方法：{', '.join(self.specialty_knowledge['treatment_approaches'])}
"""
        
        prompt = f"""
{system_prompt}

{knowledge_context}

你是一位专业的{self.specialty}医生。请根据以下患者信息进行专业评估：

患者描述：{query}

请从{self.specialty}专科角度提供专业建议：

1. 专科诊断：[基于专科知识的诊断]
2. 鉴别诊断：[其他可能的专科疾病]
3. 诊断依据：[支持诊断的专业证据]
4. 专科检查建议：[需要进行的专业检查]
5. 治疗方案：[专科特色治疗方法]
6. 预后评估：[病情发展趋势]
7. 信心度：[0-1之间的数值]

请以JSON格式返回结果：
{{
    "specialist_diagnosis": "专科诊断",
    "differential_diagnosis": ["其他可能诊断"],
    "diagnostic_evidence": ["专业证据"],
    "specialty_tests": ["专科检查"],
    "treatment_plan": "专业治疗方案",
    "prognosis": "预后评估",
    "confidence": 0.8,
    "reasoning": "专业推理过程"
}}
"""
        
        return prompt
    
    def _parse_specialty_response(self, response: str) -> Dict[str, Any]:
        """解析专科响应"""
        try:
            if response.strip().startswith('{'):
                result = json.loads(response)
                return result
            else:
                return self._extract_specialty_info(response)
        except json.JSONDecodeError:
            self.logger.warning("专科响应解析失败，尝试信息提取")
            return self._extract_specialty_info(response)
    
    def _extract_specialty_info(self, response: str) -> Dict[str, Any]:
        """从文本中提取专科信息"""
        result = {
            "specialist_diagnosis": f"{self.specialty}相关疾病",
            "differential_diagnosis": [],
            "diagnostic_evidence": [f"基于{self.specialty}专业知识"],
            "specialty_tests": self.specialty_knowledge['diagnostic_tests'][:2],
            "treatment_plan": f"{self.specialty}标准治疗方案",
            "prognosis": "需要进一步评估",
            "confidence": 0.6,
            "reasoning": response[:300] + "..." if len(response) > 300 else response
        }
        
        # 根据专科关键词匹配
        specialty_keywords = {
            "心血管科": ["胸痛", "心悸", "高血压", "心脏病"],
            "神经科": ["头痛", "头晕", "肢体无力", "癫痫"],
            "肿瘤科": ["肿块", "癌症", "肿瘤", "体重下降"],
            "内分泌科": ["糖尿病", "甲状腺", "激素", "血糖"]
        }
        
        keywords = specialty_keywords.get(self.specialty, [])
        for keyword in keywords:
            if keyword in response:
                result["specialist_diagnosis"] = f"疑似{keyword}相关疾病"
                result["confidence"] = 0.7
                break
        
        return result
    
    def _build_specialty_treatment_plan(self, specialty_result: Dict[str, Any], 
                                      context: Dict[str, Any] = None) -> Dict[str, Any]:
        """构建专科治疗方案"""
        treatment_plan = {
            "immediate_actions": [],
            "medications": [],
            "procedures": [],
            "monitoring": [],
            "lifestyle_changes": [],
            "specialty_considerations": []
        }
        
        # 根据专科添加特定治疗建议
        if self.specialty == "心血管科":
            treatment_plan["immediate_actions"].append("心电图检查")
            treatment_plan["monitoring"].append("血压监测")
            treatment_plan["specialty_considerations"].append("注意心血管风险因素")
        
        elif self.specialty == "神经科":
            treatment_plan["immediate_actions"].append("神经系统检查")
            treatment_plan["monitoring"].append("意识状态监测")
            treatment_plan["specialty_considerations"].append("注意神经功能保护")
        
        elif self.specialty == "肿瘤科":
            treatment_plan["immediate_actions"].append("肿瘤标志物检测")
            treatment_plan["procedures"].append("影像学评估")
            treatment_plan["specialty_considerations"].append("多学科会诊评估")
        
        # 添加通用治疗建议
        diagnosis = specialty_result.get("specialist_diagnosis", "")
        if "高风险" in diagnosis or "严重" in diagnosis:
            treatment_plan["immediate_actions"].append("密切观察病情变化")
            treatment_plan["monitoring"].append("定期复查")
        
        return treatment_plan
    
    def _assess_prognosis(self, specialty_result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """评估预后"""
        prognosis = {
            "short_term": "良好",
            "long_term": "需要观察",
            "factors": [],
            "monitoring_requirements": []
        }
        
        # 根据诊断和专科特点评估预后
        diagnosis = specialty_result.get("specialist_diagnosis", "")
        
        if self.specialty == "心血管科":
            if any(word in diagnosis for word in ["心肌梗死", "心衰"]):
                prognosis["short_term"] = "需要密切观察"
                prognosis["long_term"] = "预后取决于治疗反应"
                prognosis["factors"].append("心功能状态")
                prognosis["monitoring_requirements"].append("心功能定期评估")
        
        elif self.specialty == "神经科":
            if any(word in diagnosis for word in ["脑卒中", "癫痫"]):
                prognosis["short_term"] = "急性期需要稳定"
                prognosis["long_term"] = "康复潜力较大"
                prognosis["factors"].append("神经功能缺损程度")
                prognosis["monitoring_requirements"].append("神经功能定期评估")
        
        elif self.specialty == "肿瘤科":
            if "癌症" in diagnosis:
                prognosis["short_term"] = "需要综合治疗"
                prognosis["long_term"] = "取决于分期和治疗反应"
                prognosis["factors"].append("肿瘤分期", "治疗反应")
                prognosis["monitoring_requirements"].append("肿瘤标志物监测", "影像学复查")
        
        return prognosis
    
    def _generate_follow_up_recommendations(self, specialty_result: Dict[str, Any]) -> List[str]:
        """生成随访建议"""
        recommendations = []
        
        # 专科特定的随访建议
        if self.specialty == "心血管科":
            recommendations.extend([
                "定期心电图检查",
                "血压监测",
                "心脏超声复查"
            ])
        
        elif self.specialty == "神经科":
            recommendations.extend([
                "神经系统功能评估",
                "影像学复查",
                "认知功能评估"
            ])
        
        elif self.specialty == "肿瘤科":
            recommendations.extend([
                "肿瘤标志物监测",
                "影像学评估",
                "生活质量评估"
            ])
        
        # 根据诊断调整建议
        diagnosis = specialty_result.get("specialist_diagnosis", "")
        if "高风险" in diagnosis:
            recommendations.append("密切随访，及时调整治疗方案")
        
        return recommendations
    
    def get_specialty_info(self) -> Dict[str, Any]:
        """获取专科信息"""
        return {
            "specialty": self.specialty,
            "knowledge_base": self.specialty_knowledge,
            "capabilities": self.specialty_capabilities,
            "common_conditions": self.specialty_knowledge['common_conditions']
        }
    
    def get_diagnostic_algorithms(self) -> Dict[str, List[str]]:
        """获取诊断算法"""
        return {
            "step_by_step_diagnosis": [
                "详细病史采集",
                "专科体格检查", 
                "针对性辅助检查",
                "鉴别诊断分析",
                "最终诊断确定"
            ],
            "red_flags": self.specialty_knowledge['key_symptoms'],
            "recommended_tests": self.specialty_knowledge['diagnostic_tests']
        }