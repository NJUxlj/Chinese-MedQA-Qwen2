"""
初级保健临床医生智能体
"""

from typing import Dict, List, Any, Optional
import json

from .base_agent import BaseAgent
from ..config import AgentConfig, AgentType

class PrimaryCareClinician(BaseAgent):
    """初级保健临床医生智能体"""
    
    def __init__(self, config: AgentConfig, api_model=None, llm_config=None):
        """
        初始化初级保健临床医生
        
        Args:
            config: 智能体配置
            api_model: API模型实例
            llm_config: LLM配置实例
        """
        super().__init__(config, api_model, llm_config)
        
        # 专科领域
        self.specialties = ["全科", "内科", "家庭医学"]
        
        # 常见疾病知识库
        self.common_conditions = {
            "上呼吸道感染": {
                "症状": ["咳嗽", "鼻塞", "喉咙痛", "发热"],
                "诊断要点": ["症状持续时间", "体温变化", "咽喉检查"],
                "治疗建议": ["休息", "多饮水", "对症治疗", "必要时就医"]
            },
            "高血压": {
                "症状": ["头痛", "头晕", "视力模糊", "心悸"],
                "诊断要点": ["血压测量", "多次测量确认", "排除继发性高血压"],
                "治疗建议": ["生活方式改变", "药物治疗", "定期监测", "定期复查"]
            },
            "糖尿病": {
                "症状": ["多饮", "多尿", "多食", "体重下降", "疲劳"],
                "诊断要点": ["血糖检测", "糖化血红蛋白", "口服葡萄糖耐量试验"],
                "治疗建议": ["饮食控制", "运动", "药物治疗", "血糖监测"]
            }
        }

    def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        处理医疗查询
        
        Args:
            query: 医疗查询
            context: 上下文信息
            
        Returns:
            处理结果
        """
        self.log_interaction("query_received", f"收到查询: {query[:100]}...")
        
        try:
            # 构建诊断提示词
            diagnostic_prompt = self._build_diagnostic_prompt(query, context)
            
            # 生成诊断建议
            diagnostic_response = self.generate_response(diagnostic_prompt)
            
            # 解析响应
            diagnosis_result = self._parse_diagnostic_response(diagnostic_response)
            
            # 构建治疗建议
            treatment_plan = self._build_treatment_plan(diagnosis_result, context)
            
            # 评估是否需要转诊
            referral_needed = self._assess_referral_need(query, diagnosis_result)
            
            result = {
                "agent_type": self.agent_type.value,
                "agent_name": self.name,
                "query": query,
                "diagnosis": diagnosis_result,
                "treatment_plan": treatment_plan,
                "referral_needed": referral_needed,
                "confidence_score": diagnosis_result.get("confidence", 0.5),
                "reasoning": diagnosis_result.get("reasoning", ""),
                "next_steps": self._generate_next_steps(diagnosis_result, referral_needed),
                "status": "completed"
            }
            
            self.log_interaction("diagnosis_completed", f"诊断完成: {diagnosis_result.get('primary_diagnosis', 'Unknown')}")
            return result
            
        except Exception as e:
            self.logger.error(f"诊断过程出错: {e}")
            self.status = "error"
            return {
                "agent_type": self.agent_type.value,
                "agent_name": self.name,
                "query": query,
                "error": str(e),
                "status": "error"
            }
    
    def _build_diagnostic_prompt(self, query: str, context: Dict[str, Any] = None) -> str:
        """构建诊断提示词"""
        system_prompt = self.get_system_prompt()
        
        prompt = f"""
{system_prompt}

你是一个经验丰富的初级保健临床医生。请根据以下患者信息进行初步诊断：

患者描述：{query}

请按照以下格式提供诊断建议：
1. 主要诊断：[最可能的诊断]
2. 鉴别诊断：[其他可能的诊断]
3. 诊断依据：[支持诊断的证据]
4. 建议检查：[需要进行的检查]
5. 风险评估：[低/中/高风险]
6. 信心度：[0-1之间的数值]

请确保：
- 基于患者症状和描述进行分析
- 考虑常见病的可能性
- 如果症状复杂或严重，建议转诊
- 提供具体的治疗建议
- 给出明确的信心度评分

请以JSON格式返回结果，包含以下字段：
{{
    "primary_diagnosis": "主要诊断",
    "differential_diagnosis": ["鉴别诊断1", "鉴别诊断2"],
    "diagnostic_evidence": ["证据1", "证据2"],
    "recommended_tests": ["检查1", "检查2"],
    "risk_level": "低风险/中风险/高风险",
    "confidence": 0.8,
    "reasoning": "诊断推理过程",
    "treatment_approach": "治疗方案概述"
}}
"""
        
        return prompt
    
    def _parse_diagnostic_response(self, response: str) -> Dict[str, Any]:
        """解析诊断响应"""
        try:
            # 尝试解析JSON
            if response.strip().startswith('{'):
                result = json.loads(response)
                return result
            else:
                # 如果不是JSON格式，尝试提取关键信息
                return self._extract_diagnostic_info(response)
        except json.JSONDecodeError:
            self.logger.warning("响应不是有效的JSON格式，尝试提取信息")
            return self._extract_diagnostic_info(response)
    
    def _extract_diagnostic_info(self, response: str) -> Dict[str, Any]:
        """从文本响应中提取诊断信息"""
        # 这里实现简单的信息提取逻辑
        # 在实际应用中，可能需要更复杂的NLP处理
        
        result = {
            "primary_diagnosis": "需要进一步评估",
            "differential_diagnosis": [],
            "diagnostic_evidence": ["基于患者描述"],
            "recommended_tests": ["详细体检", "必要检查"],
            "risk_level": "需要评估",
            "confidence": 0.3,
            "reasoning": response[:200] + "..." if len(response) > 200 else response,
            "treatment_approach": "需要更多信息进行诊断"
        }
        
        # 简单的关键词匹配
        if any(word in response for word in ["上呼吸道", "感冒", "咳嗽"]):
            result["primary_diagnosis"] = "上呼吸道感染"
            result["confidence"] = 0.7
        
        if any(word in response for word in ["高血压", "血压高"]):
            result["primary_diagnosis"] = "高血压"
            result["confidence"] = 0.6
        
        if any(word in response for word in ["糖尿病", "血糖"]):
            result["primary_diagnosis"] = "糖尿病"
            result["confidence"] = 0.6
        
        return result
    
    def _build_treatment_plan(self, diagnosis_result: Dict[str, Any], 
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """构建治疗计划"""
        primary_diagnosis = diagnosis_result.get("primary_diagnosis", "")
        
        # 基于诊断匹配治疗方案
        treatment_plan = {
            "immediate_actions": [],
            "medications": [],
            "lifestyle_recommendations": [],
            "follow_up": "",
            "warning_signs": []
        }
        
        if primary_diagnosis in self.common_conditions:
            condition_info = self.common_conditions[primary_diagnosis]
            treatment_plan["treatment_approach"] = condition_info["治疗建议"]
            treatment_plan["immediate_actions"] = condition_info["治疗建议"][:2]
        
        # 根据风险等级调整治疗方案
        risk_level = diagnosis_result.get("risk_level", "需要评估")
        
        if "高风险" in risk_level:
            treatment_plan["immediate_actions"].append("立即就医")
            treatment_plan["warning_signs"] = ["症状加重", "新症状出现", "持续不适"]
            treatment_plan["follow_up"] = "1-2天内复查"
        
        elif "中风险" in risk_level:
            treatment_plan["warning_signs"] = ["症状持续", "功能下降"]
            treatment_plan["follow_up"] = "3-5天内复查"
        
        else:
            treatment_plan["follow_up"] = "1周内复查或症状持续时就医"
        
        return treatment_plan
    
    def _assess_referral_need(self, query: str, diagnosis_result: Dict[str, Any]) -> Dict[str, Any]:
        """评估是否需要转诊"""
        referral_assessment = {
            "needed": False,
            "reason": "",
            "specialty": None,
            "urgency": "routine"
        }
        
        # 根据症状判断
        emergency_symptoms = ["胸痛", "呼吸困难", "昏迷", "严重出血", "高热"]
        if any(symptom in query for symptom in emergency_symptoms):
            referral_assessment["needed"] = True
            referral_assessment["reason"] = "出现紧急症状"
            referral_assessment["urgency"] = "emergency"
            return referral_assessment
        
        # 根据信心度判断
        confidence = diagnosis_result.get("confidence", 0.5)
        if confidence < 0.4:
            referral_assessment["needed"] = True
            referral_assessment["reason"] = "诊断信心不足，需要专科评估"
            referral_assessment["urgency"] = "routine"
        
        # 根据风险等级判断
        risk_level = diagnosis_result.get("risk_level", "")
        if "高风险" in risk_level:
            referral_assessment["needed"] = True
            referral_assessment["reason"] = "高风险状况需要专科处理"
            referral_assessment["urgency"] = "urgent"
        
        # 根据专科需求判断
        diagnosis = diagnosis_result.get("primary_diagnosis", "")
        specialty_mapping = {
            "心脏病": "心血管科",
            "脑卒中": "神经科", 
            "癌症": "肿瘤科",
            "糖尿病": "内分泌科",
            "肾病": "肾内科"
        }
        
        for condition, specialty in specialty_mapping.items():
            if condition in diagnosis:
                referral_assessment["needed"] = True
                referral_assessment["reason"] = f"需要{condition}专科评估"
                referral_assessment["specialty"] = specialty
                break
        
        return referral_assessment
    
    def _generate_next_steps(self, diagnosis_result: Dict[str, Any], 
                           referral_needed: Dict[str, Any]) -> List[str]:
        """生成后续步骤"""
        next_steps = []
        
        if referral_needed["needed"]:
            next_steps.append(f"转诊至{referral_needed['specialty']}或急诊科")
            if referral_needed["urgency"] == "emergency":
                next_steps.append("立即前往医院急诊")
            else:
                next_steps.append("预约专科门诊")
        else:
            next_steps.append("按计划治疗并观察症状变化")
            next_steps.append("注意生活方式调整")
        
        # 添加检查建议
        recommended_tests = diagnosis_result.get("recommended_tests", [])
        for test in recommended_tests[:2]:  # 只取前两个
            next_steps.append(f"进行{test}")
        
        return next_steps
    
    def get_expertise_areas(self) -> List[str]:
        """获取专业领域"""
        return self.specialties.copy()
    
    def get_common_conditions(self) -> Dict[str, Any]:
        """获取常见疾病知识库"""
        return self.common_conditions.copy()


# 别名兼容性
PCCAgent = PrimaryCareClinician