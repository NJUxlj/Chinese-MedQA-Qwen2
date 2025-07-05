from typing import Dict, Any, Optional, List, Union, Tuple
import json
import math

from .tool_base import ToolBase
from utils.logger import setup_logger as get_logger

logger = get_logger(__name__)

class MedicalAssessmentTool(ToolBase):
    """医疗评估工具，用于进行健康风险评估和疾病筛查"""
    
    def __init__(
        self, 
        name: str = "医疗评估",
        description: str = "进行常见的健康风险评估和疾病筛查，包括心血管疾病风险评估、糖尿病风险评估等"
    ) -> None:
        """
        初始化医疗评估工具
        
        Args:
            name: 工具名称
            description: 工具描述
        """
        parameters = {
            "assessment_type": {
                "type": "string",
                "description": "评估类型，可选值：'心血管风险', '糖尿病风险', '抑郁症筛查', '焦虑症筛查'",
                "required": True,
                "enum": ["心血管风险", "糖尿病风险", "抑郁症筛查", "焦虑症筛查"]
            },
            "patient_data": {
                "type": "object",
                "description": "病人数据，具体字段取决于评估类型",
                "required": True
            }
        }
        
        super().__init__(name=name, description=description, parameters=parameters)
    
    def _run(self, assessment_type: str, patient_data: Dict[str, Any]) -> str:
        """
        执行医疗评估
        
        Args:
            assessment_type: 评估类型
            patient_data: 病人数据
            
        Returns:
            评估结果文本
        """
        if assessment_type == "心血管风险":
            return self._assess_cardiovascular_risk(patient_data)
        elif assessment_type == "糖尿病风险":
            return self._assess_diabetes_risk(patient_data)
        elif assessment_type == "抑郁症筛查":
            return self._screen_depression(patient_data)
        elif assessment_type == "焦虑症筛查":
            return self._screen_anxiety(patient_data)
        else:
            return f"不支持的评估类型: {assessment_type}"
    
    def _assess_cardiovascular_risk(self, patient_data: Dict[str, Any]) -> str:
        """
        评估心血管疾病风险（基于Framingham风险评分）
        
        Args:
            patient_data: 包含年龄、性别、总胆固醇、HDL、收缩压、是否吸烟、是否糖尿病等信息
            
        Returns:
            风险评估结果
        """
        required_fields = ["age", "gender", "total_cholesterol", "hdl_cholesterol", 
                          "systolic_bp", "is_smoker", "is_diabetic", "is_treated_bp"]
        
        # 检查必填字段
        missing_fields = [field for field in required_fields if field not in patient_data]
        if missing_fields:
            return f"缺少必要的患者数据字段: {', '.join(missing_fields)}"
        
        try:
            # 提取数据
            age = int(patient_data["age"])
            gender = patient_data["gender"].lower()  # "male" or "female"
            total_chol = float(patient_data["total_cholesterol"])  # mg/dL
            hdl = float(patient_data["hdl_cholesterol"])  # mg/dL
            systolic_bp = float(patient_data["systolic_bp"])  # mmHg
            is_smoker = bool(patient_data["is_smoker"])
            is_diabetic = bool(patient_data["is_diabetic"])
            is_treated_bp = bool(patient_data["is_treated_bp"])
            
            # 计算风险得分
            score = 0
            
            # 年龄得分
            if gender == "male":
                if age < 35:
                    score -= 7
                elif age < 40:
                    score -= 3
                elif age < 45:
                    score += 0
                elif age < 50:
                    score += 3
                elif age < 55:
                    score += 6
                elif age < 60:
                    score += 8
                elif age < 65:
                    score += 10
                elif age < 70:
                    score += 12
                else:
                    score += 14
            else:  # female
                if age < 35:
                    score -= 7
                elif age < 40:
                    score -= 3
                elif age < 45:
                    score += 0
                elif age < 50:
                    score += 3
                elif age < 55:
                    score += 6
                elif age < 60:
                    score += 8
                elif age < 65:
                    score += 10
                elif age < 70:
                    score += 12
                else:
                    score += 14
            
            # 总胆固醇得分
            if gender == "male":
                if total_chol < 160:
                    score -= 3
                elif total_chol < 200:
                    score += 0
                elif total_chol < 240:
                    score += 1
                elif total_chol < 280:
                    score += 2
                else:
                    score += 3
            else:  # female
                if total_chol < 160:
                    score -= 2
                elif total_chol < 200:
                    score += 0
                elif total_chol < 240:
                    score += 1
                elif total_chol < 280:
                    score += 2
                else:
                    score += 3
            
            # HDL胆固醇得分
            if hdl >= 60:
                score -= 1
            elif hdl >= 50:
                score += 0
            elif hdl >= 40:
                score += 1
            else:
                score += 2
            
            # 收缩压得分
            if not is_treated_bp:
                if systolic_bp < 120:
                    score += 0
                elif systolic_bp < 130:
                    score += 1
                elif systolic_bp < 140:
                    score += 2
                elif systolic_bp < 160:
                    score += 3
                else:
                    score += 4
            else:
                if systolic_bp < 120:
                    score += 0
                elif systolic_bp < 130:
                    score += 3
                elif systolic_bp < 140:
                    score += 4
                elif systolic_bp < 160:
                    score += 5
                else:
                    score += 6
            
            # 吸烟得分
            if is_smoker:
                score += 4
            
            # 糖尿病得分
            if is_diabetic:
                if gender == "male":
                    score += 3
                else:
                    score += 4
            
            # 计算10年心血管疾病风险
            if gender == "male":
                risk_percent = 100 * (1 - 0.88936 ** (math.exp(score - 23.9802)))
            else:
                risk_percent = 100 * (1 - 0.95012 ** (math.exp(score - 26.1931)))
            
            # 风险分类
            risk_category = ""
            if risk_percent < 10:
                risk_category = "低风险"
            elif risk_percent < 20:
                risk_category = "中等风险"
            else:
                risk_category = "高风险"
            
            return (
                f"心血管疾病风险评估结果:\n\n"
                f"10年心血管疾病风险: {risk_percent:.1f}%\n"
                f"风险分类: {risk_category}\n\n"
                f"评估建议:\n"
                f"- 低风险 (<10%): 保持健康生活方式\n"
                f"- 中等风险 (10-20%): 积极干预可修改的风险因素\n"
                f"- 高风险 (>20%): 考虑药物干预，严格控制风险因素\n\n"
                f"注意: 此评估仅供参考，应结合临床医生的专业意见进行综合判断。"
            )
            
        except Exception as e:
            logger.error(f"心血管风险评估失败: {str(e)}")
            return f"评估过程中发生错误: {str(e)}"
    
    def _assess_diabetes_risk(self, patient_data: Dict[str, Any]) -> str:
        """
        评估2型糖尿病风险（基于Finnish Diabetes Risk Score）
        
        Args:
            patient_data: 包含年龄、BMI、腰围、运动情况、饮食习惯等信息
            
        Returns:
            风险评估结果
        """
        required_fields = ["age", "bmi", "waist_circumference", "physical_activity",
                          "vegetables_fruits_berries", "hypertension_medication",
                          "high_blood_glucose", "family_diabetes"]
        
        # 检查必填字段
        missing_fields = [field for field in required_fields if field not in patient_data]
        if missing_fields:
            return f"缺少必要的患者数据字段: {', '.join(missing_fields)}"
        
        try:
            # 提取数据
            age = int(patient_data["age"])
            bmi = float(patient_data["bmi"])
            waist_cm = float(patient_data["waist_circumference"])  # 腰围，厘米
            physical_activity = bool(patient_data["physical_activity"])  # 每天至少30分钟的体力活动
            vegetables_fruits = bool(patient_data["vegetables_fruits_berries"])  # 每天摄入蔬菜水果
            hypertension_medication = bool(patient_data["hypertension_medication"])  # 是否服用高血压药物
            high_blood_glucose = bool(patient_data["high_blood_glucose"])  # 是否有高血糖史
            family_diabetes = patient_data["family_diabetes"]  # "no", "yes_distant", "yes_immediate"
            
            # 计算FINDRISC得分
            score = 0
            
            # 年龄得分
            if age < 45:
                score += 0
            elif age <= 54:
                score += 2
            elif age <= 64:
                score += 3
            else:
                score += 4
            
            # BMI得分
            if bmi < 25:
                score += 0
            elif bmi <= 30:
                score += 1
            else:
                score += 3
            
            # 腰围得分
            if waist_cm < 94:  # 假设为男性标准，女性应为80cm
                score += 0
            elif waist_cm <= 102:  # 女性应为88cm
                score += 3
            else:
                score += 4
            
            # 体力活动得分
            if physical_activity:
                score += 0
            else:
                score += 2
            
            # 饮食习惯得分
            if vegetables_fruits:
                score += 0
            else:
                score += 1
            
            # 高血压药物得分
            if hypertension_medication:
                score += 2
            else:
                score += 0
            
            # 高血糖史得分
            if high_blood_glucose:
                score += 5
            else:
                score += 0
            
            # 家族糖尿病史得分
            if family_diabetes == "no":
                score += 0
            elif family_diabetes == "yes_distant":
                score += 3
            else:  # "yes_immediate"
                score += 5
            
            # 风险等级
            risk_level = ""
            risk_description = ""
            if score < 7:
                risk_level = "低风险"
                risk_description = "10年内发展为2型糖尿病的风险约为1%"
            elif score <= 11:
                risk_level = "轻度风险"
                risk_description = "10年内发展为2型糖尿病的风险约为4%"
            elif score <= 14:
                risk_level = "中度风险"
                risk_description = "10年内发展为2型糖尿病的风险约为17%"
            elif score <= 20:
                risk_level = "高风险"
                risk_description = "10年内发展为2型糖尿病的风险约为33%"
            else:
                risk_level = "非常高风险"
                risk_description = "10年内发展为2型糖尿病的风险约为50%"
            
            return (
                f"2型糖尿病风险评估结果:\n\n"
                f"FINDRISC得分: {score}分\n"
                f"风险等级: {risk_level}\n"
                f"风险说明: {risk_description}\n\n"
                f"建议措施:\n"
                f"- 低风险 (<7分): 保持健康生活方式\n"
                f"- 轻度风险 (7-11分): 考虑改善饮食和增加运动\n"
                f"- 中度风险 (12-14分): 建议向医生咨询糖尿病预防措施\n"
                f"- 高风险 (15-20分): 建议进行糖耐量检测，积极干预生活方式\n"
                f"- 非常高风险 (>20分): 建议尽快就医，进行糖尿病筛查和干预\n\n"
                f"注意: 此评估仅供参考，应结合临床医生的专业意见进行综合判断。"
            )
            
        except Exception as e:
            logger.error(f"糖尿病风险评估失败: {str(e)}")
            return f"评估过程中发生错误: {str(e)}"
    
    def _screen_depression(self, patient_data: Dict[str, Any]) -> str:
        """
        抑郁症筛查（基于PHQ-9问卷）
        
        Args:
            patient_data: 包含PHQ-9的9个问题的回答
            
        Returns:
            筛查结果
        """
        try:
            # 检查是否包含PHQ-9的所有问题
            if "phq9_answers" not in patient_data or not isinstance(patient_data["phq9_answers"], list):
                return "缺少PHQ-9问卷回答数据，应提供包含9个回答的列表"
            
            phq9_answers = patient_data["phq9_answers"]
            if len(phq9_answers) != 9:
                return f"PHQ-9问卷应包含9个问题的回答，但提供了{len(phq9_answers)}个"
            
            # 计算总分
            try:
                total_score = sum(int(answer) for answer in phq9_answers)
            except (ValueError, TypeError):
                return "PHQ-9回答应为0-3的整数，分别代表'完全没有'、'几天'、'超过一半的天数'、'几乎每天'"
            
            # 抑郁症严重程度判断
            severity = ""
            recommendation = ""
            if total_score < 5:
                severity = "无抑郁症状"
                recommendation = "无需特殊干预"
            elif total_score < 10:
                severity = "轻度抑郁"
                recommendation = "观察监测，考虑支持性治疗"
            elif total_score < 15:
                severity = "中度抑郁"
                recommendation = "建议心理治疗，考虑药物治疗"
            elif total_score < 20:
                severity = "中重度抑郁"
                recommendation = "建议药物治疗和/或心理治疗"
            else:
                severity = "重度抑郁"
                recommendation = "建议积极药物治疗和心理治疗，考虑专科转诊"
            
            # 自杀风险评估（基于第9题）
            suicide_risk = ""
            if phq9_answers[8] >= 1:  # 第9题，有关自杀意念的问题
                suicide_risk = f"注意: 患者存在自杀风险，第9题得分为{phq9_answers[8]}/3分，建议进一步评估和干预"
            
            return (
                f"抑郁症筛查结果 (PHQ-9):\n\n"
                f"总分: {total_score}/27分\n"
                f"抑郁严重程度: {severity}\n"
                f"建议干预措施: {recommendation}\n"
                f"{suicide_risk}\n\n"
                f"参考分数解释:\n"
                f"- 0-4分: 无抑郁症状\n"
                f"- 5-9分: 轻度抑郁\n"
                f"- 10-14分: 中度抑郁\n"
                f"- 15-19分: 中重度抑郁\n"
                f"- 20-27分: 重度抑郁\n\n"
                f"注意: 此筛查仅供参考，不能替代专业精神科医师的诊断。"
            )
            
        except Exception as e:
            logger.error(f"抑郁症筛查失败: {str(e)}")
            return f"筛查过程中发生错误: {str(e)}"
    
    def _screen_anxiety(self, patient_data: Dict[str, Any]) -> str:
        """
        焦虑症筛查（基于GAD-7问卷）
        
        Args:
            patient_data: 包含GAD-7的7个问题的回答
            
        Returns:
            筛查结果
        """
        try:
            # 检查是否包含GAD-7的所有问题
            if "gad7_answers" not in patient_data or not isinstance(patient_data["gad7_answers"], list):
                return "缺少GAD-7问卷回答数据，应提供包含7个回答的列表"
            
            gad7_answers = patient_data["gad7_answers"]
            if len(gad7_answers) != 7:
                return f"GAD-7问卷应包含7个问题的回答，但提供了{len(gad7_answers)}个"
            
            # 计算总分
            try:
                total_score = sum(int(answer) for answer in gad7_answers)
            except (ValueError, TypeError):
                return "GAD-7回答应为0-3的整数，分别代表'完全没有'、'几天'、'超过一半的天数'、'几乎每天'"
            
            # 焦虑症严重程度判断
            severity = ""
            recommendation = ""
            if total_score < 5:
                severity = "无焦虑症状"
                recommendation = "无需特殊干预"
            elif total_score < 10:
                severity = "轻度焦虑"
                recommendation = "观察监测，考虑支持性治疗"
            elif total_score < 15:
                severity = "中度焦虑"
                recommendation = "建议心理治疗，考虑药物治疗"
            else:
                severity = "重度焦虑"
                recommendation = "建议药物治疗和/或心理治疗，考虑专科转诊"
            
            return (
                f"焦虑症筛查结果 (GAD-7):\n\n"
                f"总分: {total_score}/21分\n"
                f"焦虑严重程度: {severity}\n"
                f"建议干预措施: {recommendation}\n\n"
                f"参考分数解释:\n"
                f"- 0-4分: 无焦虑症状\n"
                f"- 5-9分: 轻度焦虑\n"
                f"- 10-14分: 中度焦虑\n"
                f"- 15-21分: 重度焦虑\n\n"
                f"注意: 此筛查仅供参考，不能替代专业精神科医师的诊断。"
            )
            
        except Exception as e:
            logger.error(f"焦虑症筛查失败: {str(e)}")
            return f"筛查过程中发生错误: {str(e)}"