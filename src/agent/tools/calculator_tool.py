from typing import Dict, Any, Optional, List, Union
import math
import re

from .tool_base import ToolBase
from ...utils.logger import get_logger

logger = get_logger(__name__)

class CalculatorTool(ToolBase):
    """计算工具，用于执行数学计算和医疗相关的单位转换"""
    
    def __init__(
        self, 
        name: str = "计算工具",
        description: str = "执行数学计算，包括基本运算、医疗剂量计算、BMI计算等"
    ) -> None:
        """
        初始化计算工具
        
        Args:
            name: 工具名称
            description: 工具描述
        """
        parameters = {
            "expression": {
                "type": "string",
                "description": "要计算的数学表达式，例如 '2 + 2' 或 '体重(kg)/身高(m)^2'",
                "required": True
            }
        }
        
        super().__init__(name=name, description=description, parameters=parameters)
    
    def _run(self, expression: str) -> str:
        """
        执行计算
        
        Args:
            expression: 数学表达式
            
        Returns:
            计算结果文本
        """
        # 剂量计算特殊处理
        dose_match = re.search(r"(\d+(\.\d+)?)\s*mg/kg\s*[*×]\s*(\d+(\.\d+)?)\s*kg", expression)
        if dose_match:
            try:
                dose_per_kg = float(dose_match.group(1))
                weight = float(dose_match.group(3))
                total_dose = dose_per_kg * weight
                return f"剂量计算结果: {total_dose} mg"
            except Exception as e:
                logger.error(f"剂量计算错误: {str(e)}")
        
        # BMI计算特殊处理
        bmi_match = re.search(r"BMI\s*[:：]?\s*(\d+(\.\d+)?)\s*kg\s*/\s*\(?(\d+(\.\d+)?)\s*m\)?[\^2²]", expression, re.IGNORECASE)
        if bmi_match:
            try:
                weight = float(bmi_match.group(1))
                height = float(bmi_match.group(3))
                bmi = weight / (height * height)
                bmi_category = self._get_bmi_category(bmi)
                return f"BMI计算结果: {bmi:.2f} ({bmi_category})"
            except Exception as e:
                logger.error(f"BMI计算错误: {str(e)}")
        
        # 体表面积计算特殊处理
        bsa_match = re.search(r"体表面积\s*[:：]?\s*(\d+(\.\d+)?)\s*kg\s*[,，]\s*(\d+(\.\d+)?)\s*cm", expression, re.IGNORECASE)
        if bsa_match:
            try:
                weight = float(bsa_match.group(1))
                height = float(bsa_match.group(3))
                # 使用Du Bois公式计算体表面积
                bsa = 0.007184 * (weight ** 0.425) * (height ** 0.725)
                return f"体表面积计算结果: {bsa:.4f} m²"
            except Exception as e:
                logger.error(f"体表面积计算错误: {str(e)}")
        
        # 一般表达式计算
        try:
            # 替换常见数学符号
            expression = expression.replace("^", "**")  # 幂运算
            expression = expression.replace("×", "*")   # 乘法
            expression = expression.replace("÷", "/")   # 除法
            
            # 安全执行计算
            # 定义允许的数学函数和变量
            safe_dict = {
                'abs': abs,
                'round': round,
                'min': min,
                'max': max,
                'pow': pow,
                'sqrt': math.sqrt,
                'pi': math.pi,
                'e': math.e,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'log': math.log,
                'log10': math.log10,
                'floor': math.floor,
                'ceil': math.ceil
            }
            
            # 使用eval执行计算
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            
            return f"计算结果: {result}"
        
        except Exception as e:
            logger.error(f"计算表达式 '{expression}' 时出错: {str(e)}")
            return f"计算错误: {str(e)}"
    
    def _get_bmi_category(self, bmi: float) -> str:
        """
        根据BMI值获取体重分类
        
        Args:
            bmi: BMI值
            
        Returns:
            体重分类描述
        """
        if bmi < 18.5:
            return "体重过轻"
        elif bmi < 24.0:
            return "正常范围"
        elif bmi < 28.0:
            return "超重"
        elif bmi < 30.0:
            return "轻度肥胖"
        elif bmi < 35.0:
            return "中度肥胖"
        else:
            return "重度肥胖"