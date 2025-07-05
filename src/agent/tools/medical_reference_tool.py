from typing import Dict, Any, Optional, List, Union
import json
import os
import re

from .tool_base import ToolBase
from utils.logger import setup_logger as get_logger

logger = get_logger(__name__)

class MedicalReferenceTool(ToolBase):
    """医疗参考工具，用于查询医学参考资料和标准"""
    
    def __init__(
        self, 
        reference_data_path: Optional[str] = None,
        name: str = "医疗参考",
        description: str = "查询医学参考数据，如诊断标准、治疗指南、药物剂量、常见医学测量的正常范围等"
    ) -> None:
        """
        初始化医疗参考工具
        
        Args:
            reference_data_path: 参考数据文件路径（JSON格式）
            name: 工具名称
            description: 工具描述
        """
        parameters = {
            "query_type": {
                "type": "string",
                "description": "查询类型，可选值：'诊断标准', '治疗指南', '药物剂量', '正常值范围', '疾病信息'",
                "required": True,
                "enum": ["诊断标准", "治疗指南", "药物剂量", "正常值范围", "疾病信息"]
            },
            "query_term": {
                "type": "string",
                "description": "查询关键词，如疾病名称、药物名称或检查名称",
                "required": True
            }
        }
        
        super().__init__(name=name, description=description, parameters=parameters)
        
        self.reference_data = {}
        if reference_data_path and os.path.exists(reference_data_path):
            try:
                with open(reference_data_path, 'r', encoding='utf-8') as f:
                    self.reference_data = json.load(f)
            except Exception as e:
                logger.error(f"加载参考数据失败: {str(e)}")
        
        # 如果没有加载数据或加载失败，使用默认的示例数据
        if not self.reference_data:
            self._load_default_data()
    
    def _load_default_data(self) -> None:
        """加载默认的示例参考数据"""
        self.reference_data = {
            "诊断标准": {
                "糖尿病": "诊断标准包括：1)空腹血糖 ≥ 7.0 mmol/L; 2)口服葡萄糖耐量试验2小时血糖 ≥ 11.1 mmol/L; 3)糖化血红蛋白 ≥ 6.5%; 4)随机血糖 ≥ 11.1 mmol/L 且有典型症状。",
                "高血压": "诊断标准为非同日3次血压测量，收缩压 ≥ 140 mmHg 和/或舒张压 ≥ 90 mmHg。",
                "冠心病": "根据典型症状、心电图改变、心肌标志物升高和冠状动脉造影等诊断。",
                "类风湿关节炎": "根据2010年ACR/EULAR分类标准，评分≥6分可确诊。"
            },
            "治疗指南": {
                "糖尿病": "治疗目标：糖化血红蛋白<7.0%。生活方式干预+二甲双胍是1型糖尿病的一线治疗。",
                "高血压": "生活方式干预+药物治疗。常用药物包括ACEI、ARB、CCB、利尿剂等。目标血压<140/90 mmHg，对于某些人群可能需要更低目标。",
                "冠心病": "抗血小板药物、他汀类、β受体阻滞剂和ACEI/ARB是基础用药。根据病情可能需要介入治疗或冠脉搭桥手术。",
                "肺炎": "社区获得性肺炎首选β-内酰胺类抗生素±大环内酯类抗生素。医院获得性肺炎需考虑广谱抗生素。"
            },
            "药物剂量": {
                "阿司匹林": "抗血小板剂量：75-100 mg/日。镇痛剂量：0.3-1 g，每4-6小时。",
                "二甲双胍": "起始剂量500 mg，每日1-2次，最大剂量2000-2500 mg/日。",
                "氨氯地平": "高血压：2.5-10 mg，每日1次。",
                "辛伐他汀": "10-40 mg，每晚服用，最大剂量80 mg/日。"
            },
            "正常值范围": {
                "血红蛋白": "男性：120-160 g/L，女性：110-150 g/L",
                "白细胞计数": "4.0-10.0 × 10^9/L",
                "血小板计数": "100-300 × 10^9/L",
                "空腹血糖": "3.9-6.1 mmol/L",
                "总胆固醇": "≤ 5.2 mmol/L",
                "甘油三酯": "≤ 1.7 mmol/L",
                "低密度脂蛋白": "≤ 3.4 mmol/L",
                "高密度脂蛋白": "男性：≥ 1.0 mmol/L，女性：≥ 1.3 mmol/L",
                "丙氨酸氨基转移酶 (ALT)": "男性：≤ 50 U/L，女性：≤ 40 U/L",
                "天门冬氨酸氨基转移酶 (AST)": "男性：≤ 40 U/L，女性：≤ 35 U/L"
            },
            "疾病信息": {
                "糖尿病": "慢性代谢性疾病，特征为长期血糖升高。主要分为1型、2型和妊娠糖尿病。",
                "高血压": "血压长期高于正常值的慢性疾病，是心脑血管疾病的主要危险因素。",
                "冠心病": "冠状动脉粥样硬化引起的心脏病，可导致心绞痛、心肌梗死等。",
                "肺炎": "肺部感染性疾病，可由细菌、病毒、真菌等病原体引起。",
                "类风湿关节炎": "慢性自身免疫性疾病，特征为关节滑膜炎症，可导致关节畸形和功能丧失。"
            }
        }
    
    def _run(self, query_type: str, query_term: str) -> str:
        """
        执行医疗参考查询
        
        Args:
            query_type: 查询类型
            query_term: 查询关键词
            
        Returns:
            查询结果文本
        """
        if query_type not in self.reference_data:
            return f"错误：不支持的查询类型 '{query_type}'"
        
        # 精确匹配查询
        if query_term in self.reference_data[query_type]:
            return f"{query_type} - {query_term}:\n{self.reference_data[query_type][query_term]}"
        
        # 模糊匹配查询
        matched_terms = []
        for term in self.reference_data[query_type].keys():
            if query_term.lower() in term.lower() or term.lower() in query_term.lower():
                matched_terms.append(term)
        
        if matched_terms:
            result = f"找到与 '{query_term}' 相关的 {len(matched_terms)} 条{query_type}:\n\n"
            for term in matched_terms:
                result += f"{term}:\n{self.reference_data[query_type][term]}\n\n"
            return result
        
        return f"未找到与 '{query_term}' 相关的{query_type}信息"
    
    def load_data_from_file(self, file_path: str) -> bool:
        """
        从文件加载参考数据
        
        Args:
            file_path: 数据文件路径（JSON格式）
            
        Returns:
            是否成功加载
        """
        if not os.path.exists(file_path):
            logger.error(f"参考数据文件不存在: {file_path}")
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.reference_data = json.load(f)
            logger.info(f"成功加载参考数据: {file_path}")
            return True
        except Exception as e:
            logger.error(f"加载参考数据失败: {str(e)}")
            return False