import os, sys
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Any




class EvaluatorConfig(BaseModel):
    """ 评估器 配置
    
    """
    model_name_or_path: str = Field(default="Qwen/Qwen3-14B", description="微调后的模型路径")
    device: str = Field(default="cuda:0", description="计算设备 (default: cuda:0)")

    test_dataset_path: str = Field(default="data/MedQALLM/test.json", description="测试数据集路径")

    prompt_key: str = Field(default="prompt", description="数据集中用于存储 prompt 的键名")

    ground_true_answer_key: str = Field(default="ground_true_answer", description="数据集中用于存储 ground true answer 的键名")
    

    per_device_eval_batch_size: int = Field(default=4, description="每个设备的评估批量大小")
    dataloader_num_workers: int = Field(default=4, description="数据加载器的工作线程数")

    eval_result_save_path: str = Field(default="eval_results.json", description="评估结果保存路径")




class CodeEvaluatorConfig(EvaluatorConfig):
    """ 代码评估器 配置
    
    """



class MathEvaluatorConfig(EvaluatorConfig):
    """ 数学评估器 配置
    
    """


class MedQALLMEvaluatorConfig(EvaluatorConfig):
    """ 医疗问答评估器 配置
    
    """


class BleuRougeEvaluatorConfig(EvaluatorConfig):
    """ BLEU-ROUGE 评估器 配置
    
    """