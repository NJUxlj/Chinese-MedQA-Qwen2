import os, sys
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Any, Optional




class EvaluatorConfig(BaseModel):
    """ 评估器 配置
    
    """
    model_name_or_path: str = Field(default="Qwen/Qwen3-14B", description="微调后的模型路径")
    device: str = Field(default="cuda:0", description="计算设备 (default: cuda:0)")

    test_dataset_path: str = Field(default="data/MedQALLM/test.json", description="测试数据集路径")

    prompt_key: str = Field(default="prompt", description="数据集中用于存储 prompt 的键名")

    ground_true_answer_key: str = Field(default="ground_true_answer", description="数据集中用于存储 ground true answer 的键名")
    
    padding_side: str = Field(default="left", description="填充侧 (default: left)")

    use_fast: bool = Field(default=False, description="是否使用快速分词器 (default: False)")

    enable_thinking: bool = Field(default=True, description="是否启用思考模式")

    temperature: float = Field(default=0.7, description="生成温度 (default: 0.7)")
    max_new_tokens: int = Field(default=512, description="最大生成令牌数 (default: 512)")
    top_p: float = Field(default=0.9, description="生成概率阈值 (default: 0.9)")
    do_sample: bool = Field(default=True, description="是否启用采样 (default: True)")

    per_device_eval_batch_size: int = Field(default=4, description="每个设备的评估批量大小")
    dataloader_num_workers: int = Field(default=4, description="数据加载器的工作线程数")

    eval_result_save_path: str = Field(default="eval_results.json", description="评估结果保存路径")




class CodeEvaluatorConfig(EvaluatorConfig):
    """ 代码评估器 配置
    
    """
    execution_timeout: int = Field(default=10, description="代码执行超时时间(秒)")
    max_memory_mb: int = Field(default=256, description="最大内存限制(MB)")
    max_output_size: int = Field(default=1024 * 1024, description="最大输出大小(字节)")
    sandbox_working_dir: Optional[str] = Field(default=None, description="沙箱工作目录")
    enable_security_check: bool = Field(default=True, description="是否启用安全检查")
    supported_languages: List[str] = Field(
        default_factory=lambda: ['python', 'python3', 'javascript', 'java', 'cpp', 'c'],
        description="支持的语言列表"
    )



class MathEvaluatorConfig(EvaluatorConfig):
    """ 数学评估器 配置
    
    """


class MedQALLMEvaluatorConfig(EvaluatorConfig):
    """ 医疗问答评估器 配置
    
    """


class BleuRougeEvaluatorConfig(EvaluatorConfig):
    """ BLEU-ROUGE 评估器 配置
    
    """


class DPOQualityEvaluatorConfig(EvaluatorConfig):
    """ DPO质量评估器 配置
    
    """
    reference_model_path: Optional[str] = Field(default=None, description="参考模型路径")
    beta: float = Field(default=0.1, description="DPO中的beta参数")
    max_length: int = Field(default=512, description="最大序列长度")
    
    query_key: str = Field(default="query", description="数据集中用于存储query的键名")
    chosen_key: str = Field(default="chosen", description="数据集中用于存储chosen回答的键名")
    rejected_key: str = Field(default="rejected", description="数据集中用于存储rejected回答的键名")
    
    output_dir: str = Field(default="dpo_evaluation_results", description="评估结果输出目录")