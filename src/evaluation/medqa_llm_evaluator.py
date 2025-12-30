import json  
import numpy as np  
import re
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from tqdm import tqdm  
from datasets import load_dataset  
from transformers import AutoModelForCausalLM, AutoTokenizer  
import torch  
from torch.utils.data import DataLoader  
import evaluate  
from typing import Dict, Any, List, Optional


from config.evaluator_config import MedQALLMEvaluatorConfig
from config.llm_config import LLMConfig
from evaluation.base_evaluator import BaseEvaluator, EvaluatorDataset

class MedQALLMEvaluator(BaseEvaluator):  
    def __init__(
        self,
        llm_config: LLMConfig,
        config: MedQALLMEvaluatorConfig):  
        """  
        初始化评估器  
        """  
        super().__init__(config)
        self.config = config
        self.llm_config = llm_config
        
    
    def format_prompt(self, sample):  
        """构建模型评估指令"""  

        system_prompt  =f"""
        ## 角色
        你是一个拥有丰富临床经验的医疗专家，你的任务是根据用户的问题和上下文，判断模型的回答是否与标准答案一致。

        ## 任务：
        请根据用户问题，判断模型回答是否与标准答案一致。

        ## 规则:
        - 模型回答必须与标准答案完全一致，包括大小写、标点符号等。
        - 如果模型回答中包含多个选项，必须全部选择正确。
        - 如果模型回答中包含数值，必须与标准答案完全一致。

        ## 用户问题
        {sample['question']}

        ## 模型回答
        {sample['output']}

        ## 标准答案
        {sample[self.config.ground_true_answer_key]}

        ## 输出格式
        你只能输出 True 或 False， 除此以外不能输出任何东西。

        ## 请你开始判断


        """
        return f"指令：{sample['instruction']}\n问题：{sample['input']}\n回答："  
    
    
    
    def evaluate_one_sample(self, sample: Dict[str, Any]) -> Dict[str, float]:  
        """  
        对单个样本进行评估  
        """  
        prompt = self.format_prompt(sample)  
        output = self.model.generate(
            prompt,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )


    
    def evaluate_batch_sample(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:  
        """  
        对批量样本进行评估  
        """  
        prompts = [self.format_prompt(sample) for sample in samples]  
        


    
    def evaluate(self):
        pass
        
            



def run():
    pass




# 使用示例  
if __name__ == "__main__":  
    pass