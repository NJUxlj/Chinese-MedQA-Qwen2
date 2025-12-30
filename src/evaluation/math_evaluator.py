import sys
from pathlib import Path
import re
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any, Tuple
from transformers import (
    PreTrainedModel, 
    PreTrainedTokenizer, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    Qwen3ForCausalLM
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.logger import setup_logger
from utils.metrics import DPOMetrics
from config.evaluator_config import DPOQualityEvaluatorConfig, MathEvaluatorConfig
from evaluation.base_evaluator import BaseEvaluator, EvaluatorDataset




class MathEvaluator(BaseEvaluator):
    def __init__(
        self,
        config: MathEvaluatorConfig
    ):
        super().__init__(config)
        self.config = config


        self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):
        """
        加载模型和分词器
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=self.config.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            padding_side=self.config.padding_side,
            use_fast=self.config.use_fast,
        )


    def evaluate_one_sample(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """
        评估单个样本
        
        Args:
            sample: 包含 prompt 和 ground_true_answer 的字典
            
        Returns:
            dict: 包含评估指标的字典
        """
        return self.evaluate_batch_samples([sample])

    def get_eval_prompt(self, sample: Dict[str, Any]) -> str:
        """
        获取评估提示词
        
        Args:
            sample: 数据样本
            
        Returns:
            str: 格式化后的提示词
        """
        prompt = sample.get(self.config.prompt_key, "")
        return prompt

    def wrap_eval_prompt_with_qwen_template(self, prompt: str, enable_thinking: bool = None) -> str:
        """
        将评估提示词包装成 Qwen 聊天模板格式
        
        Args:
            prompt: 原始提示词
            enable_thinking: 是否启用思考模式，默认使用配置中的设置
            
        Returns:
            str: 包装后的提示词
        """
        if enable_thinking is None:
            enable_thinking = self.config.enable_thinking
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        
        return text

    @torch.inference_mode()
    def batch_inference_with_reasoning(self, samples: List[Dict[str, Any]]) -> torch.Tensor:
        """
        带思考模式的批量推理
        
        Args:
            samples: 样本列表，每个样本包含 prompt
            
        Returns:
            torch.Tensor: 模型输出的 token IDs
        """
        prompts = [self.get_eval_prompt(sample) for sample in samples]
        
        wrapped_prompts = [
            self.wrap_eval_prompt_with_qwen_template(prompt, enable_thinking=True)
            for prompt in prompts
        ]
        
        inputs = self.tokenizer(
            wrapped_prompts,
            return_tensors="pt",
            padding=True,
            padding_side=self.config.padding_side,
            truncation=True,
            max_length=self.config.max_length if hasattr(self.config, 'max_length') else 4096
        )
        
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)
        
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_new_tokens if hasattr(self.config, 'max_new_tokens') else 1024,
            temperature=self.config.temperature if hasattr(self.config, 'temperature') else 0.0,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        return outputs

    @torch.inference_mode()
    def batch_inference(self, samples: List[Dict[str, Any]]) -> torch.Tensor:
        """
        不带思考模式的批量推理
        
        Args:
            samples: 样本列表，每个样本包含 prompt
            
        Returns:
            torch.Tensor: 模型输出的 token IDs
        """
        prompts = [self.get_eval_prompt(sample) for sample in samples]
        
        wrapped_prompts = [
            self.wrap_eval_prompt_with_qwen_template(prompt, enable_thinking=False)
            for prompt in prompts
        ]
        
        inputs = self.tokenizer(
            wrapped_prompts,
            return_tensors="pt",
            padding=True,
            padding_side=self.config.padding_side,
            truncation=True,
            max_length=self.config.max_length if hasattr(self.config, 'max_length') else 4096
        )
        
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)
        
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_new_tokens if hasattr(self.config, 'max_new_tokens') else 1024,
            temperature=self.config.temperature if hasattr(self.config, 'temperature') else 0.0,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        return outputs


    def evaluate_batch_samples(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算数学题准确率的评估函数
        
        基于AIME和GSM8K等数据集的评估方式：
        - 使用Exact Match (EM)指标：生成的答案必须与标准答案完全匹配
        - 支持从模型输出中提取最终答案
        - 支持数值比较和字符串匹配
        
        Args:
            eval_preds: 评估预测结果，包含 predictions 和 label_ids
            tokenizer: 分词器实例，用于解码文本
            
        Returns:
            dict: 包含准确率指标的字典
        """

        if self.config.enable_thinking:
            predictions = self.batch_inference_with_reasoning(samples)
        else:
            predictions = self.batch_inference(samples)
        
        labels = [sample[self.config.ground_true_answer_key] for sample in samples]
        
        # 解码预测结果和标签
        # 需要使用argmax获取预测的token ID，然后进行解码
        if hasattr(predictions, 'argmax'):
            pred_ids = predictions.argmax(axis=-1)
        else:
            pred_ids = predictions
        
        decoded_preds = self.tokenizer.batch_decode(
            pred_ids, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        # 解码标签（用于调试，通常不需要显示）
        decoded_labels = self.tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        # 从模型输出中提取最终答案
        extracted_answers = []
        standard_answers = []
        
        for pred_text, label_text in zip(decoded_preds, decoded_labels):
            # 从预测文本中提取答案
            pred_answer = self._extract_final_answer_from_text(pred_text)
            extracted_answers.append(pred_answer)
            
            # 从标签文本中提取标准答案
            label_answer = self._extract_final_answer_from_text(label_text)
            standard_answers.append(label_answer)
        
        # 计算Exact Match准确率
        exact_matches = 0
        total_samples = len(extracted_answers)
        
        for pred_answer, std_answer in zip(extracted_answers, standard_answers):
            if self._is_answer_correct_logic(pred_answer, std_answer):
                exact_matches += 1
        
        # 计算准确率
        accuracy = exact_matches / total_samples if total_samples > 0 else 0.0
        
        return {
            "math_accuracy": float(accuracy),
            "accuracy": float(accuracy),
            "exact_match": float(accuracy),
            "total_samples": int(total_samples),
            "correct_predictions": int(exact_matches)
        }
    




    def evaluate(self) -> Dict[str, float]:
        """
        评估整个测试数据集
        
        Returns:
            dict: 包含整体评估指标的字典
        """
        self.logger.info("开始评估...")
        
        all_results = []
        total_correct = 0
        total_samples = 0
        
        for batch_idx, samples in enumerate(tqdm(self.test_dataloader, desc="评估进度")):
            batch_results = self.evaluate_batch_samples(samples)
            all_results.append(batch_results)
            
            total_correct += batch_results["correct_predictions"]
            total_samples += batch_results["total_samples"]
        
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        final_results = {
            "overall_accuracy": float(overall_accuracy),
            "total_samples": int(total_samples),
            "total_correct": int(total_correct),
            "math_accuracy": float(overall_accuracy),
            "exact_match": float(overall_accuracy)
        }
        
        self.logger.info(f"评估完成: 准确率 = {overall_accuracy:.4f} ({total_correct}/{total_samples})")
        
        return final_results



    def _is_answer_correct_logic(self, predicted_answer: str, standard_answer: str):
        """
        判断预测答案是否正确
        
        支持多种答案格式的比较：
        1. 精确字符串匹配
        2. 数值比较（处理小数和分数）
        3. 数学表达式计算结果比较
        
        Args:
            predicted_answer: 预测的答案
            standard_answer: 标准答案
            
        Returns:
            bool: 答案是否正确
        """
        import re
        
        if not predicted_answer or not standard_answer:
            return False
        
        # 清理答案
        predicted_answer = predicted_answer.strip()
        standard_answer = standard_answer.strip()
        
        # 1. 直接字符串匹配
        if predicted_answer.lower() == standard_answer.lower():
            return True
        
        # 2. 尝试数值比较
        try:
            # 提取数字
            pred_nums = re.findall(r'[0-9]+\.?[0-9]*', predicted_answer)
            std_nums = re.findall(r'[0-9]+\.?[0-9]*', standard_answer)
            
            if pred_nums and std_nums:
                # 尝试将答案转换为数值
                pred_val = float(pred_nums[0])
                std_val = float(std_nums[0])
                
                # 数值相等检查（考虑浮点精度）
                if abs(pred_val - std_val) < 1e-6:
                    return True
        except (ValueError, IndexError):
            pass
        
        # 3. 分数比较（如 3/4 = 0.75）
        try:
            # 检查是否是分数格式
            if '/' in predicted_answer and '/' in standard_answer:
                pred_parts = predicted_answer.split('/')
                std_parts = standard_answer.split('/')
                
                if len(pred_parts) == 2 and len(std_parts) == 2:
                    pred_val = float(pred_parts[0]) / float(pred_parts[1])
                    std_val = float(std_parts[0]) / float(std_parts[1])
                    
                    if abs(pred_val - std_val) < 1e-6:
                        return True
        except (ValueError, ZeroDivisionError):
            pass
        
        # 4. 移除空格和标点符号后比较
        cleaned_pred = re.sub(r'[^\w]', '', predicted_answer.lower())
        cleaned_std = re.sub(r'[^\w]', '', standard_answer.lower())
        
        if cleaned_pred == cleaned_std:
            return True
        
        return False

    def _extract_final_answer_from_text(self, text: str):
        """
        从文本中提取最终答案
        
        支持多种答案格式：
        1. 直接数字答案：如 "答案是 42"、"答案是: 3.14"
        2. 数学表达式：如 "答案: 2+3=5"、"答案是 2+3"
        3. 分数答案：如 "答案是 3/4"
        4. 括号答案：如 "答案 (42)"、"答案是(3)"
        
        Args:
            text: 输入文本
            
        Returns:
            str: 提取的答案字符串
        """
        import re
        
        if not text:
            return ""
        
        # 清理文本
        text = text.strip()
        
        # 提取答案的多种模式
        patterns = [
            r'答案是?\s*[:：]?\s*([^\n\r]*)',  # "答案是："或"答案："
            r'答案\s*[:：]?\s*([^\n\r]*)',     # "答案："  
            r'答[:：]\s*([^\n\r]*)',          # "答："
            r'最终答案\s*[:：]?\s*([^\n\r]*)', # "最终答案："
            r'答案\s*\(([^)]*)\)',           # "答案(42)"
            r'最终答案\s*\(([^)]*)\)',        # "最终答案(42)"
            r'答案是?\s*([0-9]+\.?[0-9]*\s*[+\-*/]\s*[0-9]+\.?[0-9]*)',  # 数学表达式
            r'答案是?\s*([0-9]+\.?[0-9]*)',    # 简单数字
            r'答案\s*([0-9]+\.?[0-9]*)',      # 简单数字
            
            # 英文格式答案
            r'(?i)answer\s*[:：]?\s*([^\n\r]*)',  # "Answer:" 或 "answer："
            r'(?i)the answer\s*[:：]?\s*([^\n\r]*)',  # "The answer:"
            r'(?i)final answer\s*[:：]?\s*([^\n\r]*)',  # "Final answer:"
            
            # 负数答案
            r'[负-]\s*([0-9]+\.?[0-9]*)',      # "负5" 或 "-5"
            r'负\s*数\s*[:：]?\s*([^\n\r]*)',   # "负数："
            
            # 百分数答案
            r'([0-9]+\.?[0-9]*)\s*%',          # "50%" 或 "0.5%"
            r'([0-9]+\.?[0-9]*)\s*个百分点',    # "50个百分点"
            r'([0-9]+\.?[0-9]*)\s*百分',       # "50百分"
            
            # 科学计数法
            r'([0-9]+\.?[0-9]*[eE][+\-]?[0-9]+)',  # "1.5e-3", "2.3E+5"
            
            # 分数答案
            r'([0-9]+\.?[0-9]*)\s*/\s*([0-9]+\.?[0-9]*)',  # "3/4"
            r'二分之一|三分之一|四分之一|五分之一|六分之一|七分之一|八分之一|九分之一|十分之一',  # 中文分数
            r'([一二三四五六七八九十]+)\s*分之\s*([一二三四五六七八九十]+)',  # "四分之三"
            
            # 带单位的数字
            r'([0-9]+\.?[0-9]*)\s*[个只条件件元角分厘毫克公斤吨米厘米毫米千米公里升毫升年天日月时分秒]',  # "42个", "100元", "50公斤"
            
            # 复数答案
            r'([0-9]+\.?[0-9]*)\s*\+\s*([0-9]+\.?[0-9]*)\s*i',  # "3+4i"
            r'([0-9]+\.?[0-9]*)\s*[+\-]\s*([0-9]+\.?[0-9]*)\s*[jJ]',  # "3+4j" 或 "3-4j"
            
            # 根号答案
            r'√\s*([0-9]+\.?[0-9]*)',        # "√16"
            r'根号\s*([0-9]+\.?[0-9]*)',      # "根号16"
            r'√\s*\(\s*([0-9]+\.?[0-9]*)\s*\)',  # "√(16)"
            
            # 罗马数字
            r'\b[IVX]+\b',                    # "I", "II", "III", "IV", "V"
            
            # 括号内的各种答案格式
            r'\(\s*([^\)]*)\s*\)',            # 任意括号内容
            r'【\s*([^\]]*)\s*】',            # 方括号内容
            r'《\s*([^\>]*)\s*》',            # 书名号内容
            r'"([^"]*)"',                     # 双引号内容
            r"'([^']*)'",                     # 单引号内容
            
            # 更多数学符号
            r'≈\s*([0-9]+\.?[0-9]*)',         # "≈3.14"
            r'约\s*([0-9]+\.?[0-9]*)',        # "约3.14"
            r'大约\s*([0-9]+\.?[0-9]*)',      # "大约3.14"
            r'大概\s*([0-9]+\.?[0-9]*)',      # "大概3.14"
            
            # 中文数字
            r'[零一二三四五六七八九十百千万亿]+',  # "一百二十三"
            r'[0-9]+\.?[0-9]*\s*[、，,\s]*\s*[0-9]+\.?[0-9]*\s*[、，,\s]*\s*[0-9]+\.?[0-9]*',  # "1、2、3" 或 "1，2，3"
            
            # 比例答案
            r'([0-9]+\.?[0-9]*)\s*[:：]\s*([0-9]+\.?[0-9]*)',  # "3:4" 或 "3：4"
            r'([0-9]+\.?[0-9]*)\s*比\s*([0-9]+\.?[0-9]*)',    # "3比4"
            
            # 幂运算
            r'([0-9]+\.?[0-9]*)\s*\^\s*([0-9]+\.?[0-9]*)',    # "2^3"
            r'([0-9]+\.?[0-9]*)\s*\*\*\s*([0-9]+\.?[0-9]*)',  # "2**3"
            
            # 更多英文数学表达
            r'(?i)equals?\s*[:：]?\s*([^\n\r]*)',  # "equals:" 或 "equal:"
            r'(?i)result\s*[:：]?\s*([^\n\r]*)',   # "result:"
            r'(?i)solution\s*[:：]?\s*([^\n\r]*)', # "solution:"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                answer = match.group(1).strip()
                # 清理答案，移除多余的符号
                answer = re.sub(r'[。！？.,，；：\s]+$', '', answer)
                if answer:
                    return answer
        
        # 如果没有找到明确的答案标记，尝试提取末尾的数字
        numbers = re.findall(r'[0-9]+\.?[0-9]*', text)
        if numbers:
            return numbers[-1]  # 返回最后一个数字作为答案
        
        return ""








def run():
    pass







if __name__ == '__main__':
    run()
