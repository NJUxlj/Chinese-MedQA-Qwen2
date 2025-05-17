# src/evaluation/dpo_quality_evaluator.py

import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from utils.logger import setup_logger
from utils.metrics import DPOMetrics

logger = setup_logger(__name__)

class DPOQualityEvaluator:
    """DPO质量评估器，用于评估DPO训练的效果和质量"""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        reference_model: Optional[PreTrainedModel] = None,
        beta: float = 0.1,
        max_length: int = 512,
        device: Optional[str] = None,
        output_dir: str = "dpo_evaluation_results"
    ):
        """
        初始化DPO质量评估器
        
        Args:
            model: 训练后的DPO模型
            tokenizer: 分词器
            reference_model: 参考模型（如果有）
            beta: DPO中的beta参数
            max_length: 最大序列长度
            device: 设备（"cuda"或"cpu"）
            output_dir: 评估结果输出目录
        """
        self.model = model
        self.tokenizer = tokenizer
        self.reference_model = reference_model
        self.beta = beta
        self.max_length = max_length
        
        # 设置设备
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model.to(self.device)
        if self.reference_model:
            self.reference_model.to(self.device)
            # 确保参考模型处于评估模式
            self.reference_model.eval()
        
        # 确保模型处于评估模式
        self.model.eval()
        
        # 评估指标
        self.metrics = DPOMetrics()
        
        # 输出目录
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def prepare_dpo_batch(
        self,
        query_text: str,
        chosen_text: str,
        rejected_text: str
    ) -> Dict[str, torch.Tensor]:
        """
        准备DPO评估所需的批次数据
        
        Args:
            query_text: 查询文本
            chosen_text: 选择的回答
            rejected_text: 拒绝的回答
            
        Returns:
            包含编码后输入的字典
        """
        # 编码查询
        query_inputs = self.tokenizer(
            query_text,
            truncation=True,
            max_length=self.max_length // 2,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 编码选择的回答
        chosen_inputs = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length // 2,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 编码拒绝的回答
        rejected_inputs = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length // 2,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 将所有输入移至设备
        batch = {
            "query_ids": query_inputs.input_ids.to(self.device),
            "query_attention_mask": query_inputs.attention_mask.to(self.device),
            "chosen_ids": chosen_inputs.input_ids.to(self.device),
            "chosen_attention_mask": chosen_inputs.attention_mask.to(self.device),
            "rejected_ids": rejected_inputs.input_ids.to(self.device),
            "rejected_attention_mask": rejected_inputs.attention_mask.to(self.device)
        }
        
        return batch
    
    def compute_logps(
        self,
        model: PreTrainedModel,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        response_ids: torch.LongTensor
    ) -> torch.Tensor:
        """
        计算回答的对数概率
        
        Args:
            model: 模型
            input_ids: 输入ID
            attention_mask: 注意力掩码
            response_ids: 回答ID
            
        Returns:
            对数概率
        """
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        logits = outputs.logits
        
        # 将输入部分的logits屏蔽掉，只关注response部分
        query_len = input_ids.shape[1]
        response_logits = logits[:, query_len-1:-1, :]  # -1是因为我们要预测下一个token
        
        # 忽略第一个token（通常是BOS或填充）
        response_ids = response_ids[:, 1:]
        
        log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1)
        
        # 收集每个位置的目标token的log概率
        token_log_probs = torch.gather(log_probs, dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)
        
        # 应用attention mask只考虑实际的token
        response_mask = attention_mask[:, 1:] - input_ids.shape[1]
        response_mask = (response_mask > 0).float()
        
        # 计算每个序列的平均log概率
        seq_log_probs = (token_log_probs * response_mask).sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1e-5)
        
        return seq_log_probs
    
    def compute_rewards(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算DPO奖励
        
        Args:
            batch: 批次数据
            
        Returns:
            包含奖励的字典
        """
        # 计算策略模型的logps
        policy_chosen_logps = self.compute_logps(
            self.model,
            batch["query_ids"],
            batch["query_attention_mask"],
            batch["chosen_ids"]
        )
        
        policy_rejected_logps = self.compute_logps(
            self.model,
            batch["query_ids"],
            batch["query_attention_mask"],
            batch["rejected_ids"]
        )
        
        # 如果有参考模型，计算参考模型的logps
        if self.reference_model:
            with torch.no_grad():
                reference_chosen_logps = self.compute_logps(
                    self.reference_model,
                    batch["query_ids"],
                    batch["query_attention_mask"],
                    batch["chosen_ids"]
                )
                
                reference_rejected_logps = self.compute_logps(
                    self.reference_model,
                    batch["query_ids"],
                    batch["query_attention_mask"],
                    batch["rejected_ids"]
                )
            
            # 计算奖励
            chosen_rewards = policy_chosen_logps - reference_chosen_logps
            rejected_rewards = policy_rejected_logps - reference_rejected_logps
        else:
            # 参考自由DPO
            chosen_rewards = policy_chosen_logps
            rejected_rewards = policy_rejected_logps
        
        return {
            "policy_chosen_logps": policy_chosen_logps,
            "policy_rejected_logps": policy_rejected_logps,
            "chosen_rewards": chosen_rewards,
            "rejected_rewards": rejected_rewards
        }
    
    def evaluate_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        评估单个批次
        
        Args:
            batch: 批次数据
            
        Returns:
            评估指标
        """
        # 计算奖励
        rewards = self.compute_rewards(batch)
        
        # 计算指标
        reward_accuracy = self.metrics.calculate_reward_accuracy(
            rewards["policy_chosen_logps"],
            rewards["policy_rejected_logps"]
        )
        
        rewards_stats = self.metrics.calculate_rewards_stats(
            rewards["chosen_rewards"],
            rewards["rejected_rewards"]
        )
        
        # 合并所有指标
        metrics = {
            "reward_accuracy": reward_accuracy,
            **rewards_stats
        }
        
        return metrics
    
    def evaluate_dataset(self, dataset: Dataset) -> Dict[str, float]:
        """
        评估整个数据集
        
        Args:
            dataset: 评估数据集
            
        Returns:
            整体评估指标
        """
        all_metrics = []
        
        logger.info(f"开始评估DPO质量，数据集大小: {len(dataset)}")
        
        for example in tqdm(dataset, desc="Evaluating DPO quality"):
            # 准备批次
            batch = self.prepare_dpo_batch(
                example["query"],
                example["chosen"],
                example["rejected"]
            )
            
            # 评估批次
            batch_metrics = self.evaluate_batch(batch)
            all_metrics.append(batch_metrics)
        
        # 计算平均指标
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        
        logger.info(f"DPO质量评估完成，奖励准确率: {avg_metrics['reward_accuracy']:.4f}")
        
        # 保存评估结果
        self.save_evaluation_results(avg_metrics)
        
        return avg_metrics
    
    def evaluate_examples(
        self,
        examples: List[Dict[str, str]]
    ) -> Dict[str, float]:
        """
        评估示例列表
        
        Args:
            examples: 示例列表，每个示例是包含query、chosen和rejected的字典
            
        Returns:
            评估指标
        """
        all_metrics = []
        
        logger.info(f"开始评估DPO质量，示例数量: {len(examples)}")
        
        for example in tqdm(examples, desc="Evaluating DPO quality"):
            # 准备批次
            batch = self.prepare_dpo_batch(
                example["query"],
                example["chosen"],
                example["rejected"]
            )
            
            # 评估批次
            batch_metrics = self.evaluate_batch(batch)
            all_metrics.append(batch_metrics)
        
        # 计算平均指标
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        
        logger.info(f"DPO质量评估完成，奖励准确率: {avg_metrics['reward_accuracy']:.4f}")
        
        # 保存评估结果
        self.save_evaluation_results(avg_metrics)
        
        return avg_metrics
    
    def generate_preference_predictions(
        self,
        examples: List[Dict[str, Union[str, List[str]]]]
    ) -> List[Dict[str, Any]]:
        """
        生成偏好预测
        
        Args:
            examples: 示例列表，每个示例是包含query和responses列表的字典
            
        Returns:
            预测结果列表
        """
        predictions = []
        
        logger.info(f"开始生成偏好预测，示例数量: {len(examples)}")
        
        for example in tqdm(examples, desc="Generating preference predictions"):
            query = example["query"]
            responses = example["responses"]
            
            # 计算每个回答的logp
            response_logps = []
            
            for response in responses:
                batch = self.prepare_dpo_batch(query, response, response)  # 使用相同的回答作为chosen和rejected
                logp = self.compute_logps(
                    self.model,
                    batch["query_ids"],
                    batch["query_attention_mask"],
                    batch["chosen_ids"]
                )
                response_logps.append(logp.item())
            
            # 根据logp排序
            sorted_indices = np.argsort(response_logps)[::-1]  # 降序
            sorted_responses = [responses[i] for i in sorted_indices]
            sorted_logps = [response_logps[i] for i in sorted_indices]
            
            predictions.append({
                "query": query,
                "ranked_responses": sorted_responses,
                "logps": sorted_logps,
                "preferred_response": sorted_responses[0]
            })
        
        return predictions
    
    def generate_comparison_matrix(
        self,
        responses: List[str],
        query: str
    ) -> np.ndarray:
        """
        生成回答间的比较矩阵
        
        Args:
            responses: 回答列表
            query: 查询
            
        Returns:
            比较矩阵，其中matrix[i][j]表示模型偏好responses[i]高于responses[j]的概率
        """
        n = len(responses)
        comparison_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # 准备批次
                batch = self.prepare_dpo_batch(query, responses[i], responses[j])
                
                # 计算奖励
                rewards = self.compute_rewards(batch)
                
                # 计算模型偏好i高于j的概率
                logits = self.beta * (rewards["chosen_rewards"] - rewards["rejected_rewards"])
                prob_i_over_j = torch.sigmoid(logits).item()
                
                comparison_matrix[i, j] = prob_i_over_j
                comparison_matrix[j, i] = 1 - prob_i_over_j
        
        # 对角线上的值设为0.5（与自身比较）
        np.fill_diagonal(comparison_matrix, 0.5)
        
        return comparison_matrix
    
    def visualize_comparison_matrix(
        self,
        comparison_matrix: np.ndarray,
        responses: List[str],
        query: str,
        save_path: Optional[str] = None
    ) -> None:
        """
        可视化比较矩阵
        
        Args:
            comparison_matrix: 比较矩阵
            responses: 回答列表
            query: 查询
            save_path: 保存路径
        """
        n = len(responses)
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        plt.imshow(comparison_matrix, cmap='coolwarm', vmin=0, vmax=1)
        
        # 添加颜色条
        plt.colorbar(label='Preference Probability')
        
        # 设置坐标轴
        plt.xticks(np.arange(n), [f"Response {i+1}" for i in range(n)], rotation=45)
        plt.yticks(np.arange(n), [f"Response {i+1}" for i in range(n)])
        
        # 添加数值标签
        for i in range(n):
            for j in range(n):
                plt.text(j, i, f"{comparison_matrix[i, j]:.2f}",
                        ha="center", va="center", color="black")
        
        # 设置标题
        plt.title(f"Pairwise Preference Matrix\nQuery: {query[:50]}...")
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path)
            logger.info(f"比较矩阵可视化已保存至: {save_path}")
        else:
            plt.show()
    
    def save_evaluation_results(self, metrics: Dict[str, float]) -> None:
        """
        保存评估结果
        
        Args:
            metrics: 评估指标
        """
        # 保存为JSON
        results_path = os.path.join(self.output_dir, "dpo_evaluation_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估结果已保存至: {results_path}")
        
        # 生成摘要报告
        report_path = os.path.join(self.output_dir, "dpo_evaluation_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("DPO质量评估报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"奖励准确率: {metrics['reward_accuracy']:.4f}\n")
            f.write(f"chosen奖励均值: {metrics['chosen_rewards_mean']:.4f}\n")
            f.write(f"rejected奖励均值: {metrics['rejected_rewards_mean']:.4f}\n")
            f.write(f"奖励差均值: {metrics['rewards_margin_mean']:.4f}\n")
            f.write(f"奖励差标准差: {metrics['rewards_margin_std']:.4f}\n")
        
        logger.info(f"评估报告已保存至: {report_path}")
    
    def visualize_reward_distribution(
        self,
        chosen_rewards: List[float],
        rejected_rewards: List[float],
        save_path: Optional[str] = None
    ) -> None:
        """
        可视化奖励分布
        
        Args:
            chosen_rewards: chosen奖励列表
            rejected_rewards: rejected奖励列表
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 6))
        
        # 绘制直方图
        plt.hist(chosen_rewards, alpha=0.5, label='Chosen', bins=30)
        plt.hist(rejected_rewards, alpha=0.5, label='Rejected', bins=30)
        
        # 添加垂直线表示均值
        plt.axvline(np.mean(chosen_rewards), color='blue', linestyle='dashed', linewidth=1)
        plt.axvline(np.mean(rejected_rewards), color='orange', linestyle='dashed', linewidth=1)
        
        # 添加标题和标签
        plt.title('Reward Distribution')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.legend()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path)
            logger.info(f"奖励分布可视化已保存至: {save_path}")
        else:
            plt.show()