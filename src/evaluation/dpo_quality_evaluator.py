import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.logger import setup_logger
from utils.metrics import DPOMetrics
from config.evaluator_config import DPOQualityEvaluatorConfig
from evaluation.base_evaluator import BaseEvaluator, EvaluatorDataset



class EvaluatorDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class DPOQualityEvaluator(BaseEvaluator):
    def __init__(
        self,
        config: DPOQualityEvaluatorConfig
    ):
        super().__init__(config)
        
        self.config = config
        self.logger = setup_logger(name=__class__.__name__, level="INFO")
        
        self.reference_model: Optional[PreTrainedModel] = None
        self.reference_tokenizer: Optional[PreTrainedTokenizer] = None
        
        self.beta = config.beta
        self.max_length = config.max_length
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(self.config.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True
        )
        
        if self.config.reference_model_path:
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                self.config.reference_model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            ).to(self.config.device)
            self.reference_model.eval()
        
        # 确保模型处于评估模式
        self.model.eval()
        
        self.metrics = DPOMetrics()
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        self.logger.info(f"DPO质量评估器初始化完成，模型路径: {self.config.model_name_or_path}")
    
    def prepare_dpo_batch(
        self,
        query_text: str,
        chosen_text: str,
        rejected_text: str
    ) -> Dict[str, torch.Tensor]:
        query_inputs = self.tokenizer(
            query_text,
            truncation=True,
            max_length=self.max_length // 2,
            padding="max_length",
            return_tensors="pt"
        )
        
        chosen_inputs = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length // 2,
            padding="max_length",
            return_tensors="pt"
        )
        
        rejected_inputs = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length // 2,
            padding="max_length",
            return_tensors="pt"
        )
        
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
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        logits = outputs.logits
        
        query_len = input_ids.shape[1]
        response_logits = logits[:, query_len-1:-1, :]
        
        response_ids = response_ids[:, 1:]
        
        log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1)
        
        token_log_probs = torch.gather(log_probs, dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)
        
        response_mask = attention_mask[:, 1:] - input_ids.shape[1]
        response_mask = (response_mask > 0).float()
        
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
            chosen_rewards = policy_chosen_logps
            rejected_rewards = policy_rejected_logps
        
        return {
            "policy_chosen_logps": policy_chosen_logps,
            "policy_rejected_logps": policy_rejected_logps,
            "chosen_rewards": chosen_rewards,
            "rejected_rewards": rejected_rewards
        }
    
    def evaluate_one_sample(self, sample: Dict[str, Any]) -> Dict[str, float]:
        query = sample.get(self.config.query_key, sample.get("query", ""))
        chosen = sample.get(self.config.chosen_key, sample.get("chosen", ""))
        rejected = sample.get(self.config.rejected_key, sample.get("rejected", ""))
        
        batch = self.prepare_dpo_batch(query, chosen, rejected)
        
        rewards = self.compute_rewards(batch)
        
        reward_accuracy = self.metrics.calculate_reward_accuracy(
            rewards["policy_chosen_logps"],
            rewards["policy_rejected_logps"]
        )
        
        rewards_stats = self.metrics.calculate_rewards_stats(
            rewards["chosen_rewards"],
            rewards["rejected_rewards"]
        )
        
        metrics = {
            "reward_accuracy": reward_accuracy,
            **rewards_stats
        }
        
        return metrics
    
    def evaluate_batch_examples(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        all_metrics = []
        
        for sample in samples:
            sample_metrics = self.evaluate_one_sample(sample)
            all_metrics.append(sample_metrics)
        
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        
        return avg_metrics
    
    def evaluate(self) -> Dict[str, float]:
        all_metrics = []
        
        self.logger.info(f"开始评估DPO质量，数据集大小: {len(self.test_data)}")
        
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="Evaluating DPO quality"):
                batch_dict = [sample for sample in batch]
                batch_metrics = self.evaluate_batch_examples(batch_dict)
                all_metrics.append(batch_metrics)
        
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        
        self.logger.info(f"DPO质量评估完成，奖励准确率: {avg_metrics.get('reward_accuracy', 0):.4f}")
        
        self.save_evaluation_results(avg_metrics)
        
        return avg_metrics
    
    def generate_preference_predictions(
        self,
        examples: List[Dict[str, Union[str, List[str]]]]
    ) -> List[Dict[str, Any]]:
        predictions = []
        
        self.logger.info(f"开始生成偏好预测，示例数量: {len(examples)}")
        
        for example in tqdm(examples, desc="Generating preference predictions"):
            query = example["query"]
            responses = example["responses"]
            
            response_logps = []
            
            for response in responses:
                batch = self.prepare_dpo_batch(query, response, response)
                logp = self.compute_logps(
                    self.model,
                    batch["query_ids"],
                    batch["query_attention_mask"],
                    batch["chosen_ids"]
                )
                response_logps.append(logp.item())
            
            sorted_indices = np.argsort(response_logps)[::-1]
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
        n = len(responses)
        comparison_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                batch = self.prepare_dpo_batch(query, responses[i], responses[j])
                
                rewards = self.compute_rewards(batch)
                
                logits = self.beta * (rewards["chosen_rewards"] - rewards["rejected_rewards"])
                prob_i_over_j = torch.sigmoid(logits).item()
                
                comparison_matrix[i, j] = prob_i_over_j
                comparison_matrix[j, i] = 1 - prob_i_over_j
        
        np.fill_diagonal(comparison_matrix, 0.5)
        
        return comparison_matrix
    
    def visualize_comparison_matrix(
        self,
        comparison_matrix: np.ndarray,
        responses: List[str],
        query: str,
        save_path: Optional[str] = None
    ) -> None:
        n = len(responses)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(comparison_matrix, cmap='coolwarm', vmin=0, vmax=1)
        
        plt.colorbar(label='Preference Probability')
        
        plt.xticks(np.arange(n), [f"Response {i+1}" for i in range(n)], rotation=45)
        plt.yticks(np.arange(n), [f"Response {i+1}" for i in range(n)])
        
        for i in range(n):
            for j in range(n):
                plt.text(j, i, f"{comparison_matrix[i, j]:.2f}",
                        ha="center", va="center", color="black")
        
        plt.title(f"Pairwise Preference Matrix\nQuery: {query[:50]}...")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"比较矩阵可视化已保存至: {save_path}")
        else:
            plt.show()
    
    def save_evaluation_results(self, metrics: Dict[str, float]) -> None:
        results_path = os.path.join(self.config.output_dir, "dpo_evaluation_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"评估结果已保存至: {results_path}")
        
        report_path = os.path.join(self.config.output_dir, "dpo_evaluation_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("DPO质量评估报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"奖励准确率: {metrics.get('reward_accuracy', 0):.4f}\n")
            f.write(f"chosen奖励均值: {metrics.get('chosen_rewards_mean', 0):.4f}\n")
            f.write(f"rejected奖励均值: {metrics.get('rejected_rewards_mean', 0):.4f}\n")
            f.write(f"奖励差均值: {metrics.get('rewards_margin_mean', 0):.4f}\n")
            f.write(f"奖励差标准差: {metrics.get('rewards_margin_std', 0):.4f}\n")
        
        self.logger.info(f"评估报告已保存至: {report_path}")
    
    def visualize_reward_distribution(
        self,
        chosen_rewards: List[float],
        rejected_rewards: List[float],
        save_path: Optional[str] = None
    ) -> None:
        plt.figure(figsize=(10, 6))
        
        plt.hist(chosen_rewards, alpha=0.5, label='Chosen', bins=30)
        plt.hist(rejected_rewards, alpha=0.5, label='Rejected', bins=30)
        
        plt.axvline(np.mean(chosen_rewards), color='blue', linestyle='dashed', linewidth=1)
        plt.axvline(np.mean(rejected_rewards), color='orange', linestyle='dashed', linewidth=1)
        
        plt.title('Reward Distribution')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"奖励分布可视化已保存至: {save_path}")
        else:
            plt.show()


def run():
    config = DPOQualityEvaluatorConfig(
        model_name_or_path="Qwen/Qwen3-4B",
        device="cuda:0",
        test_dataset_path="data/dpo/test.json",
        query_key="query",
        chosen_key="chosen",
        rejected_key="rejected",
        per_device_eval_batch_size=4,
        output_dir="dpo_evaluation_results",
        beta=0.1,
        max_length=512
    )
    
    evaluator = DPOQualityEvaluator(config)
    
    results = evaluator.evaluate()
    print("评估完成！")
    print(f"评估结果: {results}")


if __name__ == "__main__":
    run()
