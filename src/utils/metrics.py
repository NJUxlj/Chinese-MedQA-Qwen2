# utils/metrics.py

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import torch
from utils.logger import setup_logger

logger = setup_logger(__name__)

class ModelEvaluator:
    """模型评估器，用于评估模型性能"""
    
    def __init__(self):
        """初始化评估器"""
        self.rouge = Rouge()
        self.smoothing = SmoothingFunction().method1
    
    def calculate_accuracy(self, predictions: List[int], references: List[int]) -> float:
        """计算分类准确率"""
        return accuracy_score(references, predictions)
    
    def calculate_precision_recall_f1(
        self, 
        predictions: List[int], 
        references: List[int],
        average: str = "weighted"
    ) -> Dict[str, float]:
        """计算精确率、召回率和F1分数"""
        precision = precision_score(references, predictions, average=average, zero_division=0)
        recall = recall_score(references, predictions, average=average, zero_division=0)
        f1 = f1_score(references, predictions, average=average, zero_division=0)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def calculate_bleu(
        self, 
        predictions: List[str], 
        references: List[List[str]],
        max_ngram: int = 4
    ) -> Dict[str, float]:
        """计算BLEU分数"""
        if len(predictions) != len(references):
            raise ValueError("预测和参考数量不匹配")
        
        bleu_scores = {}
        
        for n in range(1, max_ngram + 1):
            weights = [0] * max_ngram
            for i in range(n):
                weights[i] = 1.0 / n
            
            score_sum = 0
            for pred, ref in zip(predictions, references):
                # 转换为token列表
                pred_tokens = pred.split()
                ref_tokens = [r.split() for r in ref]
                
                try:
                    score = sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=self.smoothing)
                    score_sum += score
                except Exception as e:
                    logger.warning(f"BLEU计算错误: {e}")
                    continue
            
            bleu_scores[f"bleu-{n}"] = score_sum / len(predictions)
        
        return bleu_scores
    
    def calculate_rouge(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """计算ROUGE分数"""
        if len(predictions) != len(references):
            raise ValueError("预测和参考数量不匹配")
        
        try:
            scores = self.rouge.get_scores(predictions, references, avg=True)
            return scores
        except Exception as e:
            logger.error(f"ROUGE计算错误: {e}")
            return {}
    
    def calculate_perplexity(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> float:
        """计算困惑度"""
        if labels is None:
            labels = input_ids.clone()
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        return perplexity
    
    def calculate_embedding_similarity(
        self,
        predictions_embeddings: np.ndarray,
        references_embeddings: np.ndarray
    ) -> np.ndarray:
        """计算嵌入相似度"""
        return cosine_similarity(predictions_embeddings, references_embeddings)
    
    def calculate_retrieval_metrics(
        self,
        relevance_scores: List[List[float]],
        relevance_labels: List[List[int]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Dict[str, float]]:
        """计算检索指标（Precision@K, Recall@K, NDCG@K）"""
        metrics = {
            f"P@{k}": [],
            f"R@{k}": [],
            f"NDCG@{k}": []
        for k in k_values}
        
        for scores, labels in zip(relevance_scores, relevance_labels):
            # 按分数排序的索引
            sorted_indices = np.argsort(scores)[::-1]
            
            for k in k_values:
                # 取Top-K索引
                top_k_indices = sorted_indices[:k]
                
                # 计算Precision@K
                relevant_in_top_k = sum(1 for i in top_k_indices if labels[i] > 0)
                precision_at_k = relevant_in_top_k / k if k > 0 else 0
                metrics[f"P@{k}"].append(precision_at_k)
                
                # 计算Recall@K
                total_relevant = sum(1 for label in labels if label > 0)
                recall_at_k = relevant_in_top_k / total_relevant if total_relevant > 0 else 0
                metrics[f"R@{k}"].append(recall_at_k)
                
                # 计算NDCG@K
                dcg = sum((2 ** labels[i] - 1) / np.log2(j + 2) for j, i in enumerate(top_k_indices))
                
                # 理想排序
                ideal_indices = np.argsort(labels)[::-1]
                ideal_dcg = sum((2 ** labels[i] - 1) / np.log2(j + 2) for j, i in enumerate(ideal_indices[:k]))
                
                ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
                metrics[f"NDCG@{k}"].append(ndcg)
        
        # 计算平均值
        result = {}
        for metric_name, values in metrics.items():
            result[metric_name] = np.mean(values)
        
        return result
    
    def evaluate_rag_system(
        self,
        queries: List[str],
        references: List[str],
        predictions: List[str],
        retrieved_contexts: List[List[str]],
        relevance_labels: Optional[List[List[int]]] = None
    ) -> Dict[str, Any]:
        """评估RAG系统性能"""
        results = {}
        
        # 计算生成质量指标
        predictions_tokens = [p.split() for p in predictions]
        references_tokens = [[r.split()] for r in references]
        
        # BLEU评分
        bleu_scores = self.calculate_bleu(predictions, [[r] for r in references])
        results.update(bleu_scores)
        
        # ROUGE评分
        rouge_scores = self.calculate_rouge(predictions, references)
        for rouge_type, scores in rouge_scores.items():
            for score_type, value in scores.items():
                results[f"{rouge_type}_{score_type}"] = value
        
        # 如果提供了相关性标签，计算检索指标
        if relevance_labels:
            # 假设检索分数与上下文顺序相关（排名越高分数越高）
            relevance_scores = []
            for contexts in retrieved_contexts:
                scores = [1.0 / (i + 1) for i in range(len(contexts))]
                relevance_scores.append(scores)
            
            retrieval_metrics = self.calculate_retrieval_metrics(relevance_scores, relevance_labels)
            results.update(retrieval_metrics)
        
        return results


class DPOMetrics:
    """DPO训练相关指标计算"""
    
    @staticmethod
    def calculate_reward_accuracy(
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor
    ) -> float:
        """计算奖励准确率（chosen被判断为chosen的比例）"""
        return (policy_chosen_logps > policy_rejected_logps).float().mean().item()
    
    @staticmethod
    def calculate_kl_divergence(
        policy_logps: torch.Tensor,
        reference_logps: torch.Tensor
    ) -> float:
        """计算KL散度，用于测量策略模型与参考模型的差异"""
        return (reference_logps - policy_logps).mean().item()
    
    @staticmethod
    def calculate_margin_mean(
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor
    ) -> float:
        """计算奖励差异的平均值"""
        return (chosen_rewards - rejected_rewards).mean().item()
    
    @staticmethod
    def calculate_rewards_stats(
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor
    ) -> Dict[str, float]:
        """计算奖励相关统计指标"""
        return {
            "chosen_rewards_mean": chosen_rewards.mean().item(),
            "chosen_rewards_std": chosen_rewards.std().item(),
            "rejected_rewards_mean": rejected_rewards.mean().item(),
            "rejected_rewards_std": rejected_rewards.std().item(),
            "rewards_margin_mean": (chosen_rewards - rejected_rewards).mean().item(),
            "rewards_margin_std": (chosen_rewards - rejected_rewards).std().item()
        }


# 使用示例
if __name__ == "__main__":
    evaluator = ModelEvaluator()
    
    # 分类指标示例
    predictions = [1, 0, 1, 1, 0]
    references = [1, 0, 0, 1, 0]
    
    accuracy = evaluator.calculate_accuracy(predictions, references)
    print(f"Accuracy: {accuracy}")
    
    prf = evaluator.calculate_precision_recall_f1(predictions, references)
    print(f"Precision: {prf['precision']}, Recall: {prf['recall']}, F1: {prf['f1']}")
    
    # 生成指标示例
    pred_texts = ["这是一个测试句子", "医生建议多喝水"]
    ref_texts = [["这是一句测试语句"], ["医生建议多喝水和休息"]]
    
    bleu = evaluator.calculate_bleu(pred_texts, ref_texts)
    print(f"BLEU scores: {bleu}")

