# utils/metrics.py

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import torch
from utils.logger import setup_logger
from transformers import AutoTokenizer

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
            f"P@{k}": [] for k in k_values
        }
        # 添加其他指标列表
        for k in k_values:
            metrics[f"R@{k}"] = []
            metrics[f"NDCG@{k}"] = []
        
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





class MathEvaluator:
    # 定义评估函数，用于计算数学题准确率
    def compute_math_accuracy(self, eval_preds, tokenizer: AutoTokenizer):
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
        import re
        import numpy as np
        
        predictions, labels = eval_preds
        
        # 解码预测结果和标签
        # 需要使用argmax获取预测的token ID，然后进行解码
        if hasattr(predictions, 'argmax'):
            pred_ids = predictions.argmax(axis=-1)
        else:
            pred_ids = predictions
        
        decoded_preds = tokenizer.batch_decode(
            pred_ids, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        # 解码标签（用于调试，通常不需要显示）
        decoded_labels = tokenizer.batch_decode(
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

