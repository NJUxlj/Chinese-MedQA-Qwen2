"""
GSPO 辅助工具模块

提供 GSPO (Sequence-level Group Relative Policy Optimization) 训练所需的辅助功能：
- 序列级相对优势计算
- 奖励函数计算
- 组内相对排序
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Union, Tuple
from collections import Counter
import re


def compute_exact_match_reward(generated: str, reference: str) -> float:
    """
    计算精确匹配奖励

    Args:
        generated: 生成的文本
        reference: 参考文本

    Returns:
        1.0 如果完全匹配，否则 0.0
    """
    return 1.0 if generated.strip() == reference.strip() else 0.0


def compute_edit_distance(s1: str, s2: str) -> int:
    """
    计算两个字符串之间的编辑距离（Levenshtein Distance）

    Args:
        s1: 字符串1
        s2: 字符串2

    Returns:
        编辑距离
    """
    if len(s1) < len(s2):
        return compute_edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_edit_distance_reward(generated: str, reference: str) -> float:
    """
    计算基于编辑距离的相似度奖励

    Args:
        generated: 生成的文本
        reference: 参考文本

    Returns:
        相似度分数 [0, 1]
    """
    if not generated or not reference:
        return 0.0

    distance = compute_edit_distance(generated, reference)
    max_len = max(len(generated), len(reference))
    similarity = 1.0 - (distance / max_len) if max_len > 0 else 1.0

    return max(0.0, similarity)


def compute_bleu_reward(generated: str, reference: str, n: int = 1) -> float:
    """
    计算基于 n-gram precision 的 BLEU 奖励

    Args:
        generated: 生成的文本
        reference: 参考文本
        n: n-gram 大小（默认 1 即 unigram）

    Returns:
        BLEU precision 分数 [0, 1]
    """
    try:
        gen_tokens = re.findall(r'\w+', generated.lower())
        ref_tokens = re.findall(r'\w+', reference.lower())

        if not gen_tokens or not ref_tokens:
            return 0.0

        if n == 1:
            gen_counter = Counter(gen_tokens)
            ref_counter = Counter(ref_tokens)
            overlap = sum((gen_counter & ref_counter).values())
            precision = overlap / len(gen_tokens) if gen_tokens else 0
            return precision

        # n-gram precision
        gen_ngrams = [tuple(gen_tokens[i:i+n]) for i in range(len(gen_tokens)-n+1)]
        ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)]

        if not gen_ngrams:
            return 0.0

        gen_counter = Counter(gen_ngrams)
        ref_counter = Counter(ref_ngrams)
        overlap = sum((gen_counter & ref_counter).values())
        precision = overlap / len(gen_ngrams) if gen_ngrams else 0

        return precision

    except Exception:
        return 0.0


def compute_rouge_l_reward(generated: str, reference: str) -> float:
    """
    计算基于 ROUGE-L 的奖励

    ROUGE-L 使用最长公共子序列（LCS）计算相似度

    Args:
        generated: 生成的文本
        reference: 参考文本

    Returns:
        ROUGE-L F1 分数 [0, 1]
    """
    try:
        gen_tokens = generated.lower().split()
        ref_tokens = reference.lower().split()

        if not gen_tokens or not ref_tokens:
            return 0.0

        # 计算 LCS 长度
        m, n = len(gen_tokens), len(ref_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if gen_tokens[i-1] == ref_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        lcs_length = dp[m][n]

        # 计算 precision 和 recall
        precision = lcs_length / m if m > 0 else 0
        recall = lcs_length / n if n > 0 else 0

        # 计算 F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return f1

    except Exception:
        return 0.0


def compute_sequence_level_advantage(
    rewards: torch.Tensor,
    group_size: int = 4,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    计算序列级相对优势（GSPO 核心）

    对于同一 prompt 生成的 G 个样本，计算组内相对优势：
    A_i = (R_i - mean(R)) / (std(R) + ε)

    Args:
        rewards: 奖励张量 [batch_size]
        group_size: 每组的样本数
        eps: 防止除零的小常数

    Returns:
        相对优势张量 [batch_size]
    """
    batch_size = rewards.size(0)
    num_groups = batch_size // group_size

    if num_groups <= 1:
        # 如果只有一个组，无法计算相对优势，返回标准化奖励
        return (rewards - rewards.mean()) / (rewards.std() + eps)

    advantages = torch.zeros_like(rewards)

    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size
        group_rewards = rewards[start_idx:end_idx]

        group_mean = group_rewards.mean()
        group_std = group_rewards.std() + eps

        # 组内相对优势
        group_advantages = (group_rewards - group_mean) / group_std
        advantages[start_idx:end_idx] = group_advantages

    return advantages


def compute_group_relative_ranking(
    rewards: torch.Tensor,
    group_size: int = 4
) -> torch.Tensor:
    """
    计算组内相对排名

    对于同一 prompt 生成的 G 个样本，计算相对排名分数。
    排名分数 = (rank - 1) / (G - 1)，范围 [0, 1]

    Args:
        rewards: 奖励张量 [batch_size]
        group_size: 每组的样本数

    Returns:
        排名分数张量 [batch_size]
    """
    batch_size = rewards.size(0)
    num_groups = batch_size // group_size

    rankings = torch.zeros_like(rewards)

    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size
        group_rewards = rewards[start_idx:end_idx]

        # 计算排名（降序，奖励越高排名越靠前）
        ranks = group_rewards.argsort(descending=True).argsort()
        rankings[start_idx:end_idx] = ranks.float() / (group_size - 1) if group_size > 1 else torch.zeros_like(ranks).float()

    return rankings


def compute_token_level_advantages(
    sequence_rewards: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 1.0
) -> torch.Tensor:
    """
    计算 Token 级优势估计（使用 GAE）

    对于生成的序列中的每个 token 位置，计算累积折扣优势。

    Args:
        sequence_rewards: 序列级奖励 [batch_size, seq_len]
        gamma: 折扣因子
        lam: GAE lambda

    Returns:
        Token 级优势 [batch_size, seq_len]
    """
    batch_size, seq_len = sequence_rewards.shape
    advantages = torch.zeros_like(sequence_rewards)

    for i in range(batch_size):
        gae = 0
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                delta = sequence_rewards[i, t]
            else:
                delta = sequence_rewards[i, t] + gamma * sequence_rewards[i, t + 1] - sequence_rewards[i, t]
            gae = delta + gamma * lam * gae
            advantages[i, t] = gae

    return advantages


class RewardFunction:
    """
    奖励函数管理器

    支持多种奖励函数类型：
    - exact_match: 精确匹配
    - edit_distance: 编辑距离
    - bleu: BLEU 分数
    - ensemble: 集成多种奖励函数
    """

    def __init__(
        self,
        reward_type: str = "exact_match",
        weights: Optional[Dict[str, float]] = None,
        embedding_model: Optional[object] = None
    ):
        """
        初始化奖励函数

        Args:
            reward_type: 奖励函数类型
            weights: 集成模式下的各奖励函数权重
            embedding_model: 可选的 embedding 模型
        """
        self.reward_type = reward_type
        self.weights = weights or {}
        self.embedding_model = embedding_model

    def compute(
        self,
        generated: str,
        reference: str,
        **kwargs
    ) -> float:
        """
        计算奖励分数

        Args:
            generated: 生成的文本
            reference: 参考文本
            **kwargs: 其他参数

        Returns:
            奖励分数
        """
        if self.reward_type == "exact_match":
            return compute_exact_match_reward(generated, reference)

        elif self.reward_type == "edit_distance":
            return compute_edit_distance_reward(generated, reference)

        elif self.reward_type == "bleu":
            return compute_bleu_reward(generated, reference)

        elif self.reward_type == "rouge_l":
            return compute_rouge_l_reward(generated, reference)

        elif self.reward_type == "ensemble":
            return self._compute_ensemble_reward(generated, reference)

        else:
            return 0.0

    def _compute_ensemble_reward(self, generated: str, reference: str) -> float:
        """计算集成奖励"""
        total_weight = sum(self.weights.values())
        if total_weight == 0:
            return 0.0

        rewards = {
            "exact_match": compute_exact_match_reward(generated, reference),
            "edit_distance": compute_edit_distance_reward(generated, reference),
            "bleu": compute_bleu_reward(generated, reference),
        }

        weighted_sum = sum(
            self.weights.get(key, 0) * reward
            for key, reward in rewards.items()
        )

        return weighted_sum / total_weight


class GSPOGroupManager:
    """
    GSPO 组管理器

    负责管理序列级相对优势计算所需的组结构：
    - 对同一 prompt 生成的多个样本组成组
    - 计算组内相对优势
    - 支持变长组和填充
    """

    def __init__(self, group_size: int = 4):
        """
        初始化组管理器

        Args:
            group_size: 每组的样本数
        """
        self.group_size = group_size

    def create_groups(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        创建组结构并计算相对优势

        Args:
            prompts: prompt 列表
            responses: 响应列表
            rewards: 奖励列表

        Returns:
            包含组信息和优势的字典
        """
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        # 计算序列级相对优势
        advantages = compute_sequence_level_advantage(rewards_tensor, self.group_size)

        # 计算组内排名
        rankings = compute_group_relative_ranking(rewards_tensor, self.group_size)

        return {
            "rewards": rewards_tensor,
            "advantages": advantages,
            "rankings": rankings,
            "group_size": self.group_size
        }

    def validate_groups(self, batch_size: int) -> bool:
        """
        验证批次大小是否能被 group_size 整除

        Args:
            batch_size: 批次大小

        Returns:
            是否有效
        """
        return batch_size % self.group_size == 0


def create_reward_function(
    reward_type: str = "exact_match",
    **kwargs
) -> RewardFunction:
    """
    创建奖励函数的工厂函数

    Args:
        reward_type: 奖励函数类型
        **kwargs: 其他参数

    Returns:
        奖励函数实例
    """
    return RewardFunction(reward_type=reward_type, **kwargs)
