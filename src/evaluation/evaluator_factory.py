"""从 EvaluationTaskType 映射到具体 Evaluator 类。"""

from __future__ import annotations

from typing import Any, Dict, Type

from omegaconf import DictConfig, OmegaConf

from evaluation.bleu_rouge_evaluator import BleuRougeEvaluator
from evaluation.medqa_evaluator import MedQAEvaluator
from evaluation.task_types import EvaluationTaskType


class EvaluatorFactory:
    """按任务类型创建评估器实例。"""

    _MAP: Dict[EvaluationTaskType, Type] = {
        EvaluationTaskType.MEDQA: MedQAEvaluator,
        EvaluationTaskType.BLEU_ROUGE: BleuRougeEvaluator,
    }

    @classmethod
    def create(cls, task_type: EvaluationTaskType, config: Any):
        if isinstance(task_type, str):
            task_type = EvaluationTaskType(task_type)
        if task_type not in cls._MAP:
            raise ValueError(f"不支持的评估任务类型: {task_type}")
        eval_cls = cls._MAP[task_type]
        return eval_cls(config)