"""评估任务类型枚举，与 EvaluatorFactory 中的具体评估器类一一对应。"""

from enum import Enum


class EvaluationTaskType(str, Enum):
    """支持的评估任务类型。"""

    MEDQA_LLM = "medqa_llm"
    BLEU_ROUGE = "bleu_rouge"
    MATH = "math"
    CODE = "code"
    DPO_QUALITY = "dpo_quality"
