from enum import Enum


class EvaluationTaskType(str, Enum):
    MEDQA = "medqa"
    BLEU_ROUGE = "bleu_rouge"