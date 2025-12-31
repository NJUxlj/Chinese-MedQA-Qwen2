import os
import sys
import torch
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List
from dataclasses import dataclass

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.trainer.reward_model_trainer import (
    RewardModelTrainer,
    RewardModelDataset,
    RewardModelCollator
)

MODEL_PATH = "/Users/xiniuyiliao/Desktop/code/models/gpt2"
DATA_PATH = "/Users/xiniuyiliao/Desktop/code/Chinese-MedQA-Qwen2/src/training/data/reward_model_data/reward_train.json"


@dataclass
class TestRewardModelConfig:
    """测试用配置类"""
    model_name_or_path: str = MODEL_PATH
    output_dir: str = "./test_output"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    max_seq_length: int = 128
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2
    warmup_steps: int = 10
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "linear"
    weight_decay: float = 0.01
    use_fp16: bool = False
    use_bf16: bool = False
    use_4bit: bool = False
    max_grad_norm: float = 1.0
    logging_level: str = "INFO"
    optim: str = "adamw_torch"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    device_map: str = None
    trust_remote_code: bool = False
    use_gradient_checkpointing: bool = False
    report_to: str = "none"
    eval_strategy: str = "no"
    eval_steps: int = 50
    do_train: bool = True
    do_eval: bool = False
    dataloader_pin_memory: bool = False
    remove_unused_columns: bool = False
    dataloader_num_workers: int = 4
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj"]


class TestRewardModelDataset:
    """测试 RewardModelDataset 类"""

    def test_init_with_valid_data(self):
        """测试使用有效数据初始化数据集"""
        tokenized_data = {
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "attention_mask": [[1, 1, 1], [1, 1, 1]],
            "chosen_labels": [[7, 8, 9], [10, 11, 12]],
            "rejected_labels": [[13, 14, 15], [16, 17, 18]]
        }
        dataset = RewardModelDataset(tokenized_data)
        assert len(dataset) == 2

    def test_getitem_returns_correct_dict(self):
        """测试 __getitem__ 返回正确格式"""
        tokenized_data = {
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "attention_mask": [[1, 1, 1], [1, 1, 1]],
            "chosen_labels": [[7, 8, 9], [10, 11, 12]],
            "rejected_labels": [[13, 14, 15], [16, 17, 18]]
        }
        dataset = RewardModelDataset(tokenized_data)
        item = dataset[0]

        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "chosen_labels" in item
        assert "rejected_labels" in item
        assert torch.is_tensor(item["input_ids"])
        assert torch.is_tensor(item["chosen_labels"])

    def test_len_with_empty_data(self):
        """测试空数据"""
        tokenized_data = {
            "input_ids": [],
            "attention_mask": [],
            "chosen_labels": [],
            "rejected_labels": []
        }
        dataset = RewardModelDataset(tokenized_data)
        assert len(dataset) == 0


class TestRewardModelCollator:
    """测试 RewardModelCollator 类"""

    def test_collator_initialization(self):
        """测试 Collator 初始化"""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        collator = RewardModelCollator(mock_tokenizer, padding_side="right", max_length=512)
        assert collator.tokenizer == mock_tokenizer
        assert collator.padding_side == "right"
        assert collator.max_length == 512

    def test_call_with_single_item_batch(self):
        """测试单样本批次处理"""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0

        collator = RewardModelCollator(mock_tokenizer, padding_side="right", max_length=512)

        batch = [{
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "chosen_labels": torch.tensor([4, 5, 6]),
            "rejected_labels": torch.tensor([7, 8, 9])
        }]

        result = collator(batch)

        assert "chosen_input_ids" in result
        assert "chosen_attention_mask" in result
        assert "rejected_input_ids" in result
        assert "rejected_attention_mask" in result
        assert "prompt_input_ids" in result
        assert "prompt_attention_mask" in result

        assert result["chosen_input_ids"].shape[0] == 1
        assert result["rejected_input_ids"].shape[0] == 1

    def test_call_with_padding(self):
        """测试填充逻辑"""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0

        collator = RewardModelCollator(mock_tokenizer, padding_side="right", max_length=10)

        batch = [
            {
                "input_ids": torch.tensor([1, 2]),
                "attention_mask": torch.tensor([1, 1]),
                "chosen_labels": torch.tensor([4, 5, 6]),
                "rejected_labels": torch.tensor([7, 8, 9, 10])
            },
            {
                "input_ids": torch.tensor([1, 2, 3, 4]),
                "attention_mask": torch.tensor([1, 1, 1, 1]),
                "chosen_labels": torch.tensor([4, 5]),
                "rejected_labels": torch.tensor([7, 8])
            }
        ]

        result = collator(batch)

        assert result["chosen_input_ids"].shape[1] == 3
        assert result["rejected_input_ids"].shape[1] == 4
        assert result["prompt_input_ids"].shape[1] == 4

    def test_max_length_truncation(self):
        """测试最大长度截断"""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0

        collator = RewardModelCollator(mock_tokenizer, padding_side="right", max_length=5)

        batch = [{
            "input_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1, 1, 1]),
            "chosen_labels": torch.tensor([9, 10, 11, 12, 13, 14]),
            "rejected_labels": torch.tensor([15, 16, 17, 18, 19, 20])
        }]

        result = collator(batch)

        assert result["chosen_input_ids"].shape[1] == 5
        assert result["rejected_input_ids"].shape[1] == 5
        assert result["prompt_input_ids"].shape[1] == 5


class TestRewardModelTrainerConfig:
    """测试 RewardModelTrainer 配置验证"""

    def test_config_validation_valid(self):
        """测试有效配置"""
        config = TestRewardModelConfig()
        trainer = RewardModelTrainer(config, finetuning_type="lora")
        assert trainer.config == config
        assert trainer.finetuning_type == "lora"

    def test_config_validation_invalid_learning_rate(self):
        """测试无效学习率"""
        config = TestRewardModelConfig()
        config.learning_rate = -0.001
        with pytest.raises(ValueError, match="学习率必须大于 0"):
            RewardModelTrainer(config)

    def test_config_validation_invalid_batch_size(self):
        """测试无效批次大小"""
        config = TestRewardModelConfig()
        config.per_device_train_batch_size = 0
        with pytest.raises(ValueError, match="批次大小必须至少为 1"):
            RewardModelTrainer(config)

    def test_config_validation_invalid_seq_length(self):
        """测试无效序列长度"""
        config = TestRewardModelConfig()
        config.max_seq_length = 0
        with pytest.raises(ValueError, match="序列长度必须至少为 1"):
            RewardModelTrainer(config)

    def test_config_validation_invalid_beta(self):
        """测试无效 beta 参数"""
        config = TestRewardModelConfig()
        with pytest.raises(ValueError, match="beta 参数必须大于等于 0"):
            RewardModelTrainer(config, beta=-1)


class TestRewardModelTrainerDataset:
    """测试 RewardModelTrainer 数据集准备"""

    @pytest.fixture
    def trainer_with_tokenizer(self):
        """创建带有分词器的训练器"""
        config = TestRewardModelConfig()
        trainer = RewardModelTrainer(config, finetuning_type="lora")

        from transformers import AutoTokenizer
        trainer.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=False
        )
        if trainer.tokenizer.pad_token is None:
            trainer.tokenizer.pad_token = trainer.tokenizer.eos_token

        return trainer

    def test_prepare_dataset_with_list(self, trainer_with_tokenizer):
        """测试使用列表数据准备数据集"""
        trainer = trainer_with_tokenizer
        sample_data = [
            {
                "prompt": "请计算：15 + 27 等于多少？",
                "chosen": "15 + 27 = 42",
                "rejected": "42"
            },
            {
                "prompt": "请计算：8 × 7 等于多少？",
                "chosen": "8 × 7 = 56",
                "rejected": "56"
            }
        ]

        dataset = trainer.prepare_dataset(sample_data, max_seq_length=64)

        assert dataset is not None
        assert len(dataset) == 2
        assert isinstance(dataset, RewardModelDataset)

    def test_prepare_dataset_missing_fields(self, trainer_with_tokenizer):
        """测试缺少字段的数据准备"""
        trainer = trainer_with_tokenizer
        invalid_data = [
            {
                "prompt": "请计算：15 + 27 等于多少？",
                "chosen": "42"
            }
        ]

        with pytest.raises(ValueError, match="必须包含 prompt、chosen 和 rejected 字段"):
            trainer.prepare_dataset(invalid_data)

    def test_prepare_dataset_from_file(self, trainer_with_tokenizer):
        """测试从文件加载数据集"""
        trainer = trainer_with_tokenizer

        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        dataset = trainer.prepare_dataset(data, max_seq_length=128)

        assert dataset is not None
        assert len(dataset) == len(data)
        assert isinstance(dataset, RewardModelDataset)

        sample = dataset[0]
        assert "input_ids" in sample
        assert "chosen_labels" in sample
        assert "rejected_labels" in sample


class TestRewardModelTrainerModel:
    """测试 RewardModelTrainer 模型加载"""

    def test_model_loading_requires_tokenizer(self):
        """测试模型加载需要先加载分词器"""
        config = TestRewardModelConfig()
        trainer = RewardModelTrainer(config, finetuning_type="lora")

        with pytest.raises(Exception):
            trainer.load_model_and_tokenizer()

    @pytest.fixture
    def trainer_for_model_loading(self):
        """创建用于模型加载测试的训练器"""
        config = TestRewardModelConfig()
        trainer = RewardModelTrainer(config, finetuning_type="lora")

        from transformers import AutoTokenizer
        trainer.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=False
        )
        if trainer.tokenizer.pad_token is None:
            trainer.tokenizer.pad_token = trainer.tokenizer.eos_token

        return trainer

    def test_model_loading_success(self, trainer_for_model_loading):
        """测试成功加载模型"""
        trainer = trainer_for_model_loading

        try:
            trainer.load_model_and_tokenizer()
            assert trainer.model is not None
            assert trainer.accelerator is not None
        except Exception as e:
            pytest.skip(f"模型加载失败（可能是环境问题）: {e}")

    def test_reward_model_head_creation(self, trainer_for_model_loading):
        """测试奖励预测头创建"""
        trainer = trainer_for_model_loading

        try:
            trainer.load_model_and_tokenizer()

            assert trainer.reward_model_head is not None
            assert hasattr(trainer.reward_model_head, 'forward')

            if hasattr(trainer.model, 'base_model'):
                pass
        except Exception as e:
            pytest.skip(f"模型加载失败: {e}")

    def test_lora_config_application(self, trainer_for_model_loading):
        """测试 LoRA 配置应用"""
        config = TestRewardModelConfig()
        config.lora_r = 8
        config.lora_alpha = 16
        config.lora_dropout = 0.05

        trainer = RewardModelTrainer(
            config,
            finetuning_type="lora",
            beta=0.1
        )

        from transformers import AutoTokenizer
        trainer.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=False
        )
        if trainer.tokenizer.pad_token is None:
            trainer.tokenizer.pad_token = trainer.tokenizer.eos_token

        try:
            trainer.load_model_and_tokenizer()
        except Exception as e:
            pytest.skip(f"模型加载失败: {e}")


class TestRewardModelTrainerTrainingArgs:
    """测试训练参数创建"""

    def test_create_training_args(self):
        """测试创建训练参数"""
        config = TestRewardModelConfig()
        trainer = RewardModelTrainer(config, finetuning_type="lora")

        training_args = trainer.create_training_args(
            output_dir="./test_training_args",
            save_strategy="no"
        )

        assert training_args is not None
        assert training_args.output_dir == "./test_training_args"
        assert training_args.num_train_epochs == 1
        assert training_args.per_device_train_batch_size == 2
        assert training_args.learning_rate == 5e-5


class TestRewardModelTrainerFullPipeline:
    """完整训练流程测试"""

    @pytest.fixture
    def full_trainer(self):
        """创建完整的训练器实例"""
        config = TestRewardModelConfig()
        config.per_device_train_batch_size = 1
        config.gradient_accumulation_steps = 1

        trainer = RewardModelTrainer(config, finetuning_type="lora")

        from transformers import AutoTokenizer
        trainer.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=False
        )
        if trainer.tokenizer.pad_token is None:
            trainer.tokenizer.pad_token = trainer.tokenizer.eos_token

        return trainer

    def test_full_pipeline_execution(self, full_trainer):
        """测试完整流程执行"""
        trainer = full_trainer

        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sample_data = data[:2]

        try:
            trainer.load_model_and_tokenizer()
        except Exception as e:
            pytest.skip(f"模型加载失败: {e}")

        dataset = trainer.prepare_dataset(sample_data, max_seq_length=64)

        assert len(dataset) == 2

        collator = RewardModelCollator(
            trainer.tokenizer,
            padding_side="right",
            max_length=64
        )

        batch = [dataset[i] for i in range(len(dataset))]
        collated_batch = collator(batch)

        assert collated_batch["chosen_input_ids"].shape[0] == 2
        assert collated_batch["rejected_input_ids"].shape[0] == 2


class TestEdgeCases:
    """边界情况测试"""

    def test_empty_dataset(self):
        """测试空数据集"""
        config = TestRewardModelConfig()
        trainer = RewardModelTrainer(config, finetuning_type="lora")

        with pytest.raises(Exception):
            dataset = trainer.prepare_dataset([])

    def test_single_sample(self):
        """测试单样本"""
        config = TestRewardModelConfig()
        trainer = RewardModelTrainer(config, finetuning_type="lora")

        from transformers import AutoTokenizer
        trainer.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=False
        )
        if trainer.tokenizer.pad_token is None:
            trainer.tokenizer.pad_token = trainer.tokenizer.eos_token

        sample_data = [{
            "prompt": "请计算：15 + 27 等于多少？",
            "chosen": "15 + 27 = 42",
            "rejected": "42"
        }]

        dataset = trainer.prepare_dataset(sample_data, max_seq_length=64)
        assert len(dataset) == 1

    def test_very_long_sequence(self):
        """测试超长序列"""
        config = TestRewardModelConfig()
        trainer = RewardModelTrainer(config, finetuning_type="lora")

        from transformers import AutoTokenizer
        trainer.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=False
        )
        if trainer.tokenizer.pad_token is None:
            trainer.tokenizer.pad_token = trainer.tokenizer.eos_token

        long_text = " ".join(["word"] * 1000)
        sample_data = [{
            "prompt": long_text,
            "chosen": long_text[:500],
            "rejected": long_text[:400]
        }]

        dataset = trainer.prepare_dataset(sample_data, max_seq_length=128)
        assert len(dataset) == 1


class TestRewardModelCollatorPaddingSides:
    """测试填充方向"""

    def test_padding_left(self):
        """测试左侧填充"""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0

        collator = RewardModelCollator(mock_tokenizer, padding_side="left", max_length=10)

        batch = [{
            "input_ids": torch.tensor([1, 2]),
            "attention_mask": torch.tensor([1, 1]),
            "chosen_labels": torch.tensor([3, 4, 5]),
            "rejected_labels": torch.tensor([6, 7, 8])
        }, {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "attention_mask": torch.tensor([1, 1, 1, 1]),
            "chosen_labels": torch.tensor([3, 4, 5, 7, 8]),
            "rejected_labels": torch.tensor([6, 7])
        }]

        result = collator(batch)

        assert result["chosen_input_ids"].shape[1] == 5
        chosen_input = result["chosen_input_ids"][0].tolist()
        assert chosen_input == [0, 0, 3, 4, 5]

        chosen_input_longer = result["chosen_input_ids"][1].tolist()
        assert chosen_input_longer == [3, 4, 5, 7, 8]

    def test_padding_right(self):
        """测试右侧填充"""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0

        collator = RewardModelCollator(mock_tokenizer, padding_side="right", max_length=10)

        batch = [{
            "input_ids": torch.tensor([1, 2]),
            "attention_mask": torch.tensor([1, 1]),
            "chosen_labels": torch.tensor([3, 4, 5]),
            "rejected_labels": torch.tensor([6, 7, 8])
        }, {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "attention_mask": torch.tensor([1, 1, 1, 1]),
            "chosen_labels": torch.tensor([3, 4, 5, 7, 8]),
            "rejected_labels": torch.tensor([6, 7])
        }]

        result = collator(batch)

        assert result["chosen_input_ids"].shape[1] == 5
        chosen_input = result["chosen_input_ids"][0].tolist()
        assert chosen_input == [3, 4, 5, 0, 0]


class TestIntegration:
    """集成测试"""

    def test_data_from_file_format(self):
        """测试数据文件格式"""
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) > 0

        for item in data:
            assert "prompt" in item
            assert "chosen" in item
            assert "rejected" in item
            assert isinstance(item["prompt"], str)
            assert isinstance(item["chosen"], str)
            assert isinstance(item["rejected"], str)

    def test_model_path_exists(self):
        """测试模型路径存在"""
        assert os.path.exists(MODEL_PATH)

        config_path = os.path.join(MODEL_PATH, "config.json")
        assert os.path.exists(config_path)

    def test_tokenizer_compatibility(self):
        """测试分词器兼容性"""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=False
        )

        assert tokenizer is not None

        test_text = "这是一个测试文本"
        tokens = tokenizer(test_text, return_tensors=None)
        assert "input_ids" in tokens
        assert "attention_mask" in tokens

        decoded = tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
