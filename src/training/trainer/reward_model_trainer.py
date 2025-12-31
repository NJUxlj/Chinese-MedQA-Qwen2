import os
import sys
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datasets import Dataset
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerControl
)
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from accelerate import Accelerator, DistributedDataParallelKwargs

try:
    from deepspeed import DeepSpeedEngine
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    DeepSpeedEngine = None

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.logger import setup_logger
from training.trainer.base_trainer import BaseTrainer


@dataclass
class RewardModelTrainingConfig:
    """奖励模型训练配置"""
    model_name_or_path: str = "Qwen/Qwen3-14B"
    output_dir: str = "output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    max_seq_length: int = 2048
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 2
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    report_to: str = "none"
    eval_strategy: str = "steps"
    eval_steps: Optional[int] = 50
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    max_grad_norm: float = 1.0
    dataloader_pin_memory: bool = False
    remove_unused_columns: bool = False
    do_train: bool = True
    do_eval: bool = False
    dataloader_num_workers: int = 4
    use_gradient_checkpointing: bool = True
    device_map: Optional[Dict[str, Any]] = None
    use_4bit: bool = False
    use_8bit: bool = False
    use_fp16: bool = True
    use_bf16: bool = False
    trust_remote_code: bool = True
    logging_level: str = "INFO"
    optim: str = "adamw_torch"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])


class RewardModelDataset(Dataset):
    """
    奖励模型数据集封装
    
    奖励模型数据与标准 SFT 数据不同，每条样本包含：
    - input_ids: prompt 的分词结果
    - attention_mask: 注意力掩码
    - chosen_labels: 偏好答案的分词结果
    - rejected_labels: 不偏好答案的分词结果
    
    奖励模型采用成对学习（pairwise learning）策略：
    - 对于同一个 prompt，学习区分 chosen（优质）和 rejected（劣质）回答
    - 通过比较两个回答的得分差异来优化模型
    - 最终模型能够预测人类对不同回答的偏好程度
    """
    
    def __init__(self, tokenized_data: Dict[str, List]):
        self._data = tokenized_data
    
    def __len__(self) -> int:
        return len(self._data["input_ids"])
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self._data["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self._data["attention_mask"][idx], dtype=torch.long),
            "chosen_labels": torch.tensor(self._data["chosen_labels"][idx], dtype=torch.long),
            "rejected_labels": torch.tensor(self._data["rejected_labels"][idx], dtype=torch.long)
        }


class RewardModelCollator:
    """
    奖励模型数据整理器
    
    负责将批次数据整理为模型输入格式，
    支持动态填充和注意力掩码处理。
    
    数据处理流程：
    1. 分别对 chosen 和 rejected 的 input_ids 进行填充
    2. 创建注意力掩码，标记有效 token 和 padding
    3. 组合批次数据，准备模型输入
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        padding_side: str = "right",
        max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.padding_side = padding_side
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        
        chosen_input_ids = [item["chosen_labels"] for item in batch]
        rejected_input_ids = [item["rejected_labels"] for item in batch]
        prompt_input_ids = [item["input_ids"] for item in batch]
        prompt_attention_mask = [item["attention_mask"] for item in batch]
        
        max_chosen_len = min(max(len(ids) for ids in chosen_input_ids), self.max_length)
        max_rejected_len = min(max(len(ids) for ids in rejected_input_ids), self.max_length)
        max_prompt_len = min(max(len(ids) for ids in prompt_input_ids), self.max_length)
        
        chosen_padded = []
        chosen_attention = []
        for ids in chosen_input_ids:
            ids_list = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
            ids_len = len(ids_list)
            if ids_len > max_chosen_len:
                chosen_padded.append(ids_list[:max_chosen_len])
                chosen_attention.append([1] * max_chosen_len)
            else:
                padding = [self.tokenizer.pad_token_id] * (max_chosen_len - ids_len)
                if self.padding_side == "right":
                    chosen_padded.append(ids_list + padding)
                else:
                    chosen_padded.append(padding + ids_list)
                chosen_attention.append([1] * ids_len + [0] * (max_chosen_len - ids_len))
        
        rejected_padded = []
        rejected_attention = []
        for ids in rejected_input_ids:
            ids_list = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
            ids_len = len(ids_list)
            if ids_len > max_rejected_len:
                rejected_padded.append(ids_list[:max_rejected_len])
                rejected_attention.append([1] * max_rejected_len)
            else:
                padding = [self.tokenizer.pad_token_id] * (max_rejected_len - ids_len)
                if self.padding_side == "right":
                    rejected_padded.append(ids_list + padding)
                else:
                    rejected_padded.append(padding + ids_list)
                rejected_attention.append([1] * ids_len + [0] * (max_rejected_len - ids_len))
        
        prompt_padded = []
        prompt_att = []
        for ids, mask in zip(prompt_input_ids, prompt_attention_mask):
            ids_list = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
            ids_len = len(ids_list)
            mask_list = mask.tolist() if isinstance(mask, torch.Tensor) else list(mask)
            if ids_len > max_prompt_len:
                prompt_padded.append(ids_list[:max_prompt_len])
                prompt_att.append(mask_list[:max_prompt_len])
            else:
                padding = [self.tokenizer.pad_token_id] * (max_prompt_len - ids_len)
                if self.padding_side == "right":
                    prompt_padded.append(ids_list + padding)
                    prompt_att.append(mask_list + [0] * (max_prompt_len - ids_len))
                else:
                    prompt_padded.append(padding + ids_list)
                    prompt_att.append([0] * (max_prompt_len - ids_len) + mask_list)
        
        return {
            "chosen_input_ids": torch.tensor(chosen_padded, dtype=torch.long),
            "chosen_attention_mask": torch.tensor(chosen_attention, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_padded, dtype=torch.long),
            "rejected_attention_mask": torch.tensor(rejected_attention, dtype=torch.long),
            "prompt_input_ids": torch.tensor(prompt_padded, dtype=torch.long),
            "prompt_attention_mask": torch.tensor(prompt_att, dtype=torch.long)
        }


class RewardModelTrainer(BaseTrainer):
    """
    奖励模型训练器
    
    继承自 BaseTrainer，利用基础框架进行模型加载和训练管理，
    专注于奖励模型特有的偏好学习逻辑。
    
    核心功能：
    1. 加载基础模型和分词器（继承自 BaseTrainer）
    2. 准备偏好数据集（prompt, chosen, rejected）
    3. 实现成对排序损失函数（pairwise ranking loss）
    4. 支持多 GPU 分布式训练（Accelerate）
    5. 支持 DeepSpeed 加速
    6. 支持 LoRA 微调
    
    奖励模型训练原理：
    - 输入：prompt + chosen/rejected 回答对
    - 输出：两个回答的得分差异
    - 目标：chosen 得分应高于 rejected 得分
    - 损失：pairwise margin ranking loss
    
    数据流：
    preference_data.json → RewardModelDataset → RewardModelCollator → Model → Loss
    """
    
    def __init__(
        self,
        config: RewardModelTrainingConfig,
        finetuning_type: str = "lora",
        beta: float = 0.1,
        use_deepspeed: bool = False,
        deepspeed_config: Optional[Union[str, Dict]] = None,
        reward_model_head: Optional[torch.nn.Module] = None
    ):
        """
        初始化奖励模型训练器
        
        Args:
            config: 奖励模型训练配置
            finetuning_type: 微调类型，可选 "lora"、"full"、"qlora"
            beta: KL 散度惩罚系数，用于对齐奖励尺度
            use_deepspeed: 是否使用 DeepSpeed 加速
            deepspeed_config: DeepSpeed 配置文件路径或字典
            reward_model_head: 自定义奖励预测头，若为 None 则使用默认头
        """
        super().__init__(config)
        
        self.config = config
        self.finetuning_type = finetuning_type
        self.beta = beta
        self.use_deepspeed = use_deepspeed
        self.deepspeed_config = deepspeed_config
        self.reward_model_head = reward_model_head
        
        self.model = None
        self.tokenizer = None
        self.accelerator = None
        self.train_dataset = None
        self.eval_dataset = None
        
        self.logger = setup_logger(self.__class__.__name__, level=config.logging_level or "INFO")
        
        self._validate_config()
    
    def _validate_config(self):
        """
        验证配置参数的有效性
        
        检查配置项是否满足奖励模型训练的基本要求，
        包括学习率、批次大小、序列长度等。
        """
        if self.config.learning_rate <= 0:
            raise ValueError(f"学习率必须大于 0，当前值：{self.config.learning_rate}")
        
        if self.config.per_device_train_batch_size < 1:
            raise ValueError(f"批次大小必须至少为 1，当前值：{self.config.per_device_train_batch_size}")
        
        if self.config.max_seq_length < 1:
            raise ValueError(f"序列长度必须至少为 1，当前值：{self.config.max_seq_length}")
        
        if self.beta < 0:
            raise ValueError(f"beta 参数必须大于等于 0，当前值：{self.beta}")
        
        self.logger.info("配置验证通过")
    
    def load_model_and_tokenizer(self):
        """
        加载奖励模型和分词器
        
        重写父类方法，添加奖励模型特有的加载逻辑：
        1. 加载基础语言模型
        2. 添加或初始化奖励预测头
        3. 配置 LoRA（如果启用）
        4. 准备多 GPU 环境
        
        模型结构说明：
        - 基础模型：用于编码输入文本
        - 奖励头：线性层，将隐藏状态映射为标量分数
        - 训练时冻结基础模型参数，只更新奖励头
        """
        try:
            self.logger.info(f"正在加载奖励模型：{self.config.model_name_or_path}")
            
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                log_with=self.config.report_to,
                split_batches=False,
                ddp_find_unused_parameters=False
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=self.config.trust_remote_code,
                padding_side="right"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            model_kwargs = {
                "trust_remote_code": self.config.trust_remote_code,
                "torch_dtype": torch.float16 if self.config.use_fp16 and torch.cuda.is_available() else torch.float32,
                "device_map": self.config.device_map if torch.cuda.is_available() else None,
            }
            
            if self.config.use_4bit:
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    self.logger.info("启用 4-bit 量化")
                except ImportError:
                    self.logger.warning("未安装 bitsandbytes，禁用 4-bit 量化")
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                **model_kwargs
            )
            
            if not torch.cuda.is_available():
                base_model = base_model.to("cpu")
            
            hidden_size = base_model.config.hidden_size
            num_layers = base_model.config.num_hidden_layers
            
            if self.reward_model_head is None:
                self.reward_model_head = torch.nn.Sequential(
                    torch.nn.Linear(hidden_size, hidden_size // 2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_size // 2, 1)
                )
                self.logger.info(f"创建默认奖励预测头：{hidden_size} -> {hidden_size // 2} -> 1")
            else:
                self.logger.info("使用自定义奖励预测头")
            
            self.model = RewardModel(base_model, self.reward_model_head)
            
            if self.finetuning_type == "lora":
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=self.config.lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
                )
                self.model.base_model = get_peft_model(self.model.base_model, lora_config)
                self.model.base_model.print_trainable_parameters()
                self.logger.info(f"启用 LoRA 微调，r={self.config.lora_r}, alpha={self.config.lora_alpha}")
            
            if self.use_gradient_checkpointing:
                if hasattr(self.model.base_model, "gradient_checkpointing_enable"):
                    self.model.base_model.gradient_checkpointing_enable()
                else:
                    self.model.gradient_checkpointing_enable()
                self.logger.info("启用梯度检查点")
            
            if self.use_deepspeed:
                self._setup_deepspeed()
            
            self.model = self.accelerator.prepare(self.model)
            
            self.logger.info("奖励模型和分词器加载完成")
            
        except Exception as e:
            self.logger.error(f"模型加载失败：{e}")
            raise
    
    def _setup_deepspeed(self):
        """
        配置 DeepSpeed 加速
        
        DeepSpeed 提供以下优化：
        1. ZeRO 优化器：分片优化器状态、梯度和模型参数
        2. 混合精度训练：FP16/BF16 支持
        3. 梯度累积：支持更大的有效批次大小
        4. 激活检查点：减少显存占用
        """
        if self.deepspeed_config is None:
            ds_config = {
                "train_batch_size": self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps,
                "train_micro_batch_size_per_gpu": self.config.per_device_train_batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": self.config.learning_rate,
                        "weight_decay": self.config.weight_decay
                    }
                },
                "fp16": {
                    "enabled": self.config.use_fp16 and torch.cuda.is_available(),
                    "loss_scale": 0,
                    "initial_scale_power": 16
                },
                "bf16": {
                    "enabled": self.config.use_bf16 and torch.cuda.is_bf16_supported()
                },
                "zero_optimization": {
                    "stage": 2,
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": True
                    },
                    "allgather_partitions": True,
                    "allgather_bucket_size": 2e8,
                    "reduce_bucket_size": 2e8,
                    "overlap_comm": True,
                    "contiguous_gradients": True
                },
                "gradient_checkpointing": self.use_gradient_checkpointing,
                "logging": {"steps_per_print": 100}
            }
        else:
            if isinstance(self.deepspeed_config, str):
                with open(self.deepspeed_config, 'r') as f:
                    ds_config = json.load(f)
            else:
                ds_config = self.deepspeed_config
        
        self.logger.info(f"DeepSpeed 配置：stage=2, fp16={ds_config.get('fp16', {}).get('enabled', False)}")
    
    def prepare_dataset(
        self,
        dataset: Union[Dataset, List[Dict]],
        prompt_field: str = "prompt",
        chosen_field: str = "chosen",
        rejected_field: str = "rejected",
        max_seq_length: Optional[int] = None
    ) -> Dataset:
        """
        准备奖励模型训练数据集
        
        将原始偏好数据转换为模型可处理的格式：
        1. 分别对 prompt、chosen、rejected 进行分词
        2. 创建注意力掩码
        3. 验证数据质量
        
        Args:
            dataset: 输入数据集（支持 Dataset 对象或字典列表）
            prompt_field: prompt 字段名
            chosen_field: chosen 回答字段名
            rejected_field: rejected 回答字段名
            max_seq_length: 最大序列长度
            
        Returns:
            处理后的 RewardModelDataset
        """
        max_seq_length = max_seq_length or self.config.max_seq_length
        
        if isinstance(dataset, list):
            dataset = Dataset.from_list(dataset)
        
        self.logger.info(f"准备数据集，包含 {len(dataset)} 个样本")
        
        def tokenize_function(examples):
            prompts = examples.get(prompt_field, [])
            chosen_texts = examples.get(chosen_field, [])
            rejected_texts = examples.get(rejected_field, [])
            
            if not prompts or not chosen_texts or not rejected_texts:
                raise ValueError("数据集必须包含 prompt、chosen 和 rejected 字段")
            
            prompt_encodings = self.tokenizer(
                prompts,
                truncation=True,
                padding="max_length",
                max_length=max_seq_length,
                return_tensors=None
            )
            
            chosen_encodings = self.tokenizer(
                chosen_texts,
                truncation=True,
                padding="max_length",
                max_length=max_seq_length,
                return_tensors=None
            )
            
            rejected_encodings = self.tokenizer(
                rejected_texts,
                truncation=True,
                padding="max_length",
                max_length=max_seq_length,
                return_tensors=None
            )
            
            return {
                "input_ids": prompt_encodings["input_ids"],
                "attention_mask": prompt_encodings["attention_mask"],
                "chosen_labels": chosen_encodings["input_ids"],
                "rejected_labels": rejected_encodings["input_ids"]
            }
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="分词奖励模型数据"
        )
        
        tokenized_data = {
            "input_ids": tokenized_dataset["input_ids"],
            "attention_mask": tokenized_dataset["attention_mask"],
            "chosen_labels": tokenized_dataset["chosen_labels"],
            "rejected_labels": tokenized_dataset["rejected_labels"]
        }
        
        self.train_dataset = RewardModelDataset(tokenized_data)
        
        self.logger.info(f"数据集准备完成，包含 {len(self.train_dataset)} 个样本")
        return self.train_dataset
    
    def create_training_args(
        self,
        output_dir: str,
        **training_kwargs
    ) -> TrainingArguments:
        """
        创建奖励模型训练参数
        
        重写父类方法，针对奖励模型优化默认参数。
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            report_to=self.config.report_to,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model or "accuracy",
            greater_is_better=self.config.greater_is_better,
            max_grad_norm=self.config.max_grad_norm,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            remove_unused_columns=self.config.remove_unused_columns,
            do_train=self.config.do_train,
            do_eval=self.config.do_eval,
            fp16=self.config.use_fp16 and torch.cuda.is_available(),
            bf16=self.config.use_bf16 and torch.cuda.is_bf16_supported(),
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_prefetch_factor=self.config.dataloader_prefetch_factor if hasattr(self.config, 'dataloader_prefetch_factor') else None,
            gradient_checkpointing=self.use_gradient_checkpointing,
            save_safetensors=True,
            **training_kwargs
        )
        
        self.logger.info(f"训练参数创建完成：{self.config.num_train_epochs} epochs, batch_size={self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps}")
        return training_args
    
    def compute_loss(
        self,
        model: "RewardModel",
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        计算奖励模型损失
        
        使用成对排序损失函数（Pairwise Ranking Loss）：
        L = -log(sigmoid(r_chosen - r_rejected))
        
        其中：
        - r_chosen：模型对 chosen 回答的评分
        - r_rejected：模型对 rejected 回答的评分
        - sigmoid：将评分差转换为概率
        
        损失函数设计原理：
        - 当 chosen 评分高于 rejected 评分时，损失接近 0
        - 当 chosen 评分低于 rejected 评分时，损失较大
        - beta 参数用于缩放评分差异，控制对错误的惩罚程度
        
        Args:
            model: 奖励模型
            inputs: 输入数据字典
            return_outputs: 是否返回模型输出
            
        Returns:
            损失值，或（损失值，输出字典）
        """
        chosen_input_ids = inputs["chosen_input_ids"]
        chosen_attention_mask = inputs["chosen_attention_mask"]
        rejected_input_ids = inputs["rejected_input_ids"]
        rejected_attention_mask = inputs["rejected_attention_mask"]
        prompt_input_ids = inputs["prompt_input_ids"]
        prompt_attention_mask = inputs["prompt_attention_mask"]
        
        batch_size = chosen_input_ids.size(0)
        
        chosen_input = torch.cat([prompt_input_ids, chosen_input_ids], dim=1)
        chosen_mask = torch.cat([prompt_attention_mask, chosen_attention_mask], dim=1)
        
        rejected_input = torch.cat([prompt_input_ids, rejected_input_ids], dim=1)
        rejected_mask = torch.cat([prompt_attention_mask, rejected_attention_mask], dim=1)
        
        chosen_scores = model(
            chosen_input,
            attention_mask=chosen_mask
        ).squeeze(-1)
        
        rejected_scores = model(
            rejected_input,
            attention_mask=rejected_mask
        ).squeeze(-1)
        
        chosen_scores = chosen_scores[:, -1] if chosen_scores.size(1) > prompt_input_ids.size(1) else chosen_scores.mean(dim=1)
        rejected_scores = rejected_scores[:, -1] if rejected_scores.size(1) > prompt_input_ids.size(1) else rejected_scores.mean(dim=1)
        
        margin = self.beta * (chosen_scores - rejected_scores)
        
        losses = F.binary_cross_entropy_with_logits(margin, torch.ones_like(margin))
        
        chosen_accuracy = (chosen_scores > rejected_scores).float().mean()
        
        output_dict = {
            "loss": losses,
            "chosen_scores": chosen_scores.detach(),
            "rejected_scores": rejected_scores.detach(),
            "accuracy": chosen_accuracy
        }
        
        if return_outputs:
            return losses, output_dict
        return losses
    
    def training_step(
        self,
        model: "RewardModel",
        inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        执行单步训练
        
        重写以使用自定义损失计算。
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        with self.accelerator.accumulate(model):
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            
            self.accelerator.backward(loss)
            
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    model.parameters(),
                    self.config.max_grad_norm
                )
            
            return loss.detach()
    
    def prediction_step(
        self,
        model: "RewardModel",
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool = False,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        执行预测步骤
        
        用于评估阶段的损失计算和指标评估。
        """
        model.eval()
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        
        if prediction_loss_only:
            return loss, None, None
        
        return loss, None, outputs
    
    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备模型输入
        
        将输入数据移动到正确的设备上。
        """
        if isinstance(inputs, dict):
            return {k: v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        return inputs
    
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        """
        评估奖励模型
        
        计算以下指标：
        - accuracy：chosen 评分高于 rejected 的比例
        - avg_margin：评分差的平均值
        - loss：验证集损失
        """
        if eval_dataset is not None:
            self.eval_dataset = eval_dataset
        
        if self.eval_dataset is None:
            raise ValueError("评估数据集未设置")
        
        data_collator = RewardModelCollator(
            tokenizer=self.tokenizer,
            padding_side="right",
            max_length=self.config.max_seq_length
        )
        
        eval_dataloader = torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=self.config.per_device_eval_batch_size,
            collate_fn=data_collator,
            num_workers=0,
            pin_memory=True
        )
        
        eval_dataloader = self.accelerator.prepare(eval_dataloader)
        
        model = self.accelerator.unwrap_model(self.model)
        model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for batch in eval_dataloader:
            with torch.no_grad():
                loss, outputs = self.compute_loss(model, batch, return_outputs=True)
            
            total_loss += loss.item()
            total_accuracy += outputs["accuracy"].item()
            num_batches += 1
        
        metrics = {
            f"{metric_key_prefix}_loss": total_loss / max(num_batches, 1),
            f"{metric_key_prefix}_accuracy": total_accuracy / max(num_batches, 1)
        }
        
        self.logger.info(f"评估结果：loss={metrics[f'{metric_key_prefix}_loss']:.4f}, accuracy={metrics[f'{metric_key_prefix}_accuracy']:.4f}")
        
        return metrics
    
    def create_optimizer(self):
        """
        创建优化器
        
        根据微调类型选择不同的参数组：
        - LoRA：只优化 LoRA 参数
        - Full：优化所有参数
        """
        if self.finetuning_type == "lora":
            from peft import get_peft_model_state_dict
            
            def get_lora_parameters(model):
                lora_params = []
                for name, param in model.named_parameters():
                    if "lora_" in name or "lora_A" in name or "lora_B" in name:
                        lora_params.append(param)
                return lora_params
            
            optimizer_grouped_parameters = [
                {
                    "params": get_lora_parameters(self.model),
                    "lr": self.config.learning_rate
                },
                {
                    "params": self.reward_model_head.parameters(),
                    "lr": self.config.learning_rate
                }
            ]
            
            self.logger.info(f"LoRA 参数量：{sum(p.numel() for p in get_lora_parameters(self.model))}")
        else:
            optimizer_grouped_parameters = [
                {"params": self.model.parameters(), "lr": self.config.learning_rate}
            ]
        
        optimizer_cls = torch.optim.AdamW
        optimizer_kwargs = {
            "lr": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "betas": (self.config.adam_beta1, self.config.adam_beta2),
            "eps": self.config.adam_epsilon
        }
        
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        
        self.logger.info(f"优化器创建完成：{optimizer_cls.__name__}, lr={self.config.learning_rate}")
        return self.optimizer
    
    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        """
        创建学习率调度器
        
        支持多种调度策略：
        - linear：线性衰减
        - cosine：余弦退火
        - constant：保持不变
        - warmup：预热后保持
        """
        if optimizer is None:
            optimizer = self.optimizer
        
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        if self.config.lr_scheduler_type == "linear":
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif self.config.lr_scheduler_type == "cosine":
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif self.config.lr_scheduler_type == "constant":
            from transformers import get_constant_schedule
            scheduler = get_constant_schedule(optimizer)
        else:
            from transformers import get_polynomial_decay_schedule_with_warmup
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                power=1.0
            )
        
        self.lr_scheduler = scheduler
        
        self.logger.info(f"学习率调度器创建完成：{self.config.lr_scheduler_type}, warmup_ratio={self.config.warmup_ratio}")
        return scheduler
    
    def train(
        self,
        train_dataset: Union[Dataset, List[Dict]],
        output_dir: str,
        eval_dataset: Optional[Union[Dataset, List[Dict]]] = None,
        prompt_field: str = "prompt",
        chosen_field: str = "chosen",
        rejected_field: str = "rejected",
        **training_kwargs
    ):
        """
        开始奖励模型训练
        
        完整的训练流程：
        1. 加载模型和分词器
        2. 准备训练和评估数据集
        3. 创建优化器和学习率调度器
        4. 执行多轮训练
        5. 保存模型和配置
        
        Args:
            train_dataset: 训练数据集
            output_dir: 输出目录
            eval_dataset: 评估数据集
            prompt_field: prompt 字段名
            chosen_field: chosen 回答字段名
            rejected_field: rejected 回答字段名
            **training_kwargs: 其他训练参数
        """
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
        
        self.logger.info("=" * 60)
        self.logger.info("开始奖励模型训练")
        self.logger.info("=" * 60)
        
        prepared_train = self.prepare_dataset(
            train_dataset,
            prompt_field=prompt_field,
            chosen_field=chosen_field,
            rejected_field=rejected_field
        )
        
        if eval_dataset is not None:
            prepared_eval = self.prepare_dataset(
                eval_dataset,
                prompt_field=prompt_field,
                chosen_field=chosen_field,
                rejected_field=rejected_field
            )
            self.eval_dataset = prepared_eval
        
        training_args = self.create_training_args(output_dir, **training_kwargs)
        
        data_collator = RewardModelCollator(
            tokenizer=self.tokenizer,
            padding_side="right",
            max_length=self.config.max_seq_length
        )
        
        train_dataloader = torch.utils.data.DataLoader(
            prepared_train,
            batch_size=self.config.per_device_train_batch_size,
            collate_fn=data_collator,
            num_workers=0,
            pin_memory=True,
            shuffle=True
        )
        
        train_dataloader = self.accelerator.prepare(train_dataloader)
        
        num_update_steps_per_epoch = len(train_dataloader) // self.config.gradient_accumulation_steps
        num_training_steps = num_update_steps_per_epoch * self.config.num_train_epochs
        
        optimizer = self.create_optimizer()
        scheduler = self.create_scheduler(num_training_steps, optimizer)
        
        num_epochs = self.config.num_train_epochs
        
        self.model, optimizer, train_dataloader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, scheduler
        )
        
        total_batch_size = self.config.per_device_train_batch_size * self.accelerator.num_processes * self.config.gradient_accumulation_steps
        
        self.logger.info("=" * 60)
        self.logger.info("训练配置信息")
        self.logger.info("=" * 60)
        self.logger.info(f"  模型：{self.config.model_name_or_path}")
        self.logger.info(f"  微调类型：{self.finetuning_type}")
        self.logger.info(f"  训练样本数：{len(prepared_train)}")
        self.logger.info(f"  批次大小（总）：{total_batch_size}")
        self.logger.info(f"  训练轮数：{num_epochs}")
        self.logger.info(f"  学习率：{self.config.learning_rate}")
        self.logger.info(f"  KL 惩罚系数（beta）：{self.beta}")
        self.logger.info(f"  最大序列长度：{self.config.max_seq_length}")
        self.logger.info(f"  GPU 数量：{self.accelerator.num_processes}")
        self.logger.info("=" * 60)
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0
            
            progress_bar = None
            if self.accelerator.is_local_main_process:
                from tqdm.auto import tqdm
                progress_bar = tqdm(range(len(train_dataloader)), desc=f"Epoch {epoch + 1}")
            
            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.model):
                    loss, outputs = self.compute_loss(self.model, batch, return_outputs=True)
                    
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.detach().item()
                epoch_accuracy += outputs["accuracy"].item()
                num_batches += 1
                
                if progress_bar is not None:
                    progress_bar.update(1)
                    if step % self.config.logging_steps == 0:
                        progress_bar.set_postfix({
                            "loss": f"{loss.detach().item():.4f}",
                            "acc": f"{outputs['accuracy'].item():.4f}"
                        })
            
            if progress_bar is not None:
                progress_bar.close()
            
            avg_loss = epoch_loss / max(num_batches, 1)
            avg_acc = epoch_accuracy / max(num_batches, 1)
            
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
            
            if (epoch + 1) % self.config.save_steps == 0 or epoch == num_epochs - 1:
                self._save_model(output_dir, epoch + 1)
        
        self._save_model(output_dir, num_epochs)
        
        self.logger.info("训练完成！")
        
        return self.model
    
    def _save_model(self, output_dir: str, epoch: int):
        """
        保存模型检查点
        
        保存内容包括：
        1. 基础模型权重
        2. LoRA 适配器权重（如果使用）
        3. 奖励预测头权重
        4. 分词器
        5. 训练配置
        """
        if not self.accelerator.is_local_main_process:
            return
        
        epoch_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        if self.finetuning_type == "lora":
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            if hasattr(unwrapped_model, 'base_model'):
                unwrapped_model.base_model.save_pretrained(epoch_dir)
            else:
                unwrapped_model.save_pretrained(epoch_dir)
        else:
            self.accelerator.unwrap_model(self.model).save_pretrained(epoch_dir)
        
        self.tokenizer.save_pretrained(epoch_dir)
        
        self.config.save_to_json(os.path.join(epoch_dir, "training_config.json"))
        
        self.logger.info(f"模型已保存至：{epoch_dir}")


class RewardModel(torch.nn.Module):
    """
    奖励模型包装器
    
    将基础语言模型与奖励预测头组合：
    - 基础模型：编码输入文本为隐藏状态
    - 奖励头：将隐藏状态映射为标量分数
    
    训练时：
    - 可以选择冻结基础模型参数，只训练奖励头
    - 或端到端训练所有参数
    
    预测时：
    - 输出 prompt + response 的质量评分
    - 评分越高表示越符合人类偏好
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        reward_head: torch.nn.Module
    ):
        super().__init__()
        self.base_model = base_model
        self.reward_head = reward_head
        
        self._freeze_base_model()
    
    def _freeze_base_model(self):
        """
        冻结基础模型参数
        
        只训练奖励预测头可以：
        1. 减少显存占用
        2. 加快训练速度
        3. 保留预训练知识
        """
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def unfreeze_base_model(self):
        """
        解冻基础模型参数
        
        端到端训练可以获得更好的效果，
        但需要更多计算资源。
        """
        for param in self.base_model.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入 token ID 序列
            attention_mask: 注意力掩码
            
        Returns:
            奖励分数
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        hidden_states = outputs.last_hidden_state
        
        rewards = self.reward_head(hidden_states)
        
        return rewards
