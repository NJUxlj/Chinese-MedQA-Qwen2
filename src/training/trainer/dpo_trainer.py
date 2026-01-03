"""
DPOTrainer - 基于直接偏好优化（Direct Preference Optimization）的对齐训练器

================================================================================
DPO 核心原理详解
================================================================================

【什么是 DPO？】

DPO（Direct Preference Optimization，直接偏好优化）是一种用于大语言模型对齐的新方法，
由 Stanford 大学等机构于 2023 年提出。它的核心思想是：绕过传统 RLHF 中复杂的强化学习
过程，直接通过偏好数据优化模型。

【为什么需要 DPO？】

传统 RLHF（基于人类反馈的强化学习）存在以下问题：
1. 需要训练一个独立的奖励模型（Reward Model）
2. 使用 PPO 算法进行策略优化，训练不稳定
3. 涉及四个模型（Actor, Critic, Reward, Reference），计算开销大
4. 超参数调优困难

DPO 的革命性在于：将 RLHF 转化为简单的监督学习问题，只需要两个模型即可。

================================================================================
DPO 数学原理
================================================================================

【偏好数据的格式】

每条训练样本包含：
- prompt (x): 用户输入的提示
- chosen (y_w): 人类偏好的回答（winner）
- rejected (y_l): 人类不偏好的回答（loser）

【优化目标】

给定偏好对 (y_w, y_l)，我们希望模型 π_θ 满足：
  log(π_θ(y_w|x) / π_ref(y_w|x)) - log(π_θ(y_l|x) / π_ref(y_l|x)) > 0

其中：
- π_θ: 当前训练的策略模型（我们想要优化的模型）
- π_ref: 参考模型（通常是 SFT 后的初始模型，不更新）
- β: temperature 参数，控制偏好强度

【损失函数】

L_DPO = -E_{(x, y_w, y_l) ~ D} [ log(σ(β * (log(π_θ(y_w|x)) - log(π_ref(y_w|x)) 
                                         - log(π_θ(y_l|x)) + log(π_ref(y_l|x))))) ]

其中 σ 是 sigmoid 函数。

【直观理解】

- 当 y_w 的概率相对于参考模型提高，且 y_l 的概率相对于参考模型降低时，损失降低
- β 越大，对偏好差异的惩罚越重，训练越保守
- log(π_θ(y|x) / π_ref(y|x)) 可以理解为模型对答案 y 的"偏好得分"

【与 RLHF 的关系】

当使用 KL 散度约束时，PPO 的优化目标可以简化为：
L_RLHF ≈ -E_{(x, y_w, y_l)} [ log(σ(β * (r(x, y_w) - r(x, y_l)))) ]

其中 r(x, y) 是奖励模型给出的分数。DPO 证明了：
  r(x, y) = β * log(π(y|x) / π_ref(y|x))

因此，我们可以直接使用策略模型的概率比来替代奖励模型，省去奖励模型的训练！

================================================================================
代码实现要点
================================================================================

【参考模型的管理】

1. 参考模型 π_ref 是参考模型的深拷贝，在训练过程中不更新梯度
2. 每个训练步开始时，需要确保 ref_model 的参数与 π_ref 一致
3. 为什么要用深拷贝？因为我们要保留一个"不带梯度"的固定参考点

【训练策略】

1. 偏好数据构建：
   - 可以来自人类标注
   - 可以来自模型采样后的人工筛选
   - 可以来自规则或奖励模型的高分/低分答案

2. 数据格式设计：
   - prompt 字段：用户输入
   - chosen 字段：偏好的回答
   - rejected 字段：不偏好的回答

3. 训练技巧：
   - β 通常取 0.1 到 0.5
   - 学习率通常比 SFT 稍低（如 1e-5 到 5e-5）
   - 可以使用 LoRA 进行参数高效微调

【与 SFT 的区别】

- SFT：监督学习，学习生成符合格式的文本
- DPO：偏好学习，学习区分"好的回答"和"坏的回答"
- 两者通常结合使用：先用 SFT 建立基础能力，再用 DPO 进行对齐

================================================================================
"""

import os
import sys
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass
from datasets import Dataset
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    TrainerCallback
)
from transformers import TrainerControl
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator, DistributedDataParallelKwargs
try:
    from deepspeed import DeepSpeedEngine
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DeepSpeedEngine = None
    DEEPSPEED_AVAILABLE = False

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.logger import setup_logger
from config.training_config import DPOTrainingConfig
from training.trainer.base_trainer import BaseTrainer


class DPODataset(Dataset):
    """
    DPO 偏好数据集封装
    
    DPO 数据与标准 SFT 数据不同，每条样本包含：
    - input_ids: prompt 的分词结果
    - attention_mask: 注意力掩码
    - chosen_labels: 偏好答案的分词结果
    - rejected_labels: 不偏好答案的分词结果
    """
    
    def __init__(self, tokenized_data: Dict[str, List]):
        self.data = tokenized_data
    
    def __len__(self) -> int:
        return len(self.data["input_ids"])
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.data["input_ids"][idx],
            "attention_mask": self.data["attention_mask"][idx],
            "chosen_labels": self.data["chosen_labels"][idx],
            "rejected_labels": self.data["rejected_labels"][idx]
        }


class DPOTrainer(BaseTrainer):
    """
    DPO（直接偏好优化）训练器
    
    继承自 BaseTrainer，利用基础框架进行模型加载和训练管理，
    专注于 DPO 特有的偏好学习逻辑。
    
    核心功能：
    1. 加载基础模型和分词器（继承自 BaseTrainer）
    2. 准备偏好数据集（prompt, chosen, rejected）
    3. 计算 DPO 损失函数
    4. 管理参考模型用于对比学习
    
    使用示例：
        config = DPOTrainingConfig(
            model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            learning_rate=1e-5,
            num_train_epochs=3
        )
        trainer = DPOTrainer(config)
        trainer.start_training(dataset, output_dir)
    """
    
    def __init__(
        self,
        config: DPOTrainingConfig,
        finetuning_type: str = "lora",
        beta: float = 0.1,
        use_deepspeed: bool = False,
        deepspeed_config: Optional[Union[str, Dict]] = None
    ):
        """
        初始化 DPO 训练器
        
        Args:
            config: DPO 训练配置
            finetuning_type: 微调类型 ("lora" 或 "full")
            beta: DPO 温度参数，控制偏好学习的强度
                  - β 越大，对偏好差异越敏感，训练越保守
                  - β 越小，模型更容易改变，训练更激进
                  - 常用范围: 0.05 ~ 0.5
            use_deepspeed: 是否使用 DeepSpeed 加速
            deepspeed_config: DeepSpeed 配置文件路径或配置字典
        """
        super().__init__(config)
        
        self.finetuning_type = finetuning_type
        self.beta = beta
        self.use_deepspeed = use_deepspeed
        self.deepspeed_config = deepspeed_config
        
        self.lora_config = None
        self.ref_model = None
        self.ref_model_idx = 0
        
        self._validate_config()
        self.logger.info(f"DPO 训练器初始化完成: beta={beta}, finetuning_type={finetuning_type}")
    
    def _validate_config(self):
        """验证配置有效性"""
        if self.finetuning_type not in ["lora", "full"]:
            raise ValueError(f"finetuning_type 必须是 'lora' 或 'full'，但得到了 '{self.finetuning_type}'")
        
        if self.use_deepspeed and self.config.use_4bit:
            self.logger.warning("DeepSpeed 与 4-bit 量化可能存在兼容性问题，请谨慎使用")
    
    def _get_default_target_modules(self, model_name: str) -> List[str]:
        """获取模型默认的目标模块（用于 LoRA）"""
        model_name_lower = model_name.lower()
        
        if "qwen" in model_name_lower:
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif any(x in model_name_lower for x in ["llama", "mistral", "yi", "deepseek"]):
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "bloom" in model_name_lower:
            return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        elif "gpt2" in model_name_lower:
            return ["c_attn", "c_proj", "c_fc"]
        elif any(x in model_name_lower for x in ["gpt-neo", "gptneox", "gpt_j", "gptj"]):
            return ["q_proj", "k_proj", "v_proj", "out_proj", "c_fc", "c_proj"]
        else:
            self.logger.warning(f"未识别模型类型 {model_name}，使用默认目标模块")
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    def _setup_lora_config(self) -> LoraConfig:
        """
        配置 LoRA 参数高效微调
        
        LoRA 的核心思想：
        - 在原始权重矩阵 W_0 旁添加低秩分解矩阵 AB
        - 前向计算: h = W_0 x + BAx
        - 训练时只更新 A 和 B，大幅减少参数量
        """
        if self.config.target_modules is not None:
            target_modules = self.config.target_modules
            self.logger.info(f"使用指定的 target_modules: {target_modules}")
        else:
            model_name = os.path.basename(self.config.model_name_or_path) if os.path.exists(self.config.model_name_or_path) else self.config.model_name_or_path
            target_modules = self._get_default_target_modules(model_name)
            self.logger.info(f"自动检测到模型: {model_name}, target_modules: {target_modules}")
        
        lora_rank = getattr(self.config, 'lora_rank', 64)
        lora_alpha = getattr(self.config, 'lora_alpha', 16)
        lora_dropout = getattr(self.config, 'lora_dropout', 0.05)
        
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            modules_to_save=None,
            fan_in_fan_out=False,
            peft_type="LORA"
        )
        
        self.logger.info(f"LoRA 配置: rank={lora_rank}, alpha={lora_alpha}, target_modules={target_modules}")
        return self.lora_config
    
    def load_model_and_tokenizer(self):
        """
        加载模型和分词器（重写 BaseTrainer 方法）
        
        DPO 特有步骤：
        1. 加载基础模型
        2. 如果使用 LoRA，附加 LoRA adapter
        3. 创建参考模型的深拷贝（不参与梯度更新）
        """
        try:
            self.logger.info(f"{self.__class__.__name__}: 加载模型: {self.config.model_name_or_path}")
            
            self._initialize_accelerator()
            
            self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=self.config.trust_remote_code,
                padding_side="right"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.logger.info("已将 pad_token 设置为 eos_token")
            
            model_kwargs = {
                "trust_remote_code": self.config.trust_remote_code,
                "torch_dtype": self._get_dtype(),
                "device_map": self.config.device_map if torch.cuda.is_available() else None,
            }
            
            if self.config.use_4bit:
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16 if self.config.use_bf16 else torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    self.logger.info("启用 4-bit 量化")
                except ImportError:
                    self.logger.warning("未安装 bitsandbytes，禁用 4-bit 量化")
                    self.config.use_4bit = False
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                **model_kwargs
            )
            
            if self.finetuning_type == "lora":
                self.logger.info("配置 LoRA 微调...")
                
                self.model.enable_input_require_grads()
                
                lora_config = self._setup_lora_config()
                self.model = get_peft_model(self.model, lora_config)
                self.model.print_trainable_parameters()
                
                for name, param in self.model.named_parameters():
                    if "lora_" in name or param.requires_grad:
                        param.requires_grad_(True)
                
                self.logger.info("已确保所有 LoRA 参数 requires_grad=True")
            
            if not torch.cuda.is_available():
                self.model = self.model.to("cpu")
            
            self.model = self.accelerator.prepare(self.model)
            
            self.logger.info("模型和分词器加载完成")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise
    
    def _get_dtype(self) -> torch.dtype:
        """获取最佳计算精度"""
        if self.config.use_bf16 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        elif self.config.use_fp16 and torch.cuda.is_available():
            return torch.float16
        else:
            return torch.float32
    
    def _initialize_accelerator(self):
        """初始化 Accelerator（用于分布式训练）"""
        if self.accelerator is None:
            kwargs = DistributedDataParallelKwargs(
                find_unused_parameters=True,
                broadcast_buffers=False
            )
            
            deepspeed_plugin = None
            if self.use_deepspeed and self.deepspeed_config:
                if isinstance(self.deepspeed_config, str):
                    import json
                    with open(self.deepspeed_config, 'r') as f:
                        deepspeed_plugin = json.load(f)
                else:
                    deepspeed_plugin = self.deepspeed_config
            
            self.accelerator = Accelerator(
                kwargs_handlers=[kwargs],
                deepspeed_plugin=deepspeed_plugin
            )
            
            if self.accelerator.is_main_process:
                self.logger.info(f"Accelerator 初始化完成，分布式类型: {self.accelerator.distributed_type}")
                if torch.cuda.is_available():
                    self.logger.info(f"可用 GPU 数量: {torch.cuda.device_count()}")
    
    def _create_ref_model(self) -> AutoModelForCausalLM:
        """
        创建参考模型（Reference Model）

        参考模型的作用：
        - 提供一个固定的基准，用于计算概率比
        - 确保训练过程中模型不会偏离原始策略太远
        - 不参与梯度计算（requires_grad=False）

        实现方式：
        - 从原始模型路径加载模型
        - 不需要从当前 LoRA 模型加载权重（参考模型应该是原始模型）
        - 移动到相同设备
        - 设置 eval 模式
        """
        self.logger.info(f"创建参考模型: {self.config.model_name_or_path}")

        ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=self._get_dtype(),
            device_map=self.config.device_map if torch.cuda.is_available() else None,
        )

        ref_model.requires_grad_(False)
        ref_model.eval()

        if torch.cuda.is_available():
            ref_model = ref_model.to("cuda")

        self.logger.info("参考模型创建完成")
        return ref_model
    
    def prepare_dataset(
        self,
        dataset: Union[Dataset, List[Dict]],
        prompt_field: str = "prompt",
        chosen_field: str = "chosen",
        rejected_field: str = "rejected",
        max_prompt_length: Optional[int] = None,
        max_response_length: Optional[int] = None
    ) -> Dataset:
        """
        准备 DPO 偏好数据集
        
        处理偏好数据的三元组格式：
        {
            "prompt": "用户问题",
            "chosen": "偏好的回答",
            "rejected": "不偏好的回答"
        }
        
        Args:
            dataset: 输入数据集
            prompt_field: prompt 字段名
            chosen_field: 偏好答案字段名
            rejected_field: 不偏好答案字段名
            max_prompt_length: 最大 prompt 长度
            max_response_length: 最大回答长度
            
        Returns:
            处理后的数据集
        """
        if isinstance(dataset, list):
            dataset = Dataset.from_list(dataset)
        
        max_prompt_length = max_prompt_length or 512
        max_response_length = max_response_length or (self.config.max_seq_length - max_prompt_length - 2)
        
        def filter_and_format(examples):
            """过滤无效样本并格式化"""
            filtered = {
                "prompt": [],
                "chosen": [],
                "rejected": []
            }
            
            for i in range(len(examples.get(prompt_field, []))):
                prompt = examples[prompt_field][i]
                chosen = examples[chosen_field][i]
                rejected = examples[rejected_field][i]
                
                if self._is_valid_preference_sample(prompt, chosen, rejected):
                    filtered["prompt"].append(prompt)
                    filtered["chosen"].append(chosen)
                    filtered["rejected"].append(rejected)
            
            return filtered
        
        def tokenize_function(examples):
            """
            DPO 专用的 tokenize 处理
            
            数据流程：
            1. 将 prompt + chosen 拼接并分词 → 用于计算 chosen 的对数概率
            2. 将 prompt + rejected 拼接并分词 → 用于计算 rejected 的对数概率
            """
            prompts = examples["prompt"]
            chosen_texts = examples["chosen"]
            rejected_texts = examples["rejected"]
            
            chosen_texts = [f"User: {p}\nAssistant: {c}" for p, c in zip(prompts, chosen_texts)]
            rejected_texts = [f"User: {p}\nAssistant: {r}" for p, r in zip(prompts, rejected_texts)]
            
            chosen_tokens = self.tokenizer(
                chosen_texts,
                max_length=max_prompt_length + max_response_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            rejected_tokens = self.tokenizer(
                rejected_texts,
                max_length=max_prompt_length + max_response_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            batch = {
                "input_ids": chosen_tokens["input_ids"],
                "attention_mask": chosen_tokens["attention_mask"],
                "chosen_labels": chosen_tokens["input_ids"],
                "rejected_labels": rejected_tokens["input_ids"]
            }
            
            return batch
        
        columns_to_remove = [col for col in dataset.column_names 
                           if col not in [prompt_field, chosen_field, rejected_field]]
        
        dataset = dataset.map(
            filter_and_format,
            batched=True,
            remove_columns=columns_to_remove,
            desc="过滤无效偏好样本"
        )
        
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="分词偏好数据"
        )
        
        self.logger.info(f"偏好数据集准备完成，包含 {len(dataset)} 个有效样本")
        return dataset
    
    def _is_valid_preference_sample(
        self,
        prompt: Any,
        chosen: Any,
        rejected: Any
    ) -> bool:
        """检查偏好样本的有效性"""
        if not prompt or not chosen or not rejected:
            return False
        
        if not isinstance(prompt, str) or not isinstance(chosen, str) or not isinstance(rejected, str):
            return False
        
        if len(prompt) > 2048 or len(chosen) > 2048 or len(rejected) > 2048:
            return False
        
        if chosen == rejected:
            return False
        
        return True
    
    def tokenize_dataset(
        self,
        dataset: Dataset,
        max_seq_length: Optional[int] = None
    ) -> Dataset:
        """
        对数据集进行分词（继承方法，增加 DPO 特定处理）
        
        注意：DPO 数据的分词在 prepare_dataset 中已完成，
        此方法主要用于保持接口一致性
        """
        self.logger.info("DPO 数据集已在前置处理中完成分词")
        return dataset
    
    def _get_log_probs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算给定 logits 和标签的对数概率
        
        Args:
            logits: 模型输出的 logits，shape = (batch_size, seq_len, vocab_size)
            labels: 目标标签，shape = (batch_size, seq_len)
            
        Returns:
            每个位置的对数概率，shape = (batch_size, seq_len)
        
        计算过程：
        1. 对 logits 在 vocab 维度上计算 log_softmax
        2. 使用 torch.gather 根据 labels 索引收集对应的对数概率
        
        示例：
            logits = [
                [[-0.5, -1.0, -2.0],  # position 0
                 [-0.8, -1.2, -1.5]], # position 1
                [[-0.6, -1.1, -2.1],  # batch 2
                 [-0.9, -1.3, -1.6]]
            ]  # shape: (2, 2, 3)
            
            labels = [[0, 2], [1, 0]]  # shape: (2, 2)
            
            结果 = [
                [-0.5, -1.5],  # batch 0: log_probs[0,0,0], log_probs[0,1,2]
                [-1.1, -0.9]   # batch 1: log_probs[1,0,1], log_probs[1,1,0]
            ]
        """
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
        return log_probs
    
    def _compute_dpo_loss(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 DPO 损失函数
        
        DPO 损失的核心思想：
        - 增大 chosen 回答相对于参考模型的对数概率比
        - 减小 rejected 回答相对于参考模型的对数概率比
        
        数学公式：
        L = -log(σ(β * (Δ_log_prob_chosen - Δ_log_prob_rejected)))
        
        其中：
        - Δ_log_prob_chosen = log(π_policy(y_w|x)) - log(π_ref(y_w|x))
        - Δ_log_prob_rejected = log(π_policy(y_l|x)) - log(π_ref(y_l|x))
        - σ 是 sigmoid 函数
        
        直观理解：
        - 当 policy 对 chosen 的偏好高于 ref 时，Δ_log_prob_chosen 增大
        - 当 policy 对 rejected 的偏好低于 ref 时，Δ_log_prob_rejected 减小
        - 两者差值越大，sigmoid 输出越接近 1，log(sigmoid) 越接近 0
        - 损失最小化等价于最大化 chosen 被选中的概率
        """
        chosen_log_probs = self._get_log_probs(policy_logits, chosen_labels)
        rejected_log_probs = self._get_log_probs(policy_logits, rejected_labels)
        
        with torch.no_grad():
            ref_chosen_log_probs = self._get_log_probs(ref_logits, chosen_labels)
            ref_rejected_log_probs = self._get_log_probs(ref_logits, rejected_labels)
        
        policy_chosen_reward = self.beta * (chosen_log_probs - ref_chosen_log_probs)
        policy_rejected_reward = self.beta * (rejected_log_probs - ref_rejected_log_probs)
        
        logits = policy_chosen_reward - policy_rejected_reward
        
        loss = -F.logsigmoid(logits)
        
        if attention_mask is not None:
            loss = (loss * attention_mask).sum() / attention_mask.sum()
        
        return loss
    
    def create_training_args(
        self,
        output_dir: str,
        **training_kwargs
    ) -> TrainingArguments:
        """创建训练参数"""
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
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            max_grad_norm=self.config.max_grad_norm,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            remove_unused_columns=self.config.remove_unused_columns,
            do_train=self.config.do_train,
            do_eval=self.config.do_eval,
            fp16=self.config.use_fp16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            bf16=self.config.use_bf16 and torch.cuda.is_bf16_supported(),
            dataloader_num_workers=self.config.dataloader_num_workers,
            save_safetensors=True,
            optim="adamw_torch" if self.finetuning_type == "full" else "paged_adamw_8bit",
        )
        
        if self.use_deepspeed:
            training_args.deepspeed = self.deepspeed_config
        
        self.logger.info(f"训练参数: {self.config.num_train_epochs} epochs, batch_size={self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps}")
        return training_args
    
    def dpo_collator(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        DPO 数据整理器
        
        将多个样本整理成一个 batch：
        - input_ids: prompt + chosen 的分词结果
        - attention_mask: 注意力掩码
        - chosen_labels: chosen 回答的分词结果
        - rejected_labels: rejected 回答的分词结果
        """
        batch = {}
        for key in ["input_ids", "attention_mask", "chosen_labels", "rejected_labels"]:
            values = [f[key] for f in features]
            if isinstance(values[0], torch.Tensor):
                batch[key] = torch.stack(values)
            else:
                batch[key] = torch.tensor(values)
        return batch
    
    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        计算评估指标
        
        DPO 的核心指标：
        - preference_accuracy: policy 对 chosen 的得分是否高于 rejected
        """
        logits_chosen, logits_rejected = eval_pred.predictions
        
        chosen_scores = logits_chosen.mean(axis=-1)
        rejected_scores = logits_rejected.mean(axis=-1)
        
        accuracy = (chosen_scores > rejected_scores).mean()
        return {"dpo_accuracy": float(accuracy)}
    
    def start_training(
        self,
        dataset: Union[Dataset, List[Dict]],
        output_dir: str,
        prompt_field: str = "prompt",
        chosen_field: str = "chosen",
        rejected_field: str = "rejected",
        eval_dataset: Optional[Union[Dataset, List[Dict]]] = None,
        **training_kwargs
    ):
        """
        开始 DPO 训练
        
        Args:
            dataset: 偏好训练数据集，包含 prompt, chosen, rejected 字段
            output_dir: 模型输出目录
            prompt_field: prompt 字段名
            chosen_field: 偏好答案字段名
            rejected_field: 不偏好答案字段名
            eval_dataset: 评估数据集
            **training_kwargs: 其他训练参数
        """
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
        
        os.makedirs(output_dir, exist_ok=True)
        
        prepared_dataset = self.prepare_dataset(
            dataset,
            prompt_field=prompt_field,
            chosen_field=chosen_field,
            rejected_field=rejected_field
        )
        
        eval_tokenized_dataset = None
        if eval_dataset is not None:
            prepared_eval_dataset = self.prepare_dataset(
                eval_dataset,
                prompt_field=prompt_field,
                chosen_field=chosen_field,
                rejected_field=rejected_field
            )
            eval_tokenized_dataset = prepared_eval_dataset
        
        training_args = self.create_training_args(output_dir, **training_kwargs)
        
        if eval_tokenized_dataset is not None and training_kwargs.get("do_eval", True):
            training_args.do_eval = True
            training_args.evaluation_strategy = "steps"
            training_args.eval_steps = training_kwargs.get("eval_steps", training_args.save_steps // 2)
        
        self.ref_model = self._create_ref_model()
        
        class CustomDPOTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                """
                自定义 DPO 损失计算

                数据格式：
                - input_ids: prompt + chosen 的分词结果
                - attention_mask: 注意力掩码
                - chosen_labels: chosen 回答的分词结果
                - rejected_labels: rejected 回答的分词结果

                损失计算步骤：
                1. 取出 input_ids 和 labels
                2. 政策模型前向传播获取 logits
                3. 参考模型前向传播获取 ref_logits（无梯度）
                4. 计算 chosen 和 rejected 的对数概率差
                5. 应用 DPO 损失函数
                """
                input_ids = inputs.pop("input_ids")
                attention_mask = inputs.pop("attention_mask")
                chosen_labels = inputs.pop("chosen_labels")
                rejected_labels = inputs.pop("rejected_labels")

                policy_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                policy_logits = policy_outputs.logits

                with torch.no_grad():
                    if hasattr(model, 'base_model'):
                        base_model = model.base_model
                        ref_outputs = base_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                    else:
                        ref_outputs = model.module(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                    ref_logits = ref_outputs.logits

                if hasattr(base_model, '_compute_dpo_loss'):
                    loss = base_model._compute_dpo_loss(
                        policy_logits=policy_logits,
                        ref_logits=ref_logits,
                        chosen_labels=chosen_labels,
                        rejected_labels=rejected_labels,
                        attention_mask=attention_mask
                    )
                else:
                    loss = self.model.module._compute_dpo_loss(
                        policy_logits=policy_logits,
                        ref_logits=ref_logits,
                        chosen_labels=chosen_labels,
                        rejected_labels=rejected_labels,
                        attention_mask=attention_mask
                    )

                if return_outputs:
                    return loss, {"logits": policy_logits}

                return loss
        
        self.trainer = CustomDPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=prepared_dataset,
            eval_dataset=eval_tokenized_dataset,
            data_collator=self.dpo_collator,
            compute_metrics=self._compute_metrics,
            callbacks=[DPOCallback(beta=self.beta)]
        )
        
        self.trainer = self.accelerator.prepare(self.trainer)
        
        if self.accelerator.is_main_process:
            self.logger.info("开始 DPO 训练...")
        
        train_result = self.trainer.train()
        
        self._save_model(output_dir)
        
        if self.accelerator.is_main_process:
            self.logger.info(f"DPO 训练完成！损失: {train_result.training_loss:.4f}")
            self.logger.info(f"模型保存到: {output_dir}")
        
        return train_result.metrics
    
    def _save_model(self, output_dir: str):
        """保存模型"""
        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            
            if self.finetuning_type == "lora":
                self.model.save_pretrained(
                    output_dir,
                    safe_serialization=True,
                    save_adapter_config=True
                )
                self.logger.info(f"LoRA adapter 保存到: {output_dir}")
            else:
                self.trainer.save_model(output_dir)
                self.logger.info(f"完整模型保存到: {output_dir}")
            
            self.tokenizer.save_pretrained(output_dir)
            self.logger.info(f"分词器保存到: {output_dir}")


class DPOCallback(TrainerCallback):
    """
    DPO 训练回调
    
    功能：
    1. 初始化参考模型
    2. 在每个 epoch 开始时同步参考模型参数
    """
    
    def __init__(self, beta: float = 0.1):
        self.beta = beta
        self.ref_model = None
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerControl, control: TrainerControl, **kwargs):
        """训练开始时的初始化"""
        model = kwargs.get("model")
        if model is None:
            return
        
        if DEEPSPEED_AVAILABLE and isinstance(model, DeepSpeedEngine):
            model = model.module
        
        self.ref_model = model
        control.should_epoch_stop = False

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerControl, control: TrainerControl, **kwargs):
        """每个 epoch 开始时同步参考模型"""
        if self.ref_model is None:
            return
        
        model = kwargs.get("model")
        if model is None:
            return
        
        if DEEPSPEED_AVAILABLE and isinstance(model, DeepSpeedEngine):
            model = model.module
        
        self.ref_model.load_state_dict(model.state_dict())
        self.ref_model.eval()
    
    def on_step_end(self, args: TrainingArguments, state: TrainerControl, control: TrainerControl, **kwargs):
        """每个 step 结束时检查是否需要同步"""
        if state.global_step % 100 == 0:
            if self.ref_model is not None:
                model = kwargs.get("model")
                if model is not None and (not DEEPSPEED_AVAILABLE or not isinstance(model, DeepSpeedEngine)):
                    self.ref_model.load_state_dict(model.state_dict())





def run():
    pass





if __name__ == "__main__":
    run()
