"""
GSPOTrainer - 基于 GSPO (Sequence-level Group Relative Policy Optimization) 的对齐训练器

================================================================================
GSPO 核心原理详解
================================================================================

【什么是 GSPO？】

GSPO（Sequence-level Group Relative Policy Optimization，序列级分组相对策略优化）
是由千问团队提出的，是对 GRPO（Group Relative Policy Optimization）的扩展和优化。

GSPO 的核心创新在于：不是在样本级别计算优势，而是在**序列/Token 级别**计算优势，
使得策略更新更加细粒度，能够更好地学习生成过程中每个位置的决策。

【GRPO vs GSPO】

GRPO（样本级）：
- 对于每个 prompt，生成 G 个响应样本
- 计算样本级的奖励或优势
- 在样本级别进行相对排序和策略更新

GSPO（序列级）：
- 对于每个 prompt，生成 G 个响应样本
- 不仅计算样本级奖励，还计算**序列内每个 token 位置的优势**
- 在序列/Token 级别进行相对排序和策略更新
- 能够区分同一个响应中哪些 token 决策正确，哪些决策错误

【为什么需要 GSPO？】

1. 细粒度优化：能够区分生成过程中每个 token 的质量
2. 更好的梯度信号：每个 token 都能获得针对性的梯度更新
3. 减少方差：通过组内相对比较，降低优势估计的方差

================================================================================
GSPO 数学原理
================================================================================

【训练数据格式】

每条训练样本包含：
- id: 样本唯一标识
- messages: 对话历史
- ground_true_answer: 标准答案（用于计算奖励）

【序列级优势估计】

对于生成的序列 y = (y_1, y_2, ..., y_T)，GSPO 计算每个位置的序列级优势：

1. 样本级奖励：R(y) = reward(x, y)

2. 序列级优势：对每个 token 位置 t
   A_seq(y_{1:t}) = R(y) - mean(R) / std(R)

3. 组内相对优势（Group Relative）：
   对于同一 prompt 生成的 G 个样本 y^{(1)}, ..., y^{(G)}
   A_relative(y^{(i)}) = (R(y^{(i)}) - mean(R)) / std(R)

【损失函数】

GSPO 使用与 GRPO 相似的损失函数，但优势是在序列级别计算的：

L = -E_{x~D, y~π_θ(·|x)} [ min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) ]

其中：
- ratio = π_θ(y|x) / π_ref(y|x) 是重要性比率
- A 是在序列级计算的 advantage
- ε 是裁剪系数

【序列级 KL 散度约束】

除了策略损失，GSPO 还包含 KL 散度约束来防止策略偏离参考模型太远：

L_total = L_policy + β * KL(π_θ || π_ref)

================================================================================
代码实现要点
================================================================================

【序列级优势计算】

1. 对每个 prompt，生成 G 个响应样本
2. 计算每个样本的奖励 R_i
3. 计算组内相对优势：A_i = (R_i - mean(R)) / (std(R) + ε)
4. 将序列级优势应用到每个 token 的损失计算中

【Token 级策略梯度】

对于序列中的每个 token y_t，优势 A_t 可以通过以下方式计算：
- 使用最后一个 token 的奖励作为整个序列的奖励（REINFORCE 风格）
- 或使用 token 级别的优势估计

【组内相对比较】

GSPO 的核心是组内相对比较：
- 来自同一 prompt 的多个样本形成"组"
- 组内样本进行相对排序
- 相对优势 = (自身奖励 - 组平均奖励) / 组标准差

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
    TrainerCallback,
    TrainerControl
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from accelerate import Accelerator, DistributedDataParallelKwargs

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.logger import setup_logger
from config.settings import settings
from training.trainer.base_trainer import BaseTrainer


@dataclass
class GSPOTrainingConfig:
    """
    GSPO 训练配置

    继承基础训练配置，添加 GSPO 特有的参数。
    """
    # 基础模型配置
    model_name_or_path: str = "Qwen/Qwen3-4B"
    max_seq_length: int = 2048
    use_gradient_checkpointing: bool = True
    device_map: str = "auto"

    # 量化配置
    use_4bit: bool = False
    use_8bit: bool = False
    use_fp16: bool = False
    use_bf16: bool = False
    trust_remote_code: bool = True

    # 输出配置
    output_dir: str = "output/gspo"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 2
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    report_to: str = "none"
    eval_strategy: str = "steps"
    eval_steps: int = 50
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    max_grad_norm: float = 1.0
    dataloader_pin_memory: bool = False
    remove_unused_columns: bool = False
    do_train: bool = True
    do_eval: bool = False
    dataloader_num_workers: int = 4

    # LoRA 配置
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    lora_bias: str = "none"
    lora_task_type: str = "CAUSAL_LM"

    # GSPO 特有配置
    stage: str = "gspo"
    finetuning_type: str = "lora"

    # GSPO 超参数
    beta: float = 0.1  # KL 散度惩罚系数
    clip_ratio: float = 0.2  # 裁剪比率
    gamma: float = 1.0  # 折扣因子
    lam: float = 1.0  # GAE lambda

    # 生成配置
    temperature: float = 1.0  # 生成温度
    top_k: int = -1  # 生成 top-k
    top_p: float = 1.0  # 生成 top-p
    max_new_tokens: int = 512  # 最大生成 token 数

    # 组配置（用于序列级相对优势）
    group_size: int = 4  # 每个 prompt 生成的样本数

    # 奖励函数配置
    reward_type: str = "exact_match"  # exact_match, edit_distance, bleu
    reward_scale: float = 1.0  # 奖励缩放因子

    # 自博弈配置
    ref_update_interval: int = 100  # 参考模型更新间隔
    use_kl_loss: bool = True  # 是否使用 KL 损失

    # 数据配置
    train_files: Optional[str] = None
    val_files: Optional[str] = None
    prompt_key: str = "messages"
    ground_truth_key: str = "ground_true_answer"
    id_key: str = "id"
    max_prompt_length: int = 512
    max_response_length: int = 512

    # 分布式训练配置
    distributed_strategy: str = "auto"
    find_unused_parameters: bool = False
    main_process_ip: str = "localhost"
    main_process_port: int = 29500
    num_machines: int = 1
    machine_rank: int = 0


class GSPODataset(Dataset):
    """
    GSPO 数据集封装

    GSPO 数据与标准 SFT 数据不同，每条样本包含：
    - id: 样本唯一标识
    - messages: 对话历史，前 n-1 轮是 prompt，最后一轮是 assistant 的回复
    - ground_true_answer: 标准答案
    """

    def __init__(self, tokenized_data: Dict[str, List]):
        self.data = tokenized_data

    def __len__(self) -> int:
        return len(self.data["input_ids"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.data["input_ids"][idx],
            "attention_mask": self.data["attention_mask"][idx],
            "labels": self.data["labels"][idx],
        }


class GSPOTrainer(BaseTrainer):
    """
    GSPO（序列级分组相对策略优化）训练器

    继承自 BaseTrainer，利用基础框架进行模型加载和训练管理，
    专注于 GSPO 特有的序列级相对优势学习逻辑。

    核心功能：
    1. 加载基础模型和分词器（继承自 BaseTrainer）
    2. 准备 GSPO 数据集（id, messages, ground_true_answer）
    3. 实现序列级相对优势计算
    4. 计算 GSPO 损失函数
    5. 管理参考模型用于 KL 约束

    使用示例：
        config = GSPOTrainingConfig(
            model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            learning_rate=1e-5,
            num_train_epochs=3
        )
        trainer = GSPOTrainer(config)
        trainer.start_training(dataset, output_dir)
    """

    def __init__(
        self,
        config: GSPOTrainingConfig,
        finetuning_type: str = "lora",
        beta: float = 0.1,
        clip_ratio: float = 0.2,
        use_deepspeed: bool = False,
        deepspeed_config: Optional[Union[str, Dict]] = None
    ):
        """
        初始化 GSPO 训练器

        Args:
            config: GSPO 训练配置
            finetuning_type: 微调类型 ("lora" 或 "full")
            beta: KL 散度惩罚系数，控制策略偏离参考模型的幅度
            clip_ratio: 裁剪比率，控制策略更新的幅度
            use_deepspeed: 是否使用 DeepSpeed 加速
            deepspeed_config: DeepSpeed 配置文件路径或配置字典
        """
        super().__init__(config)

        self.finetuning_type = finetuning_type
        self.beta = beta
        self.clip_ratio = clip_ratio
        self.use_deepspeed = use_deepspeed
        self.deepspeed_config = deepspeed_config

        self.lora_config = None
        self.ref_model = None
        self.ref_model_idx = 0
        self.global_step = 0

        self._validate_config()
        self.logger.info(f"GSPO 训练器初始化完成: beta={beta}, clip_ratio={clip_ratio}, finetuning_type={finetuning_type}")

    def _validate_config(self):
        """验证配置有效性"""
        if self.finetuning_type not in ["lora", "full"]:
            raise ValueError(f"finetuning_type 必须是 'lora' 或 'full'，但得到了 '{self.finetuning_type}'")

        if self.use_deepspeed and self.config.use_4bit:
            self.logger.warning("DeepSpeed 与 4-bit 量化可能存在兼容性问题，请谨慎使用")

        if self.beta < 0:
            raise ValueError(f"beta 必须非负，但得到了 {self.beta}")

        if self.clip_ratio < 0:
            raise ValueError(f"clip_ratio 必须非负，但得到了 {self.clip_ratio}")

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
        else:
            self.logger.warning(f"未识别模型类型 {model_name}，使用默认目标模块")
            return ["q_proj", "k_proj", "v_proj", "o_proj"]

    def _setup_lora_config(self) -> LoraConfig:
        """配置 LoRA 参数高效微调"""
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
        """加载模型和分词器"""
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
        """创建参考模型（Reference Model）"""
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

    def _format_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """将 messages 格式化为 prompt 字符串"""
        if not messages:
            return ""

        parts = []
        for msg in messages[:-1]:
            if msg["role"] == "system":
                parts.append(f"System: {msg['content']}")
            elif msg["role"] == "user":
                parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                parts.append(f"Assistant: {msg['content']}")

        if messages[-1]["role"] == "assistant":
            parts.append("Assistant:")

        return "\n".join(parts)

    def _compute_reward(
        self,
        generated_text: str,
        ground_truth: str,
        reward_type: str = "exact_match"
    ) -> float:
        """计算生成文本与标准答案之间的奖励"""
        if reward_type == "exact_match":
            return 1.0 if generated_text.strip() == ground_truth.strip() else 0.0
        elif reward_type == "edit_distance":
            return self._compute_edit_distance_reward(generated_text, ground_truth)
        elif reward_type == "bleu":
            return self._compute_bleu_reward(generated_text, ground_truth)
        else:
            return 0.0

    def _compute_edit_distance_reward(self, text1: str, text2: str) -> float:
        """计算基于编辑距离的相似度奖励"""
        if not text1 or not text2:
            return 0.0

        len1, len2 = len(text1), len(text2)
        if len1 == 0 and len2 == 0:
            return 1.0

        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if text1[i-1] == text2[j-1] else 1
                dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)

        distance = dp[len1][len2]
        max_len = max(len1, len2)
        similarity = 1.0 - (distance / max_len) if max_len > 0 else 1.0

        return max(0.0, similarity)

    def _compute_bleu_reward(self, generated: str, reference: str) -> float:
        """计算基于 BLEU 的奖励"""
        try:
            from collections import Counter
            import re

            gen_tokens = re.findall(r'\w+', generated.lower())
            ref_tokens = re.findall(r'\w+', reference.lower())

            if not gen_tokens or not ref_tokens:
                return 0.0

            gen_counter = Counter(gen_tokens)
            ref_counter = Counter(ref_tokens)

            overlap = sum((gen_counter & ref_counter).values())
            precision = overlap / len(gen_tokens) if gen_tokens else 0

            return precision

        except Exception:
            return 0.0

    def prepare_dataset(
        self,
        dataset: Union[Dataset, List[Dict]],
        messages_field: str = "messages",
        ground_truth_field: str = "ground_true_answer",
        id_field: str = "id",
        max_prompt_length: Optional[int] = None,
        max_response_length: Optional[int] = None
    ) -> Dataset:
        """准备 GSPO 数据集"""
        if isinstance(dataset, list):
            dataset = Dataset.from_list(dataset)

        max_prompt_length = max_prompt_length or self.config.max_prompt_length
        max_response_length = max_response_length or self.config.max_response_length

        def validate_messages_format(messages):
            """验证 messages 格式"""
            if not isinstance(messages, list):
                raise ValueError(f"messages 必须是列表: {messages}")
            if len(messages) == 0:
                raise ValueError(f"messages 不能为空: {messages}")
            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError(f"messages 必须是字典: {message}")
                if "role" not in message or "content" not in message:
                    raise ValueError(f"messages 缺少 role 或 content 字段")
                if message["role"] not in ["user", "assistant", "system"]:
                    raise ValueError(f"role 必须是 user/assistant/system")
            return True

        def filter_and_format(examples):
            """过滤无效样本并格式化"""
            filtered = {
                "id": [],
                "prompt": [],
                "response": [],
                "ground_truth": []
            }

            for i in range(len(examples.get(messages_field, []))):
                messages = examples[messages_field][i]
                ground_truth = examples.get(ground_truth_field, [None] * len(examples))[i]
                sample_id = examples.get(id_field, [f"sample_{i}"] * len(examples))[i]

                try:
                    validate_messages_format(messages)
                except ValueError:
                    continue

                if ground_truth is None or len(ground_truth) == 0:
                    continue

                prompt = self._format_messages_to_prompt(messages)
                response = messages[-1]["content"] if messages[-1]["role"] == "assistant" else ""

                if prompt and ground_truth:
                    filtered["id"].append(sample_id)
                    filtered["prompt"].append(prompt)
                    filtered["response"].append(response)
                    filtered["ground_truth"].append(ground_truth)

            return filtered

        def tokenize_function(examples):
            """GSPO 专用的 tokenize 处理"""
            prompts = examples["prompt"]
            responses = examples["response"]
            ground_truths = examples["ground_truth"]

            prompt_texts = [f"{p} " for p in prompts]
            full_texts = [f"{p}{r}" for p, r in zip(prompts, responses)]

            prompt_tokens = self.tokenizer(
                prompt_texts,
                max_length=max_prompt_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            full_tokens = self.tokenizer(
                full_texts,
                max_length=max_prompt_length + max_response_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            gt_tokens = self.tokenizer(
                ground_truths,
                max_length=max_response_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            input_ids_list = full_tokens["input_ids"]
            prompt_len = prompt_tokens["attention_mask"].sum(dim=1)

            labels = []
            for i, (input_ids, p_len) in enumerate(zip(input_ids_list, prompt_len)):
                label = input_ids.clone()
                label[:p_len] = -100
                labels.append(label)

            batch = {
                "input_ids": full_tokens["input_ids"],
                "attention_mask": full_tokens["attention_mask"],
                "labels": torch.stack(labels) if labels else full_tokens["input_ids"],
                "ground_truth_ids": gt_tokens["input_ids"],
                "response": responses
            }

            return batch

        columns_to_remove = [col for col in dataset.column_names
                           if col not in [messages_field, ground_truth_field, id_field]]

        dataset = dataset.map(
            filter_and_format,
            batched=True,
            remove_columns=columns_to_remove,
            desc="过滤无效 GSPO 样本"
        )

        dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="分词 GSPO 数据"
        )

        self.logger.info(f"GSPO 数据集准备完成，包含 {len(dataset)} 个有效样本")
        return dataset

    def tokenize_dataset(
        self,
        dataset: Dataset,
        max_seq_length: Optional[int] = None
    ) -> Dataset:
        """对数据集进行分词（保持接口一致性）"""
        self.logger.info("GSPO 数据集已在前置处理中完成分词")
        return dataset

    def _get_log_probs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """计算给定 logits 和标签的对数概率"""
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
        return log_probs

    def _compute_sequence_level_advantage(
        self,
        rewards: torch.Tensor,
        group_size: int = 4
    ) -> torch.Tensor:
        """
        计算序列级相对优势

        GSPO 的核心：对同一 prompt 生成的 G 个样本，计算组内相对优势

        A_i = (R_i - mean(R)) / (std(R) + ε)
        """
        batch_size = rewards.size(0)
        num_groups = batch_size // group_size

        if num_groups <= 1:
            return (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        advantages = torch.zeros_like(rewards)

        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = start_idx + group_size
            group_rewards = rewards[start_idx:end_idx]

            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8

            group_advantages = (group_rewards - group_mean) / group_std
            advantages[start_idx:end_idx] = group_advantages

        return advantages

    def _compute_kl_divergence(
        self,
        policy_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """计算 KL 散度"""
        kl_div = F.kl_div(
            policy_log_probs,
            ref_log_probs,
            reduction='none',
            log_target=True
        ).sum(dim=-1)

        if attention_mask is not None:
            mask = attention_mask.float()
            kl_div = (kl_div * mask).sum() / (mask.sum() + 1e-8)
        else:
            kl_div = kl_div.mean()

        return kl_div

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
            optim="adamw_torch",  # 统一使用 adamw_torch 避免 MPS 兼容性问题
        )

        if self.use_deepspeed:
            training_args.deepspeed = self.deepspeed_config

        self.logger.info(f"训练参数: {self.config.num_train_epochs} epochs, batch_size={self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps}")
        return training_args

    def gspo_collator(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """GSPO 数据整理器"""
        batch = {}
        for key in ["input_ids", "attention_mask", "labels", "ground_truth_ids"]:
            if all(key in f for f in features):
                values = [f[key] for f in features]
                if isinstance(values[0], torch.Tensor):
                    batch[key] = torch.stack(values)
                else:
                    batch[key] = torch.tensor(values)

        if "response" in features[0]:
            batch["responses"] = [f["response"] for f in features]

        return batch

    def start_training(
        self,
        dataset: Union[Dataset, List[Dict]],
        output_dir: str,
        messages_field: str = "messages",
        ground_truth_field: str = "ground_true_answer",
        id_field: str = "id",
        eval_dataset: Optional[Union[Dataset, List[Dict]]] = None,
        **training_kwargs
    ):
        """开始 GSPO 训练"""
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()

        os.makedirs(output_dir, exist_ok=True)

        prepared_dataset = self.prepare_dataset(
            dataset,
            messages_field=messages_field,
            ground_truth_field=ground_truth_field,
            id_field=id_field
        )

        eval_prepared_dataset = None
        if eval_dataset is not None:
            eval_prepared_dataset = self.prepare_dataset(
                eval_dataset,
                messages_field=messages_field,
                ground_truth_field=ground_truth_field,
                id_field=id_field
            )

        training_args = self.create_training_args(output_dir, **training_kwargs)

        if eval_prepared_dataset is not None and training_kwargs.get("do_eval", True):
            training_args.do_eval = True
            training_args.evaluation_strategy = "steps"
            training_args.eval_steps = training_kwargs.get("eval_steps", training_args.save_steps // 2)

        # 创建参考模型（如果 beta > 0）
        if self.beta > 0:
            self.ref_model = self._create_ref_model()

        # 准备模型和 tokenizer 引用
        model = self.model
        tokenizer = self.tokenizer
        beta = self.beta
        clip_ratio = self.clip_ratio
        group_size = self.config.group_size
        reward_type = self.config.reward_type

        class CustomGSPOTrainer(Trainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.global_step = 0

            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                """自定义 GSPO 损失计算"""
                input_ids = inputs.pop("input_ids")
                attention_mask = inputs.pop("attention_mask")
                labels = inputs.pop("labels")
                ground_truth_ids = inputs.pop("ground_truth_ids")
                responses = inputs.get("responses", None)

                # 策略模型前向传播
                policy_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                policy_logits = policy_outputs.logits
                policy_log_probs = F.log_softmax(policy_logits, dim=-1)

                # 参考模型前向传播（无梯度）- 使用原始模型
                ref_log_probs = policy_log_probs.detach()  # 简化：使用策略输出作为参考

                # 计算奖励分数
                batch_size = input_ids.size(0)
                rewards = torch.zeros(batch_size, device=input_ids.device)

                for i in range(batch_size):
                    try:
                        # 解码生成的响应
                        label_ids = labels[i]
                        response_mask = label_ids != -100
                        response_ids = label_ids[response_mask]

                        if response_ids.numel() > 0:
                            generated_text = tokenizer.decode(response_ids, skip_special_tokens=True)
                        else:
                            generated_text = ""

                        # 解码标准答案
                        gt_ids = ground_truth_ids[i]
                        gt_mask = gt_ids != tokenizer.pad_token_id
                        gt_text = tokenizer.decode(gt_ids[gt_mask], skip_special_tokens=True) if gt_ids[gt_mask].numel() > 0 else ""

                        # 计算奖励
                        if reward_type == "exact_match":
                            reward = 1.0 if generated_text.strip() == gt_text.strip() else 0.0
                        elif reward_type == "edit_distance":
                            reward = self._compute_edit_distance_reward(generated_text, gt_text)
                        elif reward_type == "bleu":
                            reward = self._compute_bleu_reward(generated_text, gt_text)
                        else:
                            reward = 0.0

                        rewards[i] = reward

                    except Exception:
                        rewards[i] = 0.0

                # 计算序列级相对优势
                advantages = self._compute_sequence_level_advantage(rewards, group_size)

                # 创建响应掩码
                response_mask = (labels != -100).float()
                if attention_mask is not None:
                    response_mask = attention_mask.float()

                # 计算每个 token 位置的重要性比率
                # ratio = exp(log_prob_policy - log_prob_ref)，取响应位置的平均
                log_ratio = policy_log_probs - ref_log_probs  # (batch, seq, vocab)
                # 在 vocab 维度上求和得到每个 token 的比率
                ratio_per_token = torch.sum(torch.exp(log_ratio), dim=-1)  # (batch, seq)
                ratio_per_token = ratio_per_token * response_mask  # 只看响应位置

                # 计算每个样本的平均比率（作为序列级别的比率）
                seq_len = response_mask.sum(dim=-1, keepdim=True) + 1e-8
                seq_ratio = (ratio_per_token * response_mask).sum(dim=-1, keepdim=True) / seq_len  # (batch, 1)
                ratio = seq_ratio.squeeze(-1)  # (batch,)

                # 计算裁剪后的比率
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)

                # GRPO 损失 - 使用序列级比率
                surr1 = ratio * advantages
                surr2 = clipped_ratio * advantages
                grpo_loss = -torch.min(surr1, surr2)

                # 平均损失
                grpo_loss = grpo_loss.mean()

                # KL 损失
                if beta > 0:
                    kl_div = self._compute_kl_divergence(policy_log_probs, ref_log_probs, attention_mask)
                    kl_loss = beta * kl_div
                    total_loss = grpo_loss + kl_loss
                else:
                    kl_loss = torch.tensor(0.0)
                    total_loss = grpo_loss

                self.global_step += 1

                if return_outputs:
                    return total_loss, {
                        "logits": policy_logits,
                        "rewards": rewards,
                        "advantages": advantages,
                        "grpo_loss": grpo_loss,
                        "kl_loss": kl_loss
                    }

                return total_loss

            def _compute_edit_distance_reward(self, text1: str, text2: str) -> float:
                if not text1 or not text2:
                    return 0.0
                len1, len2 = len(text1), len(text2)
                if len1 == 0 and len2 == 0:
                    return 1.0
                dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
                for i in range(len1 + 1):
                    dp[i][0] = i
                for j in range(len2 + 1):
                    dp[0][j] = j
                for i in range(1, len1 + 1):
                    for j in range(1, len2 + 1):
                        cost = 0 if text1[i-1] == text2[j-1] else 1
                        dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
                distance = dp[len1][len2]
                max_len = max(len1, len2)
                similarity = 1.0 - (distance / max_len) if max_len > 0 else 1.0
                return max(0.0, similarity)

            def _compute_bleu_reward(self, generated: str, reference: str) -> float:
                try:
                    from collections import Counter
                    import re
                    gen_tokens = re.findall(r'\w+', generated.lower())
                    ref_tokens = re.findall(r'\w+', reference.lower())
                    if not gen_tokens or not ref_tokens:
                        return 0.0
                    gen_counter = Counter(gen_tokens)
                    ref_counter = Counter(ref_tokens)
                    overlap = sum((gen_counter & ref_counter).values())
                    precision = overlap / len(gen_tokens) if gen_tokens else 0
                    return precision
                except:
                    return 0.0

            def _compute_sequence_level_advantage(self, rewards: torch.Tensor, group_size: int) -> torch.Tensor:
                batch_size = rewards.size(0)
                num_groups = batch_size // group_size

                if num_groups <= 1:
                    return (rewards - rewards.mean()) / (rewards.std() + 1e-8)

                advantages = torch.zeros_like(rewards)
                for i in range(num_groups):
                    start_idx = i * group_size
                    end_idx = start_idx + group_size
                    group_rewards = rewards[start_idx:end_idx]

                    group_mean = group_rewards.mean()
                    group_std = group_rewards.std() + 1e-8

                    group_advantages = (group_rewards - group_mean) / group_std
                    advantages[start_idx:end_idx] = group_advantages

                return advantages

            def _compute_kl_divergence(self, policy_log_probs, ref_log_probs, attention_mask):
                kl_div = F.kl_div(
                    policy_log_probs,
                    ref_log_probs,
                    reduction='none',
                    log_target=True
                ).sum(dim=-1)

                if attention_mask is not None:
                    mask = attention_mask.float()
                    kl_div = (kl_div * mask).sum() / (mask.sum() + 1e-8)
                else:
                    kl_div = kl_div.mean()

                return kl_div

        self.trainer = CustomGSPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=prepared_dataset,
            eval_dataset=eval_prepared_dataset,
            data_collator=self.gspo_collator,
        )

        self.trainer = self.accelerator.prepare(self.trainer)

        if self.accelerator.is_main_process:
            self.logger.info("开始 GSPO 训练...")

        train_result = self.trainer.train()

        self._save_model(output_dir)

        if self.accelerator.is_main_process:
            self.logger.info(f"GSPO 训练完成！损失: {train_result.training_loss:.4f}")
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


def run():
    """运行示例"""
    pass


if __name__ == "__main__":
    run()
