import os
import torch
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from datasets import Dataset
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from transformers.trainer_utils import get_last_checkpoint
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import DistributedType
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import setup_logger
from config.training_config import SFTTrainingConfig


class SFTTrainer:
    """
    SFT (Supervised Fine-Tuning) Trainer
    
    支持：
    1. 全量微调 (Full Fine-tuning)
    2. LoRA 微调 (Low-Rank Adaptation)
    3. 4-bit/8-bit 量化
    4. DeepSpeed 加速
    5. 多卡分布式训练 (Accelerate)
    6. LoRA adapter 保存和合并
    """
    
    def __init__(
        self,
        config: SFTTrainingConfig,
        finetuning_type: str = "lora",
        use_deepspeed: bool = False,
        deepspeed_config: Optional[Dict] = None
    ):
        """
        初始化 SFT 训练器
        
        Args:
            config: 训练配置
            finetuning_type: 微调类型 ("lora" 或 "full")
            use_deepspeed: 是否使用 DeepSpeed
            deepspeed_config: DeepSpeed 配置文件路径或配置字典
        """
        self.config = config
        self.finetuning_type = finetuning_type
        self.use_deepspeed = use_deepspeed
        self.deepspeed_config = deepspeed_config
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.accelerator = None
        self.lora_config = None
        
        self.logger = setup_logger(self.__class__.__name__, level="INFO")
        
        self._validate_config()
    
    def _validate_config(self):
        """验证配置"""
        if self.finetuning_type not in ["lora", "full"]:
            raise ValueError(f"finetuning_type 必须是 'lora' 或 'full'，但得到了 '{self.finetuning_type}'")
        
        if self.use_deepspeed and self.config.use_4bit:
            self.logger.warning("DeepSpeed 与 4-bit 量化可能存在兼容性问题，请谨慎使用")
    
    def _initialize_accelerator(self):
        """初始化 Accelerator"""
        if self.accelerator is None:
            kwargs = DistributedDataParallelKwargs(
                find_unused_parameters=True,
                broadcast_buffers=False
            )
            self.accelerator = Accelerator(
                kwargs_handlers=[kwargs],
                deepspeed_plugin=self.deepspeed_config if self.use_deepspeed else None
            )
            
            if self.accelerator.is_main_process:
                self.logger.info(f"Accelerator 初始化完成，分布式类型: {self.accelerator.distributed_type}")
                if torch.cuda.is_available():
                    self.logger.info(f"可用 GPU 数量: {torch.cuda.device_count()}")
    
    def _get_default_target_modules(self, model_name: str) -> List[str]:
        """获取默认的目标模块"""
        if "qwen" in model_name.lower():
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "llama" in model_name.lower() or "mistral" in model_name.lower():
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "bloom" in model_name.lower():
            return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        elif "gpt2" in model_name.lower():
            return ["c_attn", "c_proj", "c_fc"]
        else:
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    def _setup_lora_config(self) -> LoraConfig:
        """设置 LoRA 配置"""
        target_modules = self.config.target_modules or self._get_default_target_modules(self.config.model_name_or_path)
        
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            modules_to_save=None,
            fan_in_fan_out=False,
            peft_type="LORA"
        )
        
        self.logger.info(f"LoRA 配置: rank={self.config.lora_rank}, alpha={self.config.lora_alpha}")
        return self.lora_config
    
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        try:
            self._initialize_accelerator()
            
            self.logger.info(f"加载模型: {self.config.model_name_or_path}")
            self.logger.info(f"微调类型: {self.finetuning_type}")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=self.config.trust_remote_code,
                padding_side="right"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.logger.info("已将 pad_token 设置为 eos_token")
            
            # 模型加载配置
            model_kwargs = {
                "trust_remote_code": self.config.trust_remote_code,
                "torch_dtype": self._get_dtype(),
                "device_map": self.config.device_map if torch.cuda.is_available() else None,
            }
            
            # 4-bit 量化配置
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
            
            # 加载基础模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                **model_kwargs
            )
            
            # 如果使用 LoRA
            if self.finetuning_type == "lora":
                self.logger.info("配置 LoRA 微调...")
                
                self.model.enable_input_require_grads()
                self.logger.info("已调用 enable_input_require_grads() 以支持梯度计算")
                
                lora_config = self._setup_lora_config()
                
                self.model = get_peft_model(self.model, lora_config)
                self.model.print_trainable_parameters()
                
                for name, param in self.model.named_parameters():
                    if "lora_" in name or param.requires_grad:
                        param.requires_grad_(True)
                
                self.logger.info("已确保所有 LoRA 参数 requires_grad=True")
            
            # 不使用梯度检查点（避免兼容性问题）
            # if self.config.use_gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            #     self.model.gradient_checkpointing_enable()
            #     self.logger.info("启用梯度检查点")
            
            # 移动到 CPU（如果无 GPU）
            if not torch.cuda.is_available():
                self.model = self.model.to("cpu")
            
            # 准备模型用于训练
            self.model = self.accelerator.prepare(self.model)
            
            self.logger.info("模型和分词器加载完成")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise
    
    def _get_dtype(self) -> torch.dtype:
        """获取数据类型"""
        if self.config.use_bf16 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        elif self.config.use_fp16 and torch.cuda.is_available():
            return torch.float16
        else:
            return torch.float32
    
    def prepare_dataset(
        self,
        dataset: Union[Dataset, List[Dict]],
        messages_field: str = "messages",
        max_seq_length: Optional[int] = None
    ) -> Dataset:
        """
        准备训练数据集（支持 OpenAI messages 格式）
        
        Args:
            dataset: 输入数据集
            messages_field: messages 字段名
            max_seq_length: 最大序列长度
            
        Returns:
            处理后的数据集
        """
        if isinstance(dataset, list):
            dataset = Dataset.from_list(dataset)
        
        max_seq_length = max_seq_length or self.config.max_seq_length
        
        def format_messages_func(examples):
            """格式化对话消息"""
            formatted_data = []
            
            for messages in examples[messages_field]:
                if not isinstance(messages, list) or len(messages) == 0:
                    formatted_data.append("")
                    continue
                
                text_parts = []
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    
                    if role == "system":
                        text_parts.append(f"<|system|>\n{content}")
                    elif role == "user":
                        text_parts.append(f"<|user|>\n{content}")
                    elif role == "assistant":
                        text_parts.append(f"<|assistant|>\n{content}")
                
                full_text = "\n".join(text_parts)
                if not full_text.endswith("<|assistant|>\n"):
                    if not full_text.endswith("<|assistant|>"):
                        full_text += "\n<|assistant|>\n"
                    else:
                        full_text += "\n"
                else:
                    full_text += "\n"
                
                formatted_data.append(full_text)
            
            return {"text": formatted_data}
        
        columns_to_remove = [col for col in dataset.column_names if col == messages_field]
        
        dataset = dataset.map(
            format_messages_func,
            batched=True,
            remove_columns=columns_to_remove,
            desc="格式化对话数据"
        )
        
        self.logger.info(f"数据集准备完成，包含 {len(dataset)} 个样本")
        return dataset
    
    def tokenize_dataset(
        self,
        dataset: Dataset,
        max_seq_length: Optional[int] = None
    ) -> Dataset:
        """
        对数据集进行分词
        
        Args:
            dataset: 输入数据集
            max_seq_length: 最大序列长度
            
        Returns:
            分词后的数据集
        """
        max_seq_length = max_seq_length or self.config.max_seq_length
        
        def tokenize_function(examples):
            """分词函数"""
            texts = examples.get("text", [])
            
            if not texts or len(texts) == 0:
                return {
                    "input_ids": [[0] * max_seq_length],
                    "attention_mask": [[0] * max_seq_length],
                    "labels": [[-100] * max_seq_length]
                }
            
            cleaned_texts = []
            for text in texts:
                if text is None:
                    cleaned_texts.append("")
                elif not isinstance(text, str):
                    cleaned_texts.append(str(text))
                else:
                    cleaned_texts.append(text)
            
            try:
                tokenized = self.tokenizer(
                    cleaned_texts,
                    truncation=True,
                    padding="max_length",
                    max_length=max_seq_length,
                    return_tensors=None,
                    add_special_tokens=True
                )
            except ValueError as e:
                self.logger.error(f"分词失败: {e}, texts: {cleaned_texts[:3]}")
                raise
            
            labels = [list(ids) for ids in tokenized["input_ids"]]
            
            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": labels
            }
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="分词数据集"
        )
        
        self.logger.info(f"分词完成，数据集包含 {len(tokenized_dataset)} 个样本")
        return tokenized_dataset
    
    def create_training_args(
        self,
        output_dir: str,
        **training_kwargs
    ) -> TrainingArguments:
        """
        创建训练参数
        
        Args:
            output_dir: 输出目录
            **training_kwargs: 其他训练参数
            
        Returns:
            训练参数对象
        """
        resume_from_checkpoint = training_kwargs.get("resume_from_checkpoint", None)
        
        if resume_from_checkpoint is None:
            if self.accelerator.is_main_process:
                last_checkpoint = get_last_checkpoint(output_dir)
                if last_checkpoint is not None:
                    resume_from_checkpoint = last_checkpoint
                    self.logger.info(f"发现检查点，将从 {resume_from_checkpoint} 恢复训练")
        
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
            resume_from_checkpoint=resume_from_checkpoint,
            optim="adamw_torch" if self.finetuning_type == "full" else "paged_adamw_8bit",
            seed=self.config.seed if hasattr(self.config, 'seed') else 42,
            local_rank=self.accelerator.local_process_index if hasattr(self.accelerator, 'local_process_index') else -1,
        )
        
        if self.use_deepspeed:
            training_args.deepspeed = self.deepspeed_config
        
        batch_size = self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps
        
        num_processes = 1
        if hasattr(self.accelerator, 'state') and hasattr(self.accelerator.state, 'num_processes'):
            num_processes = self.accelerator.state.num_processes
        elif hasattr(self.accelerator, 'num_processes'):
            num_processes = self.accelerator.num_processes
        
        if num_processes > 1:
            batch_size *= num_processes
        
        self.logger.info(f"训练参数: {self.config.num_train_epochs} epochs, batch_size={batch_size}")
        return training_args
    
    def train(
        self,
        dataset: Union[Dataset, List[Dict]],
        output_dir: str,
        messages_field: str = "messages",
        eval_dataset: Optional[Union[Dataset, List[Dict]]] = None,
        **training_kwargs
    ) -> Dict:
        """
        开始训练
        
        Args:
            dataset: 训练数据集
            output_dir: 输出目录
            messages_field: messages 字段名
            eval_dataset: 评估数据集
            **training_kwargs: 其他训练参数
            
        Returns:
            训练结果字典
        """
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
        
        os.makedirs(output_dir, exist_ok=True)
        
        prepared_dataset = self.prepare_dataset(dataset, messages_field)
        tokenized_dataset = self.tokenize_dataset(prepared_dataset)
        
        eval_tokenized_dataset = None
        if eval_dataset is not None:
            prepared_eval_dataset = self.prepare_dataset(eval_dataset, messages_field)
            eval_tokenized_dataset = self.tokenize_dataset(prepared_eval_dataset)
        
        training_args = self.create_training_args(output_dir, **training_kwargs)
        
        if eval_tokenized_dataset is not None and training_kwargs.get("do_eval", True):
            training_args.do_eval = True
            training_args.evaluation_strategy = "steps"
            training_args.eval_steps = training_kwargs.get("eval_steps", training_args.save_steps // 2)
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=eval_tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        self.trainer = self.accelerator.prepare(self.trainer)
        
        if self.accelerator.is_main_process:
            self.logger.info("开始训练...")
        
        train_result = self.trainer.train()
        
        self._save_model(output_dir)
        
        if self.accelerator.is_main_process:
            self.logger.info(f"训练完成！损失: {train_result.training_loss:.4f}")
            self.logger.info(f"模型保存到: {output_dir}")
        
        metrics = train_result.metrics
        return metrics
    
    def _save_model(self, output_dir: str):
        """保存模型"""
        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            
            if self.finetuning_type == "lora":
                try:
                    self.model.save_pretrained(
                        output_dir,
                        safe_serialization=True,
                        save_adapter_config=True
                    )
                    self.logger.info(f"LoRA adapter 保存到: {output_dir}")
                    
                    import json
                    adapter_config_path = os.path.join(output_dir, "adapter_config.json")
                    if not os.path.exists(adapter_config_path):
                        config_dict = {
                            "base_model_name_or_path": self.config.model_name_or_path,
                            "task_type": "CAUSAL_LM",
                            "lora_alpha": self.config.lora_alpha,
                            "lora_dropout": self.config.lora_dropout,
                            "r": self.config.lora_rank,
                            "bias": "none",
                            "target_modules": self.config.target_modules or self._get_default_target_modules(self.config.model_name_or_path),
                            "modules_to_save": None,
                            "fan_in_fan_out": False,
                            "peft_type": "LORA"
                        }
                        with open(adapter_config_path, 'w') as f:
                            json.dump(config_dict, f, indent=2)
                        self.logger.info(f"创建 adapter_config.json")
                except Exception as e:
                    self.logger.warning(f"保存 LoRA adapter 失败: {e}，尝试使用 Trainer 保存")
                    self.trainer.save_model(output_dir)
            else:
                self.trainer.save_model(output_dir)
                self.logger.info(f"全量模型保存到: {output_dir}")
            
            self.tokenizer.save_pretrained(output_dir)
    
    def save_lora_adapter(self, output_dir: str, save_model: bool = True):
        """
        保存 LoRA adapter
        
        Args:
            output_dir: 输出目录
            save_model: 是否同时保存基础模型
        """
        if self.finetuning_type != "lora":
            raise ValueError("只有 LoRA 微调模式才能保存 adapter")
        
        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            self.logger.info(f"LoRA adapter 保存到: {output_dir}")
    
    def merge_lora_and_save(
        self,
        base_model_path: str,
        adapter_path: str,
        output_dir: str,
        save_model: bool = True
    ):
        """
        合并 LoRA adapter 到基础模型并保存
        
        Args:
            base_model_path: 基础模型路径
            adapter_path: adapter 路径
            output_dir: 输出目录
            save_model: 是否保存合并后的模型
        """
        if not self.accelerator:
            self._initialize_accelerator()
        
        if self.accelerator.is_main_process:
            try:
                self.logger.info(f"加载基础模型: {base_model_path}")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    torch_dtype=self._get_dtype(),
                    device_map="auto"
                )
                
                self.logger.info(f"加载 LoRA adapter: {adapter_path}")
                merged_model = PeftModel.from_pretrained(base_model, adapter_path)
                merged_model = merged_model.merge_and_unload()
                
                if save_model:
                    os.makedirs(output_dir, exist_ok=True)
                    
                    tokenizer = AutoTokenizer.from_pretrained(
                        base_model_path,
                        trust_remote_code=self.config.trust_remote_code
                    )
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    
                    merged_model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    self.logger.info(f"合并后的模型保存到: {output_dir}")
                
                return merged_model
                
            except Exception as e:
                self.logger.error(f"模型合并失败: {e}")
                raise
    
    def load_inference_model(self, model_path: str, adapter_path: Optional[str] = None):
        """
        加载推理模型
        
        Args:
            model_path: 模型路径
            adapter_path: 可选的 adapter 路径
        """
        self._initialize_accelerator()
        
        self.logger.info(f"加载推理模型: {model_path}")
        
        if not os.path.exists(model_path):
            raise ValueError(f"模型路径不存在: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.config.trust_remote_code
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": self._get_dtype(),
            "device_map": "auto" if torch.cuda.is_available() else None,
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        
        if adapter_path is not None and self.finetuning_type == "lora":
            if os.path.exists(adapter_path):
                self.logger.info(f"加载 LoRA adapter: {adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
        
        if not torch.cuda.is_available():
            self.model = self.model.to("cpu")
        
        self.model = self.accelerator.prepare(self.model)
        self.model.eval()
        
        self.logger.info(f"推理模型加载完成")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成令牌数
            temperature: 温度参数
            top_p: top-p 采样参数
            top_k: top-k 采样参数
            repetition_penalty: 重复惩罚
            do_sample: 是否使用采样
            **kwargs: 其他生成参数
            
        Returns:
            生成的文本
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型未加载，请先调用 load_inference_model()")
        
        device = next(self.model.parameters()).device
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = response[len(prompt):].strip()
        
        return generated_text
