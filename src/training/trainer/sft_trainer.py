import os
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any
from datasets import Dataset
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    DataCollatorForSeq2Seq
)
from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator, DistributedDataParallelKwargs
from utils.logger import setup_logger
from trainer.base_trainer import BaseTrainer
from config.training_config import BaseTrainingConfig
from trainer.base_trainer import BaseTrainer


class SFTTrainer(BaseTrainer):
    '''
    基础训练器
    - 目的是为了训练模型
    - 基于 Transformers + Accelerator 框架实现高效的微调
    - 支持多卡并行训练
    '''

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-14B",
        max_seq_length: int = 2048,
        lora_rank: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        use_4bit: bool = False,
        use_gradient_checkpointing: bool = True,
        device_map: str = "auto"
    ):
        '''
        初始化 SFT 训练器
        
        Args:
            model_name: 模型名称或路径
            max_seq_length: 最大序列长度
            lora_rank: LoRA 秩
            lora_alpha: LoRA 缩放参数
            lora_dropout: LoRA dropout 率
            target_modules: 目标模块列表，None 表示使用默认配置
            use_4bit: 是否使用 4-bit 量化（注意：需要 bitsandbytes 库）
            use_gradient_checkpointing: 是否使用梯度检查点
            device_map: 设备映射策略
        '''
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.use_4bit = use_4bit
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.device_map = device_map
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.accelerator = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_model_and_tokenizer(self):
        '''加载模型和分词器'''
        try:
            self.logger.info(f"加载模型: {self.model_name}")
            
            # 初始化 Accelerator
            self.accelerator = Accelerator()
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="right"
            )
            
            # 设置特殊标记
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 模型加载配置
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "device_map": self.device_map if torch.cuda.is_available() else None,
            }
            
            # 如果使用 4-bit 量化
            if self.use_4bit:
                try:
                    from transformers import BitsAndBytesConfig
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
                    self.use_4bit = False
            
            # 加载基础模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # 如果使用 CPU，需要明确移动模型
            if not torch.cuda.is_available():
                self.model = self.model.to("cpu")
            
            # 配置 LoRA
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.target_modules,
                bias="none",
            )
            
            # 应用 LoRA
            self.model = get_peft_model(self.model, peft_config)
            
            # 启用梯度检查点
            if self.use_gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
                self.logger.info("启用梯度检查点")
            
            # 准备模型用于训练
            self.model = self.accelerator.prepare(self.model)
            
            self.logger.info("模型和分词器加载完成")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise

    def prepare_dataset(
        self,
        dataset: Union[Dataset, List[Dict]],
        input_field: str = "input",
        output_field: str = "output",
        max_seq_length: Optional[int] = None
    ) -> Dataset:
        '''
        准备训练数据集
        
        Args:
            dataset: 输入数据集
            input_field: 输入字段名
            output_field: 输出字段名
            max_seq_length: 最大序列长度
            
        Returns:
            处理后的数据集
        '''
        if isinstance(dataset, list):
            dataset = Dataset.from_list(dataset)
            
        max_seq_length = max_seq_length or self.max_seq_length
        
        def format_prompts_func(examples):
            texts = []
            for input_text, output_text in zip(examples[input_field], examples[output_field]):
                # 构建指令格式的输入
                prompt = f"用户: {input_text}\n助手: {output_text}"
                texts.append(prompt)
            
            return {"input_ids": texts}
        
        dataset = dataset.map(
            format_prompts_func,
            batched=True,
            remove_columns=[col for col in dataset.column_names if col not in [input_field, output_field]],
        )
        
        self.logger.info(f"数据集准备完成，包含 {len(dataset)} 个样本")
        return dataset

    def tokenize_dataset(
        self,
        dataset: Dataset,
        max_seq_length: Optional[int] = None
    ) -> Dataset:
        '''
        对数据集进行分词
        
        Args:
            dataset: 输入数据集
            max_seq_length: 最大序列长度
            
        Returns:
            分词后的数据集
        '''
        max_seq_length = max_seq_length or self.max_seq_length
        
        def tokenize_function(examples):
            # 如果已经有 input_ids 字段，直接使用
            if "input_ids" in examples:
                texts = examples["input_ids"]
            else:
                texts = examples.get("text", [])
            
            # 分词
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_seq_length,
                return_tensors=None
            )
            
            # 为因果语言模型添加 labels
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
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
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        per_device_eval_batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 2e-4,
        logging_steps: int = 10,
        save_steps: int = 500,
        save_total_limit: int = 2,
        warmup_ratio: float = 0.03,
        lr_scheduler_type: str = "cosine",
        report_to: str = "none",
        evaluation_strategy: str = "no",
        eval_steps: Optional[int] = None,
        load_best_model_at_end: bool = False,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: bool = False,
        max_grad_norm: float = 1.0,
        dataloader_pin_memory: bool = False,
        remove_unused_columns: bool = False,
        do_train: bool = True,
        do_eval: bool = False
    ) -> TrainingArguments:
        '''
        创建训练参数
        
        Args:
            output_dir: 输出目录
            num_train_epochs: 训练轮数
            per_device_train_batch_size: 每设备训练批次大小
            per_device_eval_batch_size: 每设备评估批次大小
            gradient_accumulation_steps: 梯度累积步数
            learning_rate: 学习率
            logging_steps: 日志记录步数
            save_steps: 保存步数
            save_total_limit: 保存限制
            warmup_ratio: 预热比例
            lr_scheduler_type: 学习率调度器类型
            report_to: 报告目标
            evaluation_strategy: 评估策略
            eval_steps: 评估步数
            load_best_model_at_end: 是否在结束时加载最佳模型
            metric_for_best_model: 最佳模型指标
            greater_is_better: 是否越大越好
            max_grad_norm: 最大梯度范数
            dataloader_pin_memory: 数据加载器固定内存
            remove_unused_columns: 移除未使用的列
            do_train: 是否训练
            do_eval: 是否评估
            
        Returns:
            训练参数对象
        '''
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            report_to=report_to,
            evaluation_strategy=evaluation_strategy,
            eval_steps=eval_steps,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            max_grad_norm=max_grad_norm,
            dataloader_pin_memory=dataloader_pin_memory,
            remove_unused_columns=remove_unused_columns,
            do_train=do_train,
            do_eval=do_eval,
            fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
            bf16=torch.cuda.is_bf16_supported(),
            dataloader_num_workers=0,
            group_by_length=True,
            length_column_name="length",
            disable_tqdm=False,
            save_safetensors=True,
        )
        
        self.logger.info(f"训练参数创建完成: {num_train_epochs} epochs, train_batch_size={per_device_train_batch_size * gradient_accumulation_steps}")
        return training_args

    def train(
        self,
        dataset: Union[Dataset, List[Dict]],
        output_dir: str = "./output",
        input_field: str = "input",
        output_field: str = "output",
        eval_dataset: Optional[Union[Dataset, List[Dict]]] = None,
        **training_kwargs
    ):
        '''
        开始训练
        
        Args:
            dataset: 训练数据集
            output_dir: 输出目录
            input_field: 输入字段名
            output_field: 输出字段名
            eval_dataset: 评估数据集
            **training_kwargs: 其他训练参数
        '''
        # 加载模型和分词器
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
        
        # 准备数据集
        prepared_dataset = self.prepare_dataset(dataset, input_field, output_field)
        tokenized_dataset = self.tokenize_dataset(prepared_dataset)
        
        # 准备评估数据集
        eval_tokenized_dataset = None
        if eval_dataset is not None:
            prepared_eval_dataset = self.prepare_dataset(eval_dataset, input_field, output_field)
            eval_tokenized_dataset = self.tokenize_dataset(prepared_eval_dataset)
        
        # 创建训练参数
        training_args = self.create_training_args(output_dir, **training_kwargs)
        
        # 如果有评估数据集，启用评估
        if eval_tokenized_dataset is not None:
            training_args.do_eval = True
            training_args.evaluation_strategy = "steps"
            training_args.eval_steps = training_kwargs.get("eval_steps", training_args.save_steps // 2)
        
        # 创建数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
        
        # 创建自定义 Trainer
        class CustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                
                # 计算交叉熵损失
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1)
                )
                
                return (loss, outputs) if return_outputs else loss
        
        # 创建训练器
        self.trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=eval_tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # 使用 Accelerator 准备训练器
        self.trainer = self.accelerator.prepare(self.trainer)
        
        # 开始训练
        self.logger.info("开始训练...")
        train_result = self.trainer.train()
        
        # 保存最终模型
        final_save_path = os.path.join(output_dir, "final_model")
        self.trainer.save_model(final_save_path)
        self.tokenizer.save_pretrained(final_save_path)
        
        self.logger.info(f"训练完成！模型已保存到: {final_save_path}")
        self.logger.info(f"训练损失: {train_result.training_loss:.4f}")
        
        return final_save_path

    def load_inference_model(self, model_path: str):
        '''
        加载推理模型
        
        Args:
            model_path: 模型路径
        '''
        try:
            self.logger.info(f"加载推理模型: {model_path}")
            
            # 初始化 Accelerator（如果还没有初始化）
            if self.accelerator is None:
                self.accelerator = Accelerator()
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # 设置特殊标记
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "device_map": "auto" if torch.cuda.is_available() else None,
            }
            
            # 如果使用 4-bit 量化
            if self.use_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    model_kwargs["quantization_config"] = quantization_config
                except ImportError:
                    self.logger.warning("未安装 bitsandbytes，禁用 4-bit 量化")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            
            # 如果使用 CPU，明确移动模型
            if not torch.cuda.is_available():
                self.model = self.model.to("cpu")
            
            # 准备模型用于推理
            self.model = self.accelerator.prepare(self.model)
            
            # 设置为评估模式
            self.model.eval()
            
            self.logger.info(f"推理模型加载完成: {model_path}")
            
        except Exception as e:
            self.logger.error(f"推理模型加载失败: {e}")
            raise

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
        '''
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
        '''
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型未加载，请先调用 load_inference_model()")
        
        # 构建输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length
        )
        
        # 移动到正确的设备
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 生成
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
                bos_token_id=self.tokenizer.bos_token_id,
                **kwargs
            )
        
        # 解码
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 移除输入部分，只返回生成的内容
        generated_text = response[len(prompt):].strip()
        
        # 移除可能的特殊前缀
        if generated_text.startswith("助手:"):
            generated_text = generated_text[3:].strip()
        
        return generated_text

    def merge_and_save(self, base_model_path: str, adapter_path: str, output_path: str):
        '''
        合并 LoRA 适配器到基础模型并保存
        
        Args:
            base_model_path: 基础模型路径
            adapter_path: 适配器路径
            output_path: 输出路径
        '''
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            
            from peft import PeftModel
            merged_model = PeftModel.from_pretrained(base_model, adapter_path)
            merged_model = merged_model.merge_and_unload()
            
            merged_model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
            
            self.logger.info(f"模型合并完成，保存到: {output_path}")
            
        except Exception as e:
            self.logger.error(f"模型合并失败: {e}")
            raise