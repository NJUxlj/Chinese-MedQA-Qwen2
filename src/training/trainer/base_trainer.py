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
from transformers import BitsAndBytesConfig


from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator, DistributedDataParallelKwargs
from utils.logger import setup_logger
from config.training_config import BaseTrainingConfig
from utils.metrics import MathEvaluator



class BaseTrainer:
    '''
    基础训练器
    - 目的是为了训练模型
    - 基于 Transformers + Accelerator 框架实现高效的微调
    - 支持多卡并行训练
    '''

    def __init__(
        self,
        config: BaseTrainingConfig
    ):
        '''
        初始化 SFT 训练器
        
        Args:
            model_name: 模型名称或路径
            max_seq_length: 最大序列长度
            use_4bit: 是否使用 4-bit 量化（注意：需要 bitsandbytes 库）
            use_gradient_checkpointing: 是否使用梯度检查点
            device_map: 设备映射策略
        '''
        self.config:BaseTrainingConfig = config
        self.model_name = config.model_name_or_path
        self.max_seq_length = config.max_seq_length
        self.use_4bit = config.use_4bit
        self.use_gradient_checkpointing = config.use_gradient_checkpointing
        self.device_map = config.device_map
        self.use_fp16 = config.use_fp16
        self.trust_remote_code = config.trust_remote_code
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.accelerator = None
        
        self.logger = setup_logger(self.__class__.__name__, level="INFO")

    def load_model_and_tokenizer(self):
        '''加载模型和分词器'''
        try:
            self.logger.info(f"{self.__class__.__name__}: 加载模型: {self.model_name}")
            
            # 初始化 Accelerator
            self.accelerator = Accelerator()
            
            # 加载分词器
            self.tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="right"
            )
            
            # 设置特殊标记
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 模型加载配置
            model_kwargs = {
                "trust_remote_code": self.trust_remote_code,
                "torch_dtype": torch.float16 if self.use_fp16 and torch.cuda.is_available() else torch.float32,
                "device_map": self.device_map if torch.cuda.is_available() else None,
            }
            
            # 如果使用 4-bit 量化
            if self.use_4bit:
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
                    self.use_4bit = False
            
            # 加载基础模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # 如果使用 CPU，需要明确移动模型
            if not torch.cuda.is_available():
                self.model = self.model.to("cpu")
        
            
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
        messages_field: str = "messages",
        ground_truth_field: str = "ground_true_answer",  # 用于兼容  GRPO 训练
        max_seq_length: Optional[int] = None
    ) -> Dataset:
        '''
        准备训练数据集（支持OpenAI messages格式）
        
        Args:
            dataset: 输入数据集
            messages_field: messages字段名
            ground_truth_field: ground truth字段名
            max_seq_length: 最大序列长度
            
        Returns:
            处理后的数据集
        '''
        if isinstance(dataset, list):
            dataset = Dataset.from_list(dataset)
            
        max_seq_length = max_seq_length or self.max_seq_length


        def validate_messages_format(messages):
            # 验证messages格式
            if not isinstance(messages, list):
                raise ValueError(f"messages必须是列表: {messages}")
            
            if len(messages) == 0:
                raise ValueError(f"messages不能为空: {messages}")
            
            # 检查每条消息的格式
            for i, message in enumerate(messages):
                if not isinstance(message, dict):
                    raise ValueError(f"messages[{i}]必须是字典: {message}")
                
                if "role" not in message or "content" not in message:
                    raise ValueError(f"messages[{i}]缺少role或content字段: {message}")
                
                if message["role"] not in ["user", "assistant", "system"]:
                    raise ValueError(f"messages[{i}]的role必须是user/assistant/system: {message['role']}")
            
            # 检查消息顺序
            last_role = None
            for i, message in enumerate(messages):
                role = message["role"]
                if role == "user":
                    if last_role == "user":
                        raise ValueError(f"user消息不能紧跟user消息 (位置{i}): {messages}")
                    last_role = "user"
                elif role == "assistant":
                    if last_role == "assistant":
                        raise ValueError(f"assistant消息不能紧跟assistant消息 (位置{i}): {messages}")
                    if last_role != "user":
                        raise ValueError(f"assistant消息前面必须是user消息 (位置{i}): {messages}")
                    last_role = "assistant"
                elif role == "system":
                    if i != 0:
                        raise ValueError(f"system消息只能出现在第一个位置 (位置{i}): {messages}")
            
            # 确保最后一轮是assistant消息
            if messages[-1]["role"] != "assistant":
                raise ValueError(f"最后一轮必须是assistant消息: {messages}")

        

        
        def format_messages_func(examples):
            '''
            len(examples) = batch_size
            '''
            texts = []
            labels = []
            
            for messages in examples[messages_field]:

                validate_messages_format(messages)
                try:
                    # 为GPT-2等传统语言模型构建训练数据
                    # 1. 构建输入（对话历史，不包含最后一轮回复）
                    # 2. 构建完整序列（包含最后一轮回复）
                    # 3. 使用掩码只对回复部分计算损失
                    
                    # 提取对话历史（除了最后一轮assistant回复）
                    history = messages[:-1] if messages and messages[-1]["role"] == "assistant" else messages
                    
                    # 手动构建对话格式（不依赖chat template）
                    input_parts = []
                    for msg in history:
                        if msg["role"] == "system":
                            input_parts.append(f"System: {msg['content']}")
                        elif msg["role"] == "user":
                            input_parts.append(f"User: {msg['content']}")
                    
                    # 添加生成提示符
                    input_text = "\n".join(input_parts) + "\nAssistant:"
                    texts.append(input_text)
                    
                    # 构建完整序列（包含最后一轮回复）
                    full_parts = input_parts.copy()
                    if messages and messages[-1]["role"] == "assistant":
                        full_parts.append(f"Assistant: {messages[-1]['content']}")
                    
                    full_text = "\n".join(full_parts)
                    labels.append(full_text)
                        
                except Exception as e:
                    self.logger.warning(f"处理messages时出错: {e}, 使用默认值")
                    texts.append("")
                    labels.append("")
            
            return {
                "input": texts,
                "label": labels
            }
        
        # 移除messages字段，保留其他字段
        columns_to_remove = [col for col in dataset.column_names if col == messages_field]
        
        dataset = dataset.map(
            format_messages_func,
            batched=True,
            remove_columns=columns_to_remove,
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
            texts = examples.get("input", [])
            labels = examples.get("label", [])
            
            # 对输入文本进行分词
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_seq_length,
                return_tensors=None
            )
            
            # 对完整序列进行分词（用于损失计算）
            if labels:
                label_tokenized = self.tokenizer(
                    labels,
                    truncation=True,
                    padding="max_length", 
                    max_length=max_seq_length,
                    return_tensors=None
                )
                
                # 创建掩码labels：只对completion部分计算损失
                input_ids_list = tokenized["input_ids"]
                label_ids_list = label_tokenized["input_ids"]
                
                masked_labels = []
                for input_ids, label_ids in zip(input_ids_list, label_ids_list):
                    masked_label = label_ids.copy()
                    
                    # 找到input_ids在label_ids中的位置
                    # 使用滑动窗口匹配，确保完全匹配
                    input_length = 0
                    label_length = len(label_ids)
                    input_len = len(input_ids)
                    
                    # 找到input_ids在label_ids中的匹配位置
                    match_pos = -1
                    for start in range(label_length - input_len + 1):
                        if label_ids[start:start + input_len] == input_ids:
                            match_pos = start
                            break
                    
                    # 如果找到匹配，将input部分设为-100
                    if match_pos != -1:
                        for i in range(input_len):
                            if input_ids[i] != self.tokenizer.pad_token_id:  # 只掩码非pad token
                                masked_label[match_pos + i] = -100
                    
                    masked_labels.append(masked_label)
                
                tokenized["labels"] = masked_labels
            else:
                # 如果没有label，使用与input_ids相同的长度
                tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,   # 把原来的列名， input、label 都移除
            desc="对数据集进行分词"
        )
        
        self.logger.info(f"分词完成，数据集包含 {len(tokenized_dataset)} 个样本")
        return tokenized_dataset

    def create_training_args(
        self,
        output_dir: str,
        **training_kwargs
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
            eval_strategy: 评估策略
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
            fp16=self.config.use_fp16 and torch.cuda.is_available(),
            bf16=self.config.use_bf16 and torch.cuda.is_bf16_supported(),
            dataloader_num_workers=self.config.dataloader_num_workers,
        )
        
        self.logger.info(f"训练参数创建完成: {self.config.num_train_epochs} epochs, train_batch_size={self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps}")
        return training_args

    def start_training(
        self,
        dataset: Union[Dataset, List[Dict]],
        output_dir: str,
        messages_field: str = "messages",
        eval_dataset: Optional[Union[Dataset, List[Dict]]] = None,
        **training_kwargs
    ):
        '''
        开始训练
        
        Args:
            dataset: 训练数据集
            output_dir: 输出目录
            messages_field: messages字段名
            eval_dataset: 评估数据集
            **training_kwargs: 其他训练参数
        '''
        # 加载模型和分词器
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
        
        # 准备数据集
        prepared_dataset = self.prepare_dataset(dataset, messages_field)
        tokenized_dataset = self.tokenize_dataset(prepared_dataset)
        
        # 准备评估数据集
        eval_tokenized_dataset = None
        if eval_dataset is not None:
            prepared_eval_dataset = self.prepare_dataset(eval_dataset, messages_field)
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
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                """
                自定义损失计算函数 - 适用于SFT训练的因果语言模型
                
                训练目标：基于对话历史生成下一轮回复
                
                数据格式：
                - input_ids: 对话历史 + generation prompt的分词结果 [batch_size, max_seq_length]
                - labels: 完整对话序列的分词结果，但input部分被-100掩码 [batch_size, max_seq_length]
                
                训练原理：
                在SFT训练中，只对completion（回复）部分计算损失，prompt部分应该被忽略
                使用ignore_index=-100来掩码掉不需要计算损失的token
                
                损失计算步骤：
                1. 取出labels并获取模型输出 logits [batch_size, max_seq_length, vocab_size]
                2. labels中已经包含了掩码：input部分为-100，completion部分为真实token
                3. 使用ignore_index=-100的交叉熵损失，自动忽略掩码部分
                
                关键原理：
                - 模型预测完整序列，但只对completion部分计算损失
                - 这是标准的SFT训练方式，符合指令微调的实践
                """
                labels = inputs.pop("labels")
                
                # 获取模型输出
                outputs = model(**inputs)
                logits = outputs.logits  # [batch_size, max_seq_length, vocab_size]
                
                # 使用标准的交叉熵损失，ignore_index=-100自动忽略掩码部分
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                
                # 展平张量以计算损失
                # logits: (batch_size * max_seq_length, vocab_size)
                # labels: (batch_size * max_seq_length,) ，其中-100为掩码
                loss = loss_fct(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                
                return (loss, outputs) if return_outputs else loss
            
            def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
                """重写evaluate方法，添加调试信息"""
                print("🔍 开始执行评估...")
                result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
                print(f"🔍 评估完成，结果: {result}")
                return result
            
            def training_step(self, model, inputs, num_items_in_batch=None):
                """
                重写training_step，使用独立的评估计数器避免频繁评估
                """
                # 执行训练步骤
                step_output = super().training_step(model, inputs, num_items_in_batch)

                # 使用独立的评估计数器，每隔eval_steps才评估一次
                # 避免与logging_steps冲突导致过于频繁的评估
                if not hasattr(self, '_last_eval_step'):
                    self._last_eval_step = 0

                current_step = self.state.global_step
                eval_interval = getattr(self.args, 'eval_steps', None) or self.args.logging_steps

                # 只在达到评估间隔且距离上次评估足够远时才评估
                if (current_step - self._last_eval_step) >= eval_interval:
                    if current_step == 1 or current_step % self.args.eval_steps == 0:
                        print(f"🔍 在训练步 {current_step} 后执行评估...")
                        eval_result = self.evaluate()
                        if hasattr(self, 'compute_metrics') and self.compute_metrics:
                            try:
                                # 获取最新的评估预测
                                eval_preds = self.predict(self.eval_dataset)
                                # 正确地传递预测结果和标签给compute_metrics
                                metrics = self.compute_metrics((eval_preds.predictions, eval_preds.label_ids))
                                self.log(metrics)
                                print(f"🔍 自定义评估指标: {metrics}")
                            except Exception as e:
                                print(f"🔍 计算自定义指标时出错: {e}")
                        self._last_eval_step = current_step

                return step_output
        
        math_evaluator = MathEvaluator()
        # 创建训练器
        self.trainer: Trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=eval_tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda eval_preds: math_evaluator.compute_math_accuracy(eval_preds, self.tokenizer)
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
            self.tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="right"
            )
            
            # 设置特殊标记
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            model_kwargs = {
                "trust_remote_code": self.trust_remote_code,
                "torch_dtype": torch.float16 if torch.cuda.is_available() and self.use_fp16 else torch.float32,
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
        
        # 获取输入长度，用于后续提取生成的内容
        input_length = inputs['input_ids'].shape[1]
        
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
                **kwargs
            )
        
        # 解码并提取生成的文本（从输入长度之后开始）
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
        
        return generated_text
