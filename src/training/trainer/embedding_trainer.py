import os
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any, Tuple
from datasets import Dataset
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    AutoModel,
    Trainer,
    DataCollatorWithPadding,
)
from transformers import BitsAndBytesConfig, BatchEncoding
from sentence_transformers import SentenceTransformer

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate import FullyShardedDataParallelPlugin
from accelerate import DeepSpeedPlugin
from utils.logger import setup_logger
from config.settings import settings
from training.trainer.base_trainer import BaseTrainer


class SentencePairDataCollator:
    """自定义数据收集器，处理句子对格式的数据"""
    
    def __init__(self, tokenizer, padding=True, truncation=True, max_length=None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
    
    def __call__(self, features):
        batch = {}
        
        batch["input_ids_1"] = torch.tensor([f["input_ids_1"] for f in features], dtype=torch.long)
        batch["attention_mask_1"] = torch.tensor([f["attention_mask_1"] for f in features], dtype=torch.long)
        batch["input_ids_2"] = torch.tensor([f["input_ids_2"] for f in features], dtype=torch.long)
        batch["attention_mask_2"] = torch.tensor([f["attention_mask_2"] for f in features], dtype=torch.long)
        
        if "label" in features[0]:
            batch["label"] = torch.tensor([f["label"] for f in features], dtype=torch.float)
        
        return batch


class EmbeddingTrainer(BaseTrainer):
    def __init__(self, config: EmbeddingTrainingConfig):
        super().__init__(config)
        self.loss_type = getattr(config, 'loss_type', 'cosine')
        self.normalized = getattr(config, 'normalized', True)
        self.temperature = getattr(config, 'temperature', 0.02)
        self.num_classes = getattr(config, 'num_classes', None)
        self.embedding_dim = getattr(config, 'embedding_dim', 768)
        
        self.distributed_strategy = getattr(config, 'distributed_strategy', 'auto')
        self.find_unused_parameters = getattr(config, 'find_unused_parameters', False)
        self.main_process_ip = getattr(config, 'main_process_ip', 'localhost')
        self.main_process_port = getattr(config, 'main_process_port', 29500)
        self.num_machines = getattr(config, 'num_machines', 1)
        self.machine_rank = getattr(config, 'machine_rank', 0)
        
        if self.loss_type == 'softmax' and self.num_classes is not None:
            self.class_projection = torch.nn.Linear(self.embedding_dim, self.num_classes)
            self.logger.info(f"初始化 Softmax 分类头，类别数: {self.num_classes}")

    def _setup_accelerator(self):
        report_to = getattr(self.config, 'report_to', 'none')
        
        if self.distributed_strategy == "fsdp":
            fsdp_plugin = FullyShardedDataParallelPlugin()
            self.accelerator = Accelerator(
                fsdp_plugin=fsdp_plugin,
                log_with=report_to if report_to != "none" else None
            )
        elif self.distributed_strategy == "deepspeed":
            deepspeed_plugin = DeepSpeedPlugin(
                zero_stage=2,
                gradient_accumulation_steps=self.gradient_accumulation_steps
            )
            self.accelerator = Accelerator(
                deepspeed_plugin=deepspeed_plugin,
                log_with=report_to if report_to != "none" else None
            )
        elif self.distributed_strategy in ["ddp", "auto"]:
            ddp_kwargs = DistributedDataParallelKwargs(
                find_unused_parameters=self.find_unused_parameters
            )
            self.accelerator = Accelerator(
                kwargs_handlers=[ddp_kwargs],
                log_with=report_to if report_to != "none" else None
            )
        else:
            self.accelerator = Accelerator(
                log_with=report_to if report_to != "none" else None
            )
        
        self.is_distributed = self.accelerator.num_processes > 1
        self.logger.info(f"Accelerator 初始化完成，分布式训练: {self.is_distributed}, "
                        f"进程数: {self.accelerator.num_processes}, "
                        f"策略: {self.distributed_strategy}")

    def load_model_and_tokenizer(self):
        try:
            self.logger.info(f"{self.__class__.__name__}: 加载 embedding 模型: {self.model_name}")
            
            self._setup_accelerator()
            
            self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="right"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            model_kwargs = {
                "trust_remote_code": self.trust_remote_code,
                "torch_dtype": torch.float16 if self.use_fp16 and torch.cuda.is_available() else torch.float32,
            }
            
            if self.is_distributed and self.distributed_strategy in ["ddp", "auto"]:
                if self.device_map == "auto":
                    model_kwargs["device_map"] = None
            else:
                model_kwargs["device_map"] = self.device_map if torch.cuda.is_available() else None
            
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
            
            self.model = AutoModel.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            if not torch.cuda.is_available():
                self.model = self.model.to("cpu")
            
            if self.use_gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
                self.logger.info("启用梯度检查点")
            
            self.model = self.accelerator.prepare(self.model)
            
            if self.loss_type == 'softmax' and self.num_classes is not None and hasattr(self, 'class_projection'):
                self.class_projection = self.accelerator.prepare(self.class_projection)
            
            self.logger.info("Embedding 模型和分词器加载完成")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise

    def prepare_dataset(
        self,
        dataset: Union[Dataset, List[Dict]],
        sentence1_field: str = "sentence1",
        sentence2_field: str = "sentence2",
        label_field: str = "label",
        max_seq_length: Optional[int] = None
    ) -> Dataset:
        if isinstance(dataset, list):
            dataset = Dataset.from_list(dataset)
        
        max_seq_length = max_seq_length or self.max_seq_length
        
        def format_pairs_func(examples):
            sentence1s = examples.get(sentence1_field, [])
            sentence2s = examples.get(sentence2_field, [])
            labels = examples.get(label_field, [0.0] * len(sentence1s))
            
            return {
                "sentence1": sentence1s,
                "sentence2": sentence2s,
                "label": labels if isinstance(labels, list) else [labels] * len(sentence1s)
            }
        
        columns_to_remove = [col for col in dataset.column_names if col in [sentence1_field, sentence2_field, label_field]]
        
        dataset = dataset.map(
            format_pairs_func,
            batched=True,
            remove_columns=columns_to_remove,
        )
        
        self.logger.info(f"句子对数据集准备完成，包含 {len(dataset)} 个样本")
        return dataset

    def tokenize_dataset(
        self,
        dataset: Dataset,
        max_seq_length: Optional[int] = None
    ) -> Dataset:
        max_seq_length = max_seq_length or self.max_seq_length
        
        def tokenize_function(examples):
            sentence1s = examples.get("sentence1", [])
            sentence2s = examples.get("sentence2", [])
            
            tokenized1 = self.tokenizer(
                sentence1s,
                truncation=True,
                padding="max_length",
                max_length=max_seq_length,
                return_tensors=None
            )
            
            tokenized2 = self.tokenizer(
                sentence2s,
                truncation=True,
                padding="max_length",
                max_length=max_seq_length,
                return_tensors=None
            )
            
            return {
                "input_ids_1": tokenized1["input_ids"],
                "attention_mask_1": tokenized1["attention_mask"],
                "input_ids_2": tokenized2["input_ids"],
                "attention_mask_2": tokenized2["attention_mask"],
                "label": examples.get("label", [0.0] * len(sentence1s))
            }
        
        columns_to_remove = [col for col in dataset.column_names]
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=columns_to_remove,
            desc="分词句子对数据集"
        )
        
        self.logger.info(f"分词完成，数据集包含 {len(tokenized_dataset)} 个样本")
        return tokenized_dataset

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids_1 = inputs.pop("input_ids_1")
        attention_mask_1 = inputs.pop("attention_mask_1")
        input_ids_2 = inputs.pop("input_ids_2")
        attention_mask_2 = inputs.pop("attention_mask_2")
        labels = inputs.pop("label", None)
        
        num_classes = getattr(self, 'num_classes', None)
        
        outputs1 = model(input_ids=input_ids_1, attention_mask=attention_mask_1)
        outputs2 = model(input_ids=input_ids_2, attention_mask=attention_mask_2)
        
        embeddings1 = self.mean_pooling(outputs1, attention_mask_1)
        embeddings2 = self.mean_pooling(outputs2, attention_mask_2)
        
        if self.normalized:
            embeddings1 = F.normalize(embeddings1, p=2, dim=1)
            embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        
        if self.loss_type == 'cosine':
            cosine_scores = torch.sum(embeddings1 * embeddings2, dim=1)
            loss = F.mse_loss(cosine_scores, torch.tensor(labels, device=cosine_scores.device, dtype=cosine_scores.dtype))
        elif self.loss_type == 'contrastive':
            similarity_matrix = torch.matmul(embeddings1, embeddings2.T) / self.temperature
            target = torch.arange(len(similarity_matrix), device=similarity_matrix.device)
            loss = F.cross_entropy(similarity_matrix, target)
        elif self.loss_type == 'info_nce':
            similarity_matrix = torch.matmul(embeddings1, embeddings2.T) / self.temperature
            target = torch.arange(len(similarity_matrix), device=similarity_matrix.device)
            loss = F.cross_entropy(similarity_matrix, target)
        elif self.loss_type == 'softmax':
            if num_classes is None:
                raise ValueError("SoftmaxLoss requires num_classes to be set in config")
            logits = torch.matmul(embeddings1, self.class_weight) if hasattr(self, 'class_weight') else embeddings1 @ self.class_projection.weight
            loss = F.cross_entropy(logits, torch.tensor(labels, device=logits.device, dtype=torch.long))
        elif self.loss_type == 'mnr':
            similarity_matrix = torch.matmul(embeddings1, embeddings2.T) / self.temperature
            target = torch.arange(len(similarity_matrix), device=similarity_matrix.device)
            loss = F.cross_entropy(similarity_matrix, target)
        else:
            cosine_scores = torch.sum(embeddings1 * embeddings2, dim=1)
            loss = F.mse_loss(cosine_scores, torch.tensor(labels, device=cosine_scores.device, dtype=cosine_scores.dtype))
        
        if return_outputs:
            return loss, {
                'embeddings1': embeddings1,
                'embeddings2': embeddings2,
                'loss': loss
            }
        return loss

    def start_training(
        self,
        dataset: Union[Dataset, List[Dict]],
        output_dir: str,
        sentence1_field: str = "sentence1",
        sentence2_field: str = "sentence2",
        label_field: str = "label",
        eval_dataset: Optional[Union[Dataset, List[Dict]]] = None,
        **training_kwargs
    ):
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
        
        prepared_dataset = self.prepare_dataset(dataset, sentence1_field, sentence2_field, label_field)
        tokenized_dataset = self.tokenize_dataset(prepared_dataset)
        
        eval_tokenized_dataset = None
        if eval_dataset is not None:
            prepared_eval_dataset = self.prepare_dataset(eval_dataset, sentence1_field, sentence2_field, label_field)
            eval_tokenized_dataset = self.tokenize_dataset(prepared_eval_dataset)
        
        training_args = self.create_training_args(output_dir, **training_kwargs)
        
        if eval_tokenized_dataset is not None:
            training_args.do_eval = True
            training_args.evaluation_strategy = "steps"
            training_args.eval_steps = training_kwargs.get("eval_steps", training_args.save_steps // 2)
        
        if self.is_distributed:
            training_args.ddp_find_unused_parameters = self.find_unused_parameters
            training_args.ddp_broadcast_buffers = False
            if self.distributed_strategy == "fsdp":
                training_args.fsdp = "full_shard"
            self.logger.info(f"分布式训练配置: find_unused_parameters={self.find_unused_parameters}, "
                           f"strategy={self.distributed_strategy}")
        
        embedding_trainer = self
        
        class CustomEmbeddingTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                return embedding_trainer.compute_loss(model, inputs, return_outputs)
        
        data_collator = SentencePairDataCollator(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=self.max_seq_length
        )
        
        trainer = CustomEmbeddingTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=eval_tokenized_dataset,
            data_collator=data_collator,
        )
        
        trainer = self.accelerator.prepare(trainer)
        
        self.trainer = trainer
        
        self.logger.info("开始训练 Embedding 模型...")
        train_result = trainer.train()
        
        if self.accelerator.is_main_process:
            self.logger.info(f"训练完成，保存模型到 {output_dir}")
            trainer.save_model(output_dir)
        
        metrics = train_result.metrics
        self.logger.info(f"训练指标: {metrics}")
        
        return metrics
