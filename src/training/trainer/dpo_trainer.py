from dataclasses import dataclass  
from typing import Optional, Dict , List, Tuple, Callable
import os  
import torch  
import torch.nn.functional as F  
from torch.utils.data import Dataset  
from transformers import (  
    TrainingArguments,  
    AutoTokenizer,  
    # Qwen2ForCausalLM,  
    BitsAndBytesConfig,
    Trainer,  
    TrainerCallback  
)  

from model.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model  
from datasets import load_dataset  
# from trl import DPOTrainer  
# import deepspeed  
from deepspeed import DeepSpeedEngine 

from config.config import MODEL_PATH


class CustomDPODataset(Dataset):  
    def __init__(self, tokenized_data):  
        self.data = tokenized_data  
        
    def __len__(self):  
        return len(self.data["input_ids"])  
    
    def __getitem__(self, idx):  
        return {  
            "input_ids": self.data["input_ids"][idx],  
            "attention_mask": self.data["attention_mask"][idx],  
            "chosen_labels": self.data["chosen_labels"][idx],  
            "rejected_labels": self.data["rejected_labels"][idx]  
        }  

class CustomDPOTrainer:  
    def __init__(  
        self,  
        output_dir: str,  
        dataset_name_or_path: str,  
        model_name: str = MODEL_PATH,  
        is_ds: bool = True,  
        ds_config_path: Optional[str] = None,  
        is_peft: bool = True,  
        peft_config: Optional[LoraConfig] = None,  
        is_quantized: bool = False,  
        bnb_config: Optional[BitsAndBytesConfig] = None,  
        max_seq_length: int = 1024,  
        beta: float = 0.1  
    ):  
        self.output_dir = output_dir  
        self.dataset_name_or_path = dataset_name_or_path  
        self.beta = beta  
        self.max_seq_length = max_seq_length  

        # 初始化模型和tokenizer  
        self.model, self.tokenizer = self._init_model_and_tokenizer(  
            model_name, is_quantized, bnb_config  
        )  
        
        # 应用LoRA  
        if is_peft:  
            self.peft_config = peft_config or self._default_lora_config()  
            self.model = get_peft_model(self.model, self.peft_config)  

        # 准备数据集  
        self.dataset = self._prepare_dataset()  

        # 配置训练参数  
        self.training_args = TrainingArguments(  
            output_dir=output_dir,  
            deepspeed=ds_config_path if is_ds else None,  
            per_device_train_batch_size=4,  
            gradient_accumulation_steps=2,  
            learning_rate=2e-5,  
            bf16=True,  
            logging_steps=10,  
            save_steps=500,  
            remove_unused_columns=False,  
            optim="adamw_torch",  
            max_grad_norm=0.3,  
            num_train_epochs=3  
        )  

        # 初始化自定义Trainer  
        self.trainer = Trainer(  
            model=self.model,  
            args=self.training_args,  
            train_dataset=self.dataset,  
            data_collator=self.dpo_collator,  
            compute_metrics=self._compute_metrics,  
            callbacks=[DPOCallback()]  
        )  

    def _init_model_and_tokenizer(self, model_name, is_quantized, bnb_config):  
        """初始化模型和分词器"""  
        bnb_config = bnb_config or BitsAndBytesConfig(  
            load_in_4bit=True,  
            bnb_4bit_quant_type="nf4",  
            bnb_4bit_compute_dtype=torch.bfloat16,  
            bnb_4bit_use_double_quant=True,  
        ) if is_quantized else None  

        tokenizer = AutoTokenizer.from_pretrained(model_name)  
        tokenizer.pad_token = tokenizer.eos_token  

        model = Qwen2ForCausalLM.from_pretrained(  
            model_name,  
            quantization_config=bnb_config,  
            device_map="auto",  
            trust_remote_code=True  
        )  
        return model, tokenizer  

    def _default_lora_config(self):  
        """默认LoRA配置"""  
        return LoraConfig(  
            r=64,  
            lora_alpha=16,  
            lora_dropout=0.05,  
            target_modules=["q_proj", "v_proj"],  
            bias="none",  
            task_type="CAUSAL_LM"  
        )  

    def _prepare_dataset(self):  
        """数据预处理管道"""  
        dataset = load_dataset(self.dataset_name_or_path, split="train")  
        dataset = dataset.filter(self._data_filter)  
        
        # 并行化tokenize处理  
        tokenized_data = dataset.map(  
            self._tokenize_function,  
            batched=True,  
            num_proc=4,  
            remove_columns=dataset.column_names  
        )  
        
        return CustomDPODataset(tokenized_data)  

    def _data_filter(self, sample):  
        """数据过滤逻辑"""  
        return all([sample["prompt"], sample["chosen"], sample["rejected"]]) and \
               len(sample["prompt"]) <= 512 and \
               len(sample["chosen"]) <= 1024 and \
               len(sample["rejected"]) <= 1024  

    def _tokenize_function(self, samples):  
        """DPO专用tokenize处理"""  
        batch = {"input_ids": [], "attention_mask": [],   
                "chosen_labels": [], "rejected_labels": []}  
        
        for prompt, chosen, rejected in zip(samples["prompt"], samples["chosen"], samples["rejected"]):  
            # 生成prompt模板  
            full_prompt = f"Instruction: {prompt}\nResponse: "  
            
            # Tokenize chosen响应  
            chosen_tokens = self.tokenizer(  
                full_prompt + chosen,  
                max_length=self.max_seq_length,  
                padding="max_length",  
                truncation=True,  
                return_tensors="pt"  
            )  
            
            # Tokenize rejected响应  
            rejected_tokens = self.tokenizer(  
                full_prompt + rejected,  
                max_length=self.max_seq_length,  
                padding="max_length",  
                truncation=True,  
                return_tensors="pt"  
            )  
            
            batch["input_ids"].append(chosen_tokens["input_ids"][0])  
            batch["attention_mask"].append(chosen_tokens["attention_mask"][0])  
            batch["chosen_labels"].append(chosen_tokens["input_ids"][0])  
            batch["rejected_labels"].append(rejected_tokens["input_ids"][0])  
            
        return batch  

    def dpo_collator(self, features):  
        """自定义数据整理函数"""  
        batch = {  
            "input_ids": torch.stack([f["input_ids"] for f in features]),  
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),  
            "chosen_labels": torch.stack([f["chosen_labels"] for f in features]),  
            "rejected_labels": torch.stack([f["rejected_labels"] for f in features])  
        }  
        return batch  

    def _compute_metrics(self, eval_pred):  
        """自定义评估指标"""  
        logits_chosen, logits_rejected = eval_pred.predictions  
        accuracy = (logits_chosen > logits_rejected).mean()  
        return {"dpo_accuracy": accuracy}  

    def train(self):  
        """启动训练流程"""  
        self.trainer.train()  

    def save_model(self):  
        """模型保存逻辑"""  
        save_path = os.path.join(self.output_dir, "qwen2_cmed_dpo")  
        self.trainer.save_model(save_path)  
        self.tokenizer.save_pretrained(save_path)  

class DPOCallback(TrainerCallback):  
    def on_train_begin(self, args, state, control, **kwargs):  
        """训练开始前的初始化"""  
        self.model = kwargs.pop("model")  
        if isinstance(self.model, DeepSpeedEngine):  
            self.model = self.model.module  

    def on_step_begin(self, args, state, control, **kwargs):  
        """梯度累积期间冻结参考模型"""  
        if state.global_step == 0:  
            # 初始化参考模型参数  
            self.ref_model = self._clone_model(self.model)  
            self.ref_model.requires_grad_(False)  

    def _clone_model(self, model):  
        """
        创建参考模型的深拷贝
        
        
        type(model)：获取模型的类类型（如 Qwen2ForCausalLM）
        **model.config.to_dict()：将模型的配置转换为字典并解包为关键字参数
        type(model)(**model.config.to_dict())：使用原始模型的配置创建一个新的模型实例
        """  
        return type(model)(**model.config.to_dict()).load_state_dict(model.state_dict())  

    def compute_loss(self, model, inputs, return_outputs=False):  
        """核心DPO损失计算"""  
        # 前向传播获取logits  
        outputs = model(  
            input_ids=inputs["input_ids"],  
            attention_mask=inputs["attention_mask"]  
        )  
        
        # 获取chosen和rejected的log概率  
        chosen_log_probs = self._get_log_probs(outputs.logits, inputs["chosen_labels"])  
        rejected_log_probs = self._get_log_probs(outputs.logits, inputs["rejected_labels"])  
        
        # 计算参考模型的log概率  
        with torch.no_grad():  
            ref_outputs = self.ref_model(  
                input_ids=inputs["input_ids"],  
                attention_mask=inputs["attention_mask"]  
            )  
            ref_chosen_log_probs = self._get_log_probs(ref_outputs.logits, inputs["chosen_labels"])  # 计算 log(π(y|x))
            ref_rejected_log_probs = self._get_log_probs(ref_outputs.logits, inputs["rejected_labels"])  
        
        # 计算DPO损失  L = -log(σ(r(x,y_w) - r(x,y_l))) , where r(x,y) = β * log(π(y|x)/π_ref(y|x)) = β * [log(π(y|x)) - log(π_ref(y|x))]
        losses = -F.logsigmoid(  
            self.beta * (  
                (chosen_log_probs - ref_chosen_log_probs) -     # log(π(y_win|x)/π_ref(y_win|x))
                (rejected_log_probs - ref_rejected_log_probs)  
            )  
        )  
        
        return losses.mean()  

    def _get_log_probs(self, logits, labels):  
        """
        计算每个token的对数概率
        
        ##Args:
        logits: x  shape = (batch_size, seq_len, vocab_size)
        labels: y  shape = (batch_size, seq_len)
        
        计算 log(π(y|x))
        
        为了以后计算 reward r(x,y) = β * log(π(y|x)/π_ref(y|x)) 
        
        
        ##Return
        返回值：(batch_size, seq_len)
            每个位置的值表示对应token的对数概率
                
        """  
        log_probs = F.log_softmax(logits, dim=-1)  
        # 在 log_probs 的最后一个维度（vocab_size维度）上，根据 labels 的索引收集对应的对数概率
        return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)  
    
    
    
        '''
        log_probs = torch.tensor([
            [[-0.5, -1.0, -2.0],  # 第一个样本，第一个token的对数概率
            [-0.8, -1.2, -1.5]], # 第一个样本，第二个token的对数概率
            [[-0.6, -1.1, -2.1],  # 第二个样本，第一个token的对数概率
            [-0.9, -1.3, -1.6]]  # 第二个样本，第二个token的对数概率
        ])  # shape: (2, 2, 3)  (batch_size=2, seq_len=2, vocab_size=3)

        labels = torch.tensor([
            [0, 2],  # 第一个样本的标签
            [1, 0]   # 第二个样本的标签
        ])  # shape: (2, 2)  (batch_size=2, seq_len=2)
        
        
        执行过程：
        labels.unsqueeze(-1)：

        在 labels 的最后一个维度上增加一个维度
        结果：

        python
        Apply
        tensor([
            [[0], [2]],  # 第一个样本
            [[1], [0]]   # 第二个样本
        ])  # shape: (2, 2, 1)
        torch.gather(log_probs, -1, labels.unsqueeze(-1))：

        在 log_probs 的最后一个维度（vocab_size维度）上，根据 labels 的索引收集对应的对数概率
        结果：

        python
        Apply
        tensor([
            [[-0.5], [-1.5]],  # 第一个样本
            [[-1.1], [-0.9]]   # 第二个样本
        ])  # shape: (2, 2, 1)
        .squeeze(-1)：

        移除最后一个维度
        最终结果：

        python
        Apply
        tensor([
            [-0.5, -1.5],  # 第一个样本
            [-1.1, -0.9]   # 第二个样本
        ])  # shape: (2, 2)
        解释：
        对于第一个样本的第一个token，labels[0,0]=0，所以收集 log_probs[0,0,0]=-0.5
        对于第一个样本的第二个token，labels[0,1]=2，所以收集 log_probs[0,1,2]=-1.5
        对于第二个样本的第一个token，labels[1,0]=1，所以收集 log_probs[1,0,1]=-1.1
        对于第二个样本的第二个token，labels[1,1]=0，所以收集 log_probs[1,1,0]=-0.9
        
        最终得到的矩阵表示每个样本中每个token对应的对数概率值。
        '''
    