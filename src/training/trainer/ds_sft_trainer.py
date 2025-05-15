import torch
import torch.nn as nn
import os

from datasets import load_dataset  
from transformers import (  
    AutoModelForCausalLM,  
    AutoTokenizer,  
    TrainingArguments,  
    BitsAndBytesConfig,
    set_seed
)  

from model.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from peft import LoraConfig  
from trl import SFTTrainer  


from accelerate import Accelerator  


from config.config import MODEL_PATH, TOKENIZER_PATH, DEVICE, DEEPSPEED_CONFIG_PATH
# 初始化accelerator（需在TrainingArguments之前）  
# accelerator = Accelerator()  



# # 修改训练参数  
# training_args = TrainingArguments(  
#     output_dir="./results",  
#     num_train_epochs=3,  
#     per_device_train_batch_size=4,  # 单卡batch_size  
#     gradient_accumulation_steps=2,  # 梯度累积次数  
#     learning_rate=2e-5,  
#     weight_decay=0.01,  
#     warmup_ratio=0.1,  
#     logging_steps=10,  
#     save_strategy="steps",  
#     save_steps=500,  
#     deepspeed="./ds_config.json",  # DeepSpeed配置文件路径  
#     gradient_checkpointing=True,  # 激活梯度检查点  
#     fp16=False,  # 与DeepSpeed的bf16配置互斥  
#     bf16=False,   # 由DeepSpeed配置文件控制  
#     optim="adamw_hf",  
#     report_to="tensorboard",  
#     ddp_find_unused_parameters=False,  
#     seed=42  
# )  




# # 保持其他部分不变，修改SFTTrainer初始化  
# trainer = SFTTrainer(  
#     model=model,  
#     args=training_args,  
#     train_dataset=dataset,  
#     peft_config=peft_config,  
#     max_seq_length=1024,  
#     tokenizer=tokenizer,  
#     formatting_func=format_instruction,  
#     dataset_text_field="text",  
#     packing=True  # 启用序列打包提升效率  
# )  

# # 分布式训练准备  
# model = accelerator.prepare_model(model)  
# train_dataloader = accelerator.prepare_data_loader(trainer.get_train_dataloader())  

# # 开始训练  
# trainer.train()  

# # 仅主进程保存模型  
# if accelerator.is_main_process:  
#     trainer.save_model("qwen2_tcm_deepspeed")  
    
    
    



class DeepSpeedSftTrainer:
    def __init__(
            self, 
            output_dir:str, 
            dataset_name_or_path:str,
            is_ds:bool = True, 
            ds_config_path:str=None, 
            is_peft=True, 
            peft_config = None, 
            is_quantized:bool=False, 
            bnb_config = None,
            max_seq_length:int=1024,
        ):
        
        
        self.output_dir = output_dir
        self.datset_name_or_path = dataset_name_or_path
        self.ds_config_path = ds_config_path
        self.is_peft = is_peft
        self.is_quantized = is_quantized
        self.is_ds = is_ds
        self.max_seq_length = max_seq_length
        
        # self.accelerator = Accelerator()

        
        
        if is_quantized:
            if bnb_config is None:
                self.bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,           # 4位量化加载  
                    bnb_4bit_quant_type="nf4",  # 量化类型  
                    bnb_4bit_compute_dtype=torch.bfloat16,  
                    bnb_4bit_use_double_quant=True  # 嵌套量化  
                )
            else:
                self.bnb_config = bnb_config
                
                
                
        if is_ds:
            if ds_config_path is None:
                self.ds_config_path = DEEPSPEED_CONFIG_PATH
            else:
                self.ds_config_path = ds_config_path
                
        if is_peft:
            if peft_config is None:
                self.peft_config = LoraConfig(
                    r=64,                # 低秩矩阵维度  
                    lora_alpha=16,       # 缩放系数  
                    lora_dropout=0.05,   # Dropout概率  
                    target_modules=["q_proj", "v_proj"],  # 目标注意力层  
                    bias="none",         # 不训练偏置项  
                    task_type="CAUSAL_LM"  
                )
            else:
                self.peft_config = peft_config
        
        self.model = Qwen2ForCausalLM.from_pretrained(  
            MODEL_PATH,  
            quantization_config=self.bnb_config if is_quantized else None,
            device_map="auto",  
            trust_remote_code=True  
        )  
        
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)  
        self.tokenizer.pad_token = self.tokenizer.eos_token  # 设置填充token 
        
        '''
        自动集成机制：
            当TrainingArguments中设置deepspeed参数后，Trainer会自动调用accelerator = Accelerator(deepspeed_plugin=...)1
            SFTTrainer继承自Trainer，共享相同的分布式初始化逻辑
        冲突风险：
            手动prepare会覆盖DeepSpeed的自动配置
            可能导致ZeRO stage参数不生效
        '''
        ds_kwargs = {
            "deepspeed": self.ds_config_path,  # DeepSpeed配置文件路径  
            "gradient_checkpointing": True,  # 激活梯度检查点  
            "fp16":False,  # 与DeepSpeed的bf16配置互斥  
            "bf16":False,   # 由DeepSpeed配置文件控制  
            "ddp_find_unused_parameters": False,  
            }
        
        trainer_kwargs = {
            "output_dir": self.output_dir,  
            "num_train_epochs":3,  
            "per_device_train_batch_size":4,  # 单卡batch_size  
            "gradient_accumulation_steps":2,  # 梯度累积次数  
            "learning_rate":2e-5,  
            "weight_decay":0.01,  
            "warmup_ratio":0.1,  
            "logging_steps":10,  
            "save_strategy":"steps",  
            "save_steps":500,  
            "gradient_checkpointing":True,  # 激活梯度检查点  
            "optim":"adamw_hf",  
            "report_to":"tensorboard",  
            "ddp_find_unused_parameters":False,    # 必须设置避免内存泄漏  
            "seed":42  
        }
        
        if is_ds:
            trainer_kwargs.update(ds_kwargs)
        
        self.training_args = TrainingArguments(
           **trainer_kwargs 
        )
        
        self.dataset = self.load_dataset_from_hf()
        
        self.trainer = SFTTrainer(  
            model=self.model,  
            args=self.training_args,  
            train_dataset=self.dataset,  
            peft_config=peft_config,  
            max_seq_length=self.max_seq_length,  
            tokenizer=self.tokenizer,  
            formatting_func=self.format_instruction,  
            dataset_text_field="text",  
            packing=True  # 启用序列打包提升效率  
        )  




    # 格式化函数（将问答对转换为模型输入格式）  
    def format_instruction(self, sample):  
        # 拼接对话历史
        history = sample["history"] or []  
        
        return f"指令：{sample['instruction']}\n历史对话：{history}\n问题：{sample['input']}\n回答：{sample['output']}"  
    
    # 清洗异常数据  
    def data_clean(self, sample):  
        # 过滤空值数据  
        if not all([sample["instruction"], sample["input"], sample["output"]]):  
            return False  
        # 限制输入长度  
        if len(sample["input"]) > 512 or len(sample["output"]) > 1024:  
            return False  
        return True 
    
    
    
    def train(self):
        self.trainer.train()
    
    
    
    
    def save_model(self):
        # if self.accelerator.is_main_process:
        self.trainer.save_model(os.path.join(self.output_dir, "qwen2_cmed_deepspeed"))
        
        
        
    def load_dataset_from_hf(self, split = "train", is_json=False):
        
        if is_json:
            dataset = load_dataset("json", data_files=self.dataset_name_or_path)[split]
        else:
            dataset = load_dataset(self.dataset_name_or_path)[split]
