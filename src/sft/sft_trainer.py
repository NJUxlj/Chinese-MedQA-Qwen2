import torch
import torch.nn as nn


from datasets import load_dataset  
from transformers import (  
    AutoModelForCausalLM,  
    AutoTokenizer,  
    TrainingArguments,  
    BitsAndBytesConfig  
)  
from peft import LoraConfig  
from trl import SFTTrainer  

from config.config import MODEL_PATH, TOKENIZER_PATH, DEVICE



# 量化配置（减少显存消耗）  
bnb_config = BitsAndBytesConfig(  
    load_in_4bit=True,           # 4位量化加载  
    bnb_4bit_quant_type="nf4",  # 量化类型  
    bnb_4bit_compute_dtype=torch.bfloat16,  
    bnb_4bit_use_double_quant=True  # 嵌套量化  
)  



# 加载预训练模型和分词器（使用7B指令版）  
model = AutoModelForCausalLM.from_pretrained(  
    MODEL_PATH,  
    quantization_config=bnb_config,  
    device_map="auto",  
    trust_remote_code=True  
)  


tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)  
tokenizer.pad_token = tokenizer.eos_token  # 设置填充token  




# LoRA配置（参数高效微调）  
peft_config = LoraConfig(  
    r=64,                # 低秩矩阵维度  
    lora_alpha=16,       # 缩放系数  
    lora_dropout=0.05,   # Dropout概率  
    target_modules=["q_proj", "v_proj"],  # 目标注意力层  
    bias="none",         # 不训练偏置项  
    task_type="CAUSAL_LM"  
)  

# 加载中医问答数据集（假设格式为{"instruction": ..., "response": ...}）  
dataset = load_dataset("json", data_files="tcm_qa.json")["train"] 

# 格式化函数（将问答对转换为模型输入格式）  
def format_instruction(sample):  
    return f"问：{sample['instruction']}\n答：{sample['response']}"  


# 训练参数配置  
training_args = TrainingArguments(  
    output_dir="./output",          # 输出目录  
    num_train_epochs=3,              # 训练轮次  
    per_device_train_batch_size=2,   # 批次大小  
    gradient_accumulation_steps=4,   # 梯度累积  
    learning_rate=2e-5,              # 学习率  
    fp16=True,                       # 混合精度训练  
    logging_steps=10,                # 日志间隔  
    save_strategy="epoch",           # 保存策略  
    report_to="tensorboard"          # 监控工具  
)  


# 初始化SFTTrainer  
trainer = SFTTrainer(  
    model=model,  
    args=training_args,  
    train_dataset=dataset,  
    peft_config=peft_config,  
    max_seq_length=1024,            # 最大序列长度  
    tokenizer=tokenizer,  
    formatting_func=format_instruction,  # 数据格式化函数  
    dataset_text_field="text"       # 数据集文本字段（自动生成）  
)  

# 开始训练  
trainer.train()  

# 保存微调后的模型  
trainer.save_model("qwen2_tcm_sft")  
