import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig, DataCollatorForCompletionOnlyLM

# 设置Hugging Face token (如果需要访问私有模型或上传结果)
# os.environ["HF_TOKEN"] = "你的HF token"

# 定义模型和数据集
model_name = "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct"
dataset_name = "/root/autodl-tmp/Chinese-MedQA-Qwen2/data/med-dpo-10k"  # 替换为你的偏好数据集 [prompr, chosen, rejected]

# # 使用BitsAndBytes进行量化，节省显存
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
# )

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# 为QLoRA准备模型
# model = prepare_model_for_kbit_training(model)

# 定义LoRA配置
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_attn", "c_proj", "w1", "w2"], # 根据Qwen2.5-1.5B模型结构调整
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    trust_remote_code=True,
    padding_side="right",
)
tokenizer.pad_token = tokenizer.eos_token

# 确保添加了特殊的Qwen指令格式
tokenizer.chat_template = tokenizer.chat_template if hasattr(tokenizer, "chat_template") else "{% for message in messages %}\n{% if message['role'] == 'user' %}\n<|im_start|>user\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'assistant' %}\n<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'system' %}\n<|im_start|>system\n{{ message['content'] }}<|im_end|>\n{% endif %}\n{% endfor %}"

# 加载数据集
dataset = load_dataset(dataset_name)
train_dataset = dataset["train"]

# 定义数据处理函数
def preprocess_function(examples):
    prompt_format = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # 提取prompt和completion
    model_inputs = {
        "prompt": [prompt_format.format(prompt=p) for p in examples["question"]],
        "chosen": [c for c in examples["chosen"]],
        "rejected": [r for r in examples["rejected"]],
    }
    
    return model_inputs

# 应用数据预处理
processed_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)

# 定义GRPO配置
grpo_config = GRPOConfig(
    mini_batch_size=1,
    num_groups=5,  # 根据你的硬件和数据集调整
    chunk_size=16,
    beta=0.1,
)

# 创建数据整理器，用于只对completion部分计算loss
response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./save",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.05,
    fp16=True,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    report_to="tensorboard",  # 如果不需要wandb，可以改为"none"
    seed=42,
    push_to_hub=False,  # 设置为True，如果你想推送到Hugging Face Hub
)

# 创建GRPO训练器
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    data_collator=collator,
    grpo_config=grpo_config,
    compute_metrics=None,  # 如果需要自定义评估指标，可以添加
)

# 开始训练
trainer.train()

# 保存最终模型
trainer.save_model("./save/qwen2_5_1_5b_instruct_grpo")

# 如果需要保存到HF Hub
# trainer.push_to_hub()


