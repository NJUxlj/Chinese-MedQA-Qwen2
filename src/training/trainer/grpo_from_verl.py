import os
import torch
from typing import Dict, List

import verl  # 导入ByteDance的verl库
from verl.worker.computation.compute_client import LocalComputeClient
from verl.worker.algorithm.grpo.grpo_worker import GRPOWorker, GRPOConfig
from verl.worker.algorithm.grpo.grpo_worker_manager import GRPOWorkerManager
from verl.worker.scheduler.preemptible_scheduler import PreemptibleScheduler
from verl.worker.protocol.protocol import Protocol
from verl.trainer.policy.basic_policy_trainer import BasicPolicyTrainer
from verl.trainer.protocol.protocol import Protocol as TrainerProtocol
from verl.dataset.data_source import DataSource, PromptOnlyTask
from verl.dataset.shm_dict_dataset import ShmDictDataset
from verl.common.config import RunConfig, ModelConfig
from verl.common.metric_manager import create_metric_manager
from verl.worker.computation.remote_container import RemoteModelContainer

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

# 配置日志
import logging
logging.basicConfig(level=logging.INFO)

# 设置模型和训练配置
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
dataset_path = "your_preference_dataset.json"  # 你的偏好数据集路径
output_dir = "./qwen_grpo_output"
os.makedirs(output_dir, exist_ok=True)

# 运行配置
run_config = RunConfig(
    experiment_name="qwen-1.5b-grpo-alignment",
    run_name="run1",
    seed=42,
    resume=False,
    output_dir=output_dir,
)

# 模型配置
model_config = ModelConfig(
    model_name_or_path=model_name,
    trust_remote_code=True,
    # 其他模型配置...
)

# 1. 创建或加载数据源
def load_dataset():
    # 加载你的偏好数据集
    # 数据格式应该包含: prompt, chosen, rejected
    # 返回格式为verl期望的格式
    import json
    with open(dataset_path, "r") as f:
        data = json.load(f)
    
    prompts = []
    responses = []
    for item in data:
        prompts.append(item["prompt"])
        responses.append({
            "chosen": item["chosen"],
            "rejected": item["rejected"],
        })
    
    return prompts, responses

prompts, responses = load_dataset()
data_source = DataSource(
    tasks=[PromptOnlyTask(prompt) for prompt in prompts],
    responses=responses,
)

# 创建共享内存数据集
dataset = ShmDictDataset()
dataset.add_data(data_source.as_dict())

# 2. 设置模型和Tokenizer

# 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# LoRA配置
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_attn", "c_proj", "w1", "w2"],  # 适用于Qwen2.5架构
)

def create_model_and_tokenizer():
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 为QLoRA准备模型
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 确保添加了特殊的Qwen指令格式
    if not hasattr(tokenizer, "chat_template"):
        tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n<|im_start|>user\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'assistant' %}\n<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'system' %}\n<|im_start|>system\n{{ message['content'] }}<|im_end|>\n{% endif %}\n{% endfor %}"
    
    return model, tokenizer

model, tokenizer = create_model_and_tokenizer()

# 3. 配置GRPO算法

# GRPO配置
grpo_config = GRPOConfig(
    lr=5e-5,
    mini_batch_size=4,
    num_groups=4,
    chunk_size=16,
    beta=0.1,
    group_temperature=1.0,
    reward_temperature=1.0,
    weight_decay=0.05,
    gradient_accumulation_steps=8,
    normalize_advantage=True,
    clip_range=0.2,
    max_grad_norm=1.0,
    use_advantage=True,
    ignore_eos_token_reward=False,
    vhead_coef=0.0,  # 在GRPO中不使用value head
    prompt_truncation_side="right",
    response_template="<|im_start|>assistant\n",
    metrics_to_monitor=[],
    kl_penalty=0.1,
)

# 4. 创建Compute Client和Remote Model Container
compute_client = LocalComputeClient(
    device="cuda",
    mixed_precision="fp16",
)

remote_model_container = RemoteModelContainer(
    model=model,
    tokenizer=tokenizer,
    device_map="cuda",
)

# 5. 创建GRPO Worker
grpo_worker = GRPOWorker(
    config=grpo_config,
    compute_client=compute_client,
    dataset=dataset,
    remote_model_container=remote_model_container,
    protocol=Protocol(),
)

# 6. 创建Worker Manager和Scheduler
worker_manager = GRPOWorkerManager(
    workers=[grpo_worker],
    dataset=dataset,
)

scheduler = PreemptibleScheduler(
    worker_manager=worker_manager,
    epochs=3,
    steps_per_epoch=500,
    eval_every_n_steps=100,
)

# 7. 配置指标管理器
metric_manager = create_metric_manager(
    experiment_name=run_config.experiment_name,
    run_name=run_config.run_name,
    output_dir=run_config.output_dir,
    use_wandb=False,  # 设置为True启用WandB
)

# 8. 配置Policy Trainer
trainer = BasicPolicyTrainer(
    model=model,
    tokenizer=tokenizer,
    protocol=TrainerProtocol(),
    compute_client=compute_client,
    scheduler=scheduler,
    metric_manager=metric_manager,
    run_config=run_config,
)

# 9. 开始训练
trainer.train()

# 10. 保存最终模型
model.save_pretrained(os.path.join(output_dir, "final_model"))
tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))

print("GRPO训练完成！模型已保存到:", os.path.join(output_dir, "final_model"))