# Qwen2.5-1.5B-Instruct GRPO对齐代码

根据Hugging Face TRL库的最佳实践，以下是对Qwen2.5-1.5B-Instruct模型使用GRPO进行对齐的完整代码。GRPO是一种在线学习算法，通过使用模型训练过程中自身生成的数据来迭代改进模型 [ref:7]。

## 安装必要的库

```bash
pip install transformers datasets trl peft accelerate bitsandbytes wandb
```


## 关键组件说明

1. **GRPO配置**：GRPO是一种比PPO更内存高效的算法，通过分组处理样本来进行策略优化
2. **LoRA微调**：使用参数高效微调方法，大大减少显存需求
3. **BitsAndBytes量化**：将模型参数量化为4位，进一步减少内存使用
4. **数据处理**：使用`DataCollatorForCompletionOnlyLM`确保只对助手回复部分计算损失

## 数据集格式要求

你的偏好数据集应该包含以下列：
- `prompt`: 用户的提问或指令
- `chosen`: 优质的回答（正面样本）
- `rejected`: 质量较差的回答（负面样本）



## DPO/GRPO 数据集下载
```bash
cd data

huggingface-cli download --repo-type dataset Zaynoid/med-dpo-10k --local-dir med-dpo-10k
```



## 进阶配置

1. 可以根据你的硬件资源调整以下参数：
   - `mini_batch_size`
   - `num_groups`
   - `per_device_train_batch_size`
   - `gradient_accumulation_steps`

2. 对于Qwen2.5-1.5B-Instruct模型，可能需要调整LoRA的`target_modules`以适配模型架构

3. 建议使用wandb或tensorboard监控训练过程中的指标变化





## 运行脚本
```bash

accelerate launch \
--config_file /root/autodl-tmp/Chinese-MedQA-Qwen2/src/config/deepspeed_config/ds_stage3_config.json \
/root/autodl-tmp/Chinese-MedQA-Qwen2/src/training/trainer/grpo_from_trl.py
```
