

使用说明：

1. **数据准备**：
   - SFT阶段：使用标准QA数据集（格式同之前）
   - DPO阶段：需要偏好数据集，格式示例：
     ```json
     [
       {
         "instruction": "如何泡茶？",
         "input": "",
         "output": [
           {"answer": "用沸水冲泡茶叶3分钟", "rank": 1},
           {"answer": "冷水浸泡茶叶12小时", "rank": 0}
         ]
       }
     ]
     ```

2. **启动训练**：
```bash
# 两阶段连续训练
llamafactory-cli train qwen2_sft_dpo.yaml

# 或分阶段训练
llamafactory-cli train --stage sft qwen2_sft_dpo.yaml
llamafactory-cli train --stage dpo qwen2_sft_dpo.yaml
```

关键配置说明：
1. **内存优化**：通过`flash_attn`和`gradient_checkpointing`降低显存消耗
2. **LoRA增强**：使用较大的rank值(64)适配DPO训练需求
3. **学习率调度**：DPO阶段使用更小的学习率(5e-6)进行微调
4. **偏好数据格式**：DPO数据集需要成对的偏好排序数据

建议配合DeepSpeed配置（如`deepspeed_zero3.json`）进行分布式训练，完整示例可参考LLaMA-Factory的examples目录[^2](https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/README.md)。