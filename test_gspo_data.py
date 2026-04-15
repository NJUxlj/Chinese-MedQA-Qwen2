"""
GSPO 训练测试脚本（简化版 - 避免 Mac MPS 内存问题）

使用本地 Qwen3-0.6B 模型进行测试
"""

import os
import sys
import json
from pathlib import Path

# 添加项目路径
project_root = '/Users/xiniuyiliao/Desktop/code/Chinese-MedQA-Qwen2'
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from datasets import Dataset

from training.trainer.gspo_trainer import GSPOTrainer, GSPOTrainingConfig


def create_test_dataset():
    """创建 10 条 GSPO 测试数据"""
    data = [
        {
            "id": "sample_001",
            "messages": [
                {"role": "user", "content": "1 + 1 等于多少?"},
                {"role": "assistant", "content": "2"}
            ],
            "ground_true_answer": "2"
        },
        {
            "id": "sample_002",
            "messages": [
                {"role": "user", "content": "2 + 3 等于多少?"},
                {"role": "assistant", "content": "5"}
            ],
            "ground_true_answer": "5"
        },
        {
            "id": "sample_003",
            "messages": [
                {"role": "user", "content": "5 - 2 等于多少?"},
                {"role": "assistant", "content": "3"}
            ],
            "ground_true_answer": "3"
        },
        {
            "id": "sample_004",
            "messages": [
                {"role": "user", "content": "3 * 4 等于多少?"},
                {"role": "assistant", "content": "12"}
            ],
            "ground_true_answer": "12"
        },
        {
            "id": "sample_005",
            "messages": [
                {"role": "user", "content": "10 / 2 等于多少?"},
                {"role": "assistant", "content": "5"}
            ],
            "ground_true_answer": "5"
        },
        {
            "id": "sample_006",
            "messages": [
                {"role": "user", "content": "中国的首都是哪里?"},
                {"role": "assistant", "content": "北京"}
            ],
            "ground_true_answer": "北京"
        },
        {
            "id": "sample_007",
            "messages": [
                {"role": "user", "content": "日本的首都是哪里?"},
                {"role": "assistant", "content": "东京"}
            ],
            "ground_true_answer": "东京"
        },
        {
            "id": "sample_008",
            "messages": [
                {"role": "user", "content": "美国的首都是哪里?"},
                {"role": "assistant", "content": "华盛顿"}
            ],
            "ground_true_answer": "华盛顿"
        },
        {
            "id": "sample_009",
            "messages": [
                {"role": "user", "content": "英国的首都是哪里?"},
                {"role": "assistant", "content": "伦敦"}
            ],
            "ground_true_answer": "伦敦"
        },
        {
            "id": "sample_010",
            "messages": [
                {"role": "user", "content": "法国的首都是哪里?"},
                {"role": "assistant", "content": "巴黎"}
            ],
            "ground_true_answer": "巴黎"
        },
    ]

    return Dataset.from_list(data)


def main():
    print("=" * 60)
    print("GSPO 训练器测试（简化版 - Mac MPS 友好）")
    print("=" * 60)

    # 创建测试数据集
    print("\n[1/4] 创建测试数据集...")
    dataset = create_test_dataset()
    print(f"数据集大小: {len(dataset)}")
    for i, item in enumerate(dataset):
        print(f"  样本 {i+1}: id={item['id']}, question={item['messages'][0]['content']}")

    # 配置 GSPO 训练器（使用较小batch和简化设置）
    print("\n[2/4] 配置 GSPO 训练器...")
    config = GSPOTrainingConfig(
        model_name_or_path="/Users/xiniuyiliao/Desktop/code/models/Qwen3-0.6B",
        output_dir="output/gspo_test",
        num_train_epochs=1,  # 测试只跑 1 个 epoch
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        logging_steps=5,
        save_steps=50,
        warmup_ratio=0.0,

        # GSPO 特有配置
        finetuning_type="lora",
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        beta=0.0,  # 跳过 KL 散度项以节省内存
        clip_ratio=0.2,
        group_size=2,  # 测试用小一点的 group size

        # 生成配置
        temperature=0.7,
        max_new_tokens=100,

        # 其他配置
        use_fp16=False,  # 关闭 fp16 避免兼容性问题
        use_bf16=False,
        trust_remote_code=True,
        do_eval=False,
        eval_strategy="no",
        dataloader_num_workers=0,
    )
    print(f"模型: {config.model_name_or_path}")
    print(f"LoRA: rank={config.lora_rank}, alpha={config.lora_alpha}")
    print(f"GSPO: beta={config.beta}, clip_ratio={config.clip_ratio}, group_size={config.group_size}")

    # 初始化训练器
    print("\n[3/4] 初始化 GSPO 训练器...")
    trainer = GSPOTrainer(
        config=config,
        finetuning_type="lora",
        beta=config.beta,
        clip_ratio=config.clip_ratio,
    )
    print("训练器初始化完成")

    # 开始训练
    print("\n[4/4] 开始 GSPO 训练...")
    print("-" * 40)
    try:
        metrics = trainer.start_training(
            dataset=dataset,
            output_dir=config.output_dir,
            messages_field="messages",
            ground_truth_field="ground_true_answer",
            id_field="id",
            do_eval=False,
            optim="adamw_torch",  # 使用标准 adamw 避免 MPS 兼容性问题
        )
        print("-" * 40)
        print(f"\n训练完成！")
        print(f"训练指标: {metrics}")
        print(f"模型保存到: {config.output_dir}")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
