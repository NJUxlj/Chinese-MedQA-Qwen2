#!/usr/bin/env python3
"""
SFTTrainer 测试脚本
使用 GPT-2 模型测试 SFT 训练功能
"""

import os
import sys
from pathlib import Path

# 添加项目根目录和 src 目录到 Python 路径
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

from training.trainer.sft_trainer import SFTTrainer
from config.training_config import SFTTrainingConfig


def create_sample_dataset():
    """创建示例数据集"""
    train_data = [
        {
            "messages": [
                {"role": "user", "content": "什么是机器学习？"},
                {"role": "assistant", "content": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习，而不需要明确的编程。"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "解释一下深度学习。"},
                {"role": "assistant", "content": "深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的复杂模式。"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "什么是神经网络？"},
                {"role": "assistant", "content": "神经网络是一种受人类大脑启发的计算系统，由相互连接的节点（神经元）组成。"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "介绍一下自然语言处理。"},
                {"role": "assistant", "content": "自然语言处理是人工智能的一个领域，专注于计算机与人类语言之间的交互。"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "什么是强化学习？"},
                {"role": "assistant", "content": "强化学习是一种机器学习方法，智能体通过与环境交互来学习最佳行为策略。"}
            ]
        }
    ]
    
    eval_data = [
        {
            "messages": [
                {"role": "user", "content": "什么是卷积神经网络？"},
                {"role": "assistant", "content": "卷积神经网络是一种专门用于处理网格状数据的深度学习架构。"}
            ]
        }
    ]
    
    return train_data, eval_data


def test_lora_finetuning():
    """测试 LoRA 微调"""
    print("=" * 60)
    print("测试 1: LoRA 微调")
    print("=" * 60)
    
    config = SFTTrainingConfig(
        model_name_or_path="openai-community/gpt2",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        logging_steps=10,
        save_steps=100,
        max_seq_length=128,
        output_dir="./test_output/gpt2_lora",
        use_fp16=False,
        use_bf16=False,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    
    trainer = SFTTrainer(
        config=config,
        finetuning_type="lora"
    )
    
    train_data, eval_data = create_sample_dataset()
    
    print(f"训练数据量: {len(train_data)}")
    print(f"评估数据量: {len(eval_data)}")
    
    try:
        metrics = trainer.train(
            dataset=train_data,
            output_dir=config.output_dir,
            eval_dataset=eval_data,
            do_eval=True
        )
        
        print(f"训练完成！最终损失: {metrics.get('train_loss', 'N/A'):.4f}")
        return True
        
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_finetuning():
    """测试全量微调"""
    print("\n" + "=" * 60)
    print("测试 2: 全量微调")
    print("=" * 60)
    
    config = SFTTrainingConfig(
        model_name_or_path="openai-community/gpt2",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=100,
        max_seq_length=128,
        output_dir="./test_output/gpt2_full",
        use_fp16=False,
        use_bf16=False,
    )
    
    trainer = SFTTrainer(
        config=config,
        finetuning_type="full"
    )
    
    train_data, eval_data = create_sample_dataset()
    
    print(f"训练数据量: {len(train_data)}")
    print(f"评估数据量: {len(eval_data)}")
    
    try:
        metrics = trainer.train(
            dataset=train_data,
            output_dir=config.output_dir,
            eval_dataset=eval_data,
            do_eval=True
        )
        
        print(f"训练完成！最终损失: {metrics.get('train_loss', 'N/A'):.4f}")
        return True
        
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lora_merge():
    """测试 LoRA adapter 合并"""
    print("\n" + "=" * 60)
    print("测试 3: LoRA Adapter 合并")
    print("=" * 60)
    
    config = SFTTrainingConfig(
        model_name_or_path="openai-community/gpt2",
        max_seq_length=128,
    )
    
    trainer = SFTTrainer(config=config)
    
    base_model_path = "openai-community/gpt2"
    adapter_path = "./test_output/gpt2_lora"
    output_dir = "./test_output/gpt2_merged"
    
    try:
        merged_model = trainer.merge_lora_and_save(
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            output_dir=output_dir,
            save_model=True
        )
        
        print(f"合并完成！合并后的模型已保存到: {output_dir}")
        return True
        
    except Exception as e:
        print(f"合并失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference():
    """测试推理功能"""
    print("\n" + "=" * 60)
    print("测试 4: 推理测试")
    print("=" * 60)
    
    config = SFTTrainingConfig(
        model_name_or_path="openai-community/gpt2",
        max_seq_length=128,
    )
    
    trainer = SFTTrainer(config=config)
    
    model_path = "./test_output/gpt2_merged"
    
    try:
        trainer.load_inference_model(model_path=model_path)
        
        prompt = "什么是机器学习？"
        response = trainer.generate(
            prompt=prompt,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True
        )
        
        print(f"输入: {prompt}")
        print(f"输出: {response}")
        return True
        
    except Exception as e:
        print(f"推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup():
    """清理测试输出"""
    import shutil
    test_dirs = [
        "./test_output/gpt2_lora",
        "./test_output/gpt2_full",
        "./test_output/gpt2_merged"
    ]
    
    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"已清理: {dir_path}")
            except Exception as e:
                print(f"清理失败 {dir_path}: {e}")


def main():
    """主函数"""
    print("SFTTrainer 测试脚本")
    print("=" * 60)
    
    all_passed = True
    
    if not test_lora_finetuning():
        all_passed = False
    
    if not test_full_finetuning():
        all_passed = False
    
    if not test_lora_merge():
        all_passed = False
    
    if not test_inference():
        all_passed = False
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    if all_passed:
        print("✓ 所有测试通过！")
    else:
        print("✗ 部分测试失败")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
