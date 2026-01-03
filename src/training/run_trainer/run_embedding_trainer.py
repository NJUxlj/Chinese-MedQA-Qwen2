import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import json
import argparse
from datasets import Dataset
from config.training_config import EmbeddingTrainingConfig
from training.trainer.embedding_trainer import EmbeddingTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Embedding Model Training")
    
    parser.add_argument("--model_name_or_path", type=str, required=True, help="模型名称或路径")
    parser.add_argument("--train_data_dir", type=str, required=True, help="训练数据目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--max_seq_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="每个设备的训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=500, help="模型保存步数")
    parser.add_argument("--warmup_steps", type=int, default=100, help="预热步数")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--fp16", action="store_true", help="是否使用 FP16 混合精度")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="是否启用梯度检查点")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="数据加载器工作进程数")
    
    parser.add_argument("--loss_type", type=str, default="softmax", help="损失类型: softmax, cosine, contrastive, info_nce, mnr")
    parser.add_argument("--temperature", type=float, default=0.02, help="温度参数")
    parser.add_argument("--margin", type=float, default=0.3, help="对比损失边界值")
    parser.add_argument("--normalized", type=str, default="true", choices=["true", "false"], help="是否对 embedding 归一化 (true/false)")
    parser.add_argument("--embedding_dim", type=int, default=768, help="embedding 维度")
    parser.add_argument("--num_classes", type=int, default=None, help="类别数（用于 softmax 损失）")
    
    parser.add_argument("--distributed_strategy", type=str, default="auto", help="分布式策略: auto, ddp, fsdp, deepspeed")
    parser.add_argument("--find_unused_parameters", action="store_true", default=False, help="是否查找未使用参数")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="日志报告目标")
    
    return parser.parse_args()


def load_json_data(data_dir, split="train"):
    data_file = os.path.join(data_dir, f"embedding_{split}.json")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"数据文件不存在: {data_file}")
    
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"加载 {split} 数据: {len(data)} 条样本")
    return data


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Embedding Model Training")
    print("=" * 60)
    print(f"模型: {args.model_name_or_path}")
    print(f"训练数据: {args.train_data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"损失类型: {args.loss_type}")
    print(f"温度参数: {args.temperature}")
    print(f"序列长度: {args.max_seq_length}")
    print(f"批次大小: {args.per_device_train_batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"训练轮数: {args.num_train_epochs}")
    print("=" * 60)
    
    config = EmbeddingTrainingConfig(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        use_fp16=args.fp16,
        use_gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        loss_type=args.loss_type,
        temperature=args.temperature,
        margin=args.margin,
        normalized=args.normalized.lower() == "true",
        embedding_dim=args.embedding_dim,
        num_classes=args.num_classes,
        distributed_strategy=args.distributed_strategy,
        find_unused_parameters=args.find_unused_parameters,
        report_to=args.report_to if args.report_to != "none" else "none",
    )
    
    trainer = EmbeddingTrainer(config)
    
    trainer.load_model_and_tokenizer()
    
    train_data = load_json_data(args.train_data_dir, "train")
    eval_data = load_json_data(args.train_data_dir, "valid")
    
    trainer.start_training(
        dataset=train_data,
        output_dir=args.output_dir,
        eval_dataset=eval_data,
    )
    
    print("=" * 60)
    print("训练完成！")
    print(f"模型已保存到: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
