import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Any
from datasets import Dataset

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.training_config import SFTTrainingConfig
from training.trainer.sft_trainer import SFTTrainer
from utils.logger import setup_logger


def load_json_data(data_dir: str, split: str) -> List[Dict]:
    """加载 JSON 格式的数据集"""
    possible_names = [
        f"{split}.json",
        f"{split}.jsonl",
        f"sft_{split}_data.json",
        f"sft_{split}_data.jsonl",
        f"sft_{split}.json",
        f"sft_{split}.jsonl",
    ]

    for name in possible_names:
        json_file = os.path.join(data_dir, name)
        jsonl_file = os.path.join(data_dir, name.replace(".json", ".jsonl"))

        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'data' in data:
                    return data['data']
                return data
        elif os.path.exists(jsonl_file):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                return [json.loads(line) for line in f]

    raise FileNotFoundError(f"数据文件不存在于 {data_dir}: {possible_names}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="SFT Training")

    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="模型名称或本地路径")
    parser.add_argument("--train_data_dir", type=str, required=True,
                        help="训练数据目录")
    parser.add_argument("--output_dir", type=str, default="./outputs/sft",
                        help="输出目录")

    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="最大序列长度")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="训练批次大小")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="验证批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志步数")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="保存步数")
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="保存总限制")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="预热比例")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="学习率调度器类型")

    parser.add_argument("--eval_strategy", type=str, default="steps",
                        help="评估策略")
    parser.add_argument("--eval_steps", type=int, default=None,
                        help="评估步数")
    parser.add_argument("--do_eval", action="store_true", default=False,
                        help="是否进行评估")
    parser.add_argument("--load_best_model_at_end", action="store_true", default=False,
                        help="是否在结束时加载最佳模型")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss",
                        help="最佳模型指标")
    parser.add_argument("--greater_is_better", action="store_true", default=False,
                        help="指标是否越大越好")

    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="最大梯度范数")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="数据加载器工作线程数")
    parser.add_argument("--remove_unused_columns", action="store_true", default=False,
                        help="是否移除未使用的列")
    parser.add_argument("--dataloader_pin_memory", action="store_true", default=False,
                        help="是否固定数据加载器内存")

    parser.add_argument("--use_fp16", action="store_true", default=False,
                        help="是否使用 FP16 精度")
    parser.add_argument("--use_bf16", action="store_true", default=False,
                        help="是否使用 BF16 精度")
    parser.add_argument("--use_4bit", action="store_true", default=False,
                        help="是否使用 4-bit 量化")
    parser.add_argument("--use_8bit", action="store_true", default=False,
                        help="是否使用 8-bit 量化")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=True,
                        help="是否使用梯度检查点")

    parser.add_argument("--distributed_strategy", type=str, default="single",
                        help="分布式策略: single, ddp, fsdp, deepspeed")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="本地进程编号")
    parser.add_argument("--find_unused_parameters", action="store_true", default=False,
                        help="是否查找未使用的参数")

    parser.add_argument("--finetuning_type", type=str, default="lora",
                        choices=["lora", "full"],
                        help="微调类型")
    parser.add_argument("--lora_rank", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--target_modules", type=str, default=None,
                        help="LoRA 目标模块，逗号分隔")

    parser.add_argument("--report_to", type=str, default="none",
                        help="报告到")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    parser.add_argument("--messages_field", type=str, default="messages",
                        help="messages 字段名（支持 OpenAI 格式）")

    return parser.parse_args()


def main():
    args = parse_args()

    logger = setup_logger("SFT Training", level="INFO")

    logger.info("=" * 60)
    logger.info("SFT Training (Supervised Fine-Tuning)")
    logger.info("=" * 60)
    logger.info(f"模型: {args.model_name_or_path}")
    logger.info(f"训练数据: {args.train_data_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"微调类型: {args.finetuning_type}")
    logger.info(f"序列长度: {args.max_seq_length}")
    logger.info(f"批次大小: {args.per_device_train_batch_size}")
    logger.info(f"学习率: {args.learning_rate}")
    logger.info(f"训练轮数: {args.num_epochs}")
    logger.info("=" * 60)

    target_modules = None
    if args.target_modules:
        target_modules = [m.strip() for m in args.target_modules.split(",")]

    config = SFTTrainingConfig(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to=args.report_to if args.report_to != "none" else "none",
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        remove_unused_columns=args.remove_unused_columns,
        do_train=True,
        do_eval=args.do_eval,
        use_fp16=args.use_fp16,
        use_bf16=args.use_bf16,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        distributed_strategy=args.distributed_strategy,
        find_unused_parameters=args.find_unused_parameters,
        seed=args.seed,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )

    trainer = SFTTrainer(
        config=config,
        finetuning_type=args.finetuning_type,
    )

    trainer.load_model_and_tokenizer()

    train_data = load_json_data(args.train_data_dir, "train")
    logger.info(f"加载 train 数据: {len(train_data)} 条样本")

    eval_data = None
    if args.do_eval:
        eval_data = load_json_data(args.train_data_dir, "valid")
        logger.info(f"加载 valid 数据: {len(eval_data)} 条样本")

    trainer.train(
        dataset=train_data,
        output_dir=args.output_dir,
        messages_field=args.messages_field,
        eval_dataset=eval_data,
    )

    logger.info("=" * 60)
    logger.info("SFT 训练完成！")
    logger.info(f"模型已保存到: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
