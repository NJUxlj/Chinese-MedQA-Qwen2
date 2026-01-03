import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Any
from datasets import Dataset

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.training_config import DPOTrainingConfig
from training.trainer.dpo_trainer import DPOTrainer
from utils.logger import setup_logger


def load_json_data(data_dir: str, split: str) -> List[Dict]:
    """加载 JSON 格式的数据集"""
    json_file = os.path.join(data_dir, f"{split}.json")
    jsonl_file = os.path.join(data_dir, f"{split}.jsonl")
    dpo_json_file = os.path.join(data_dir, f"dpo_{split}.json")
    dpo_jsonl_file = os.path.join(data_dir, f"dpo_{split}.jsonl")

    if os.path.exists(jsonl_file):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    elif os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'data' in data:
                return data['data']
            return data
    elif os.path.exists(dpo_jsonl_file):
        with open(dpo_jsonl_file, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    elif os.path.exists(dpo_json_file):
        with open(dpo_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'data' in data:
                return data['data']
            return data
    else:
        raise FileNotFoundError(f"数据文件不存在: {json_file} 或 {jsonl_file} 或 {dpo_json_file} 或 {dpo_jsonl_file}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DPO Training")

    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="模型名称或本地路径")
    parser.add_argument("--train_data_dir", type=str, required=True,
                        help="训练数据目录")
    parser.add_argument("--output_dir", type=str, default="./outputs/dpo",
                        help="输出目录")

    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="最大序列长度")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="训练批次大小")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="验证批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="学习率")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志步数")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="保存步数")
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="保存总限制")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="Learning rate scheduler type")

    parser.add_argument("--eval_strategy", type=str, default="steps",
                        help="评估策略")
    parser.add_argument("--eval_steps", type=int, default=None,
                        help="评估步数")
    parser.add_argument("--do_eval", action="store_true", default=False,
                        help="是否进行评估")
    parser.add_argument("--load_best_model_at_end", action="store_true", default=False,
                        help="是否在结束时加载最佳模型")

    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="最大梯度范数")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="数据加载器工作线程数")
    parser.add_argument("--remove_unused_columns", action="store_true", default=False,
                        help="是否移除未使用的列")

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
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO 温度参数")
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

    parser.add_argument("--prompt_field", type=str, default="prompt",
                        help="prompt 字段名")
    parser.add_argument("--chosen_field", type=str, default="chosen",
                        help="偏好答案字段名")
    parser.add_argument("--rejected_field", type=str, default="rejected",
                        help="不偏好答案字段名")
    parser.add_argument("--max_prompt_length", type=int, default=512,
                        help="最大 prompt 长度")
    parser.add_argument("--max_response_length", type=int, default=1536,
                        help="最大回答长度")

    return parser.parse_args()


def main():
    args = parse_args()

    logger = setup_logger("DPO Training", level="INFO")

    logger.info("=" * 60)
    logger.info("DPO Training (Direct Preference Optimization)")
    logger.info("=" * 60)
    logger.info(f"模型: {args.model_name_or_path}")
    logger.info(f"训练数据: {args.train_data_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"微调类型: {args.finetuning_type}")
    logger.info(f"DPO beta: {args.beta}")
    logger.info(f"序列长度: {args.max_seq_length}")
    logger.info(f"批次大小: {args.per_device_train_batch_size}")
    logger.info(f"学习率: {args.learning_rate}")
    logger.info(f"训练轮数: {args.num_train_epochs}")
    logger.info("=" * 60)

    target_modules = None
    if args.target_modules:
        target_modules = [m.strip() for m in args.target_modules.split(",")]

    config = DPOTrainingConfig(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to=args.report_to if args.report_to != "none" else "none",
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        load_best_model_at_end=args.load_best_model_at_end,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.dataloader_num_workers,
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

    trainer = DPOTrainer(
        config=config,
        finetuning_type=args.finetuning_type,
        beta=args.beta,
    )

    trainer.load_model_and_tokenizer()

    train_data = load_json_data(args.train_data_dir, "train")
    logger.info(f"加载 train 数据: {len(train_data)} 条样本")

    eval_data = None
    if args.do_eval:
        try:
            eval_data = load_json_data(args.train_data_dir, "valid")
            logger.info(f"加载 valid 数据: {len(eval_data)} 条样本")
        except FileNotFoundError:
            logger.warning("未找到验证数据，禁用评估")
            config.eval_strategy = "no"
            config.do_eval = False

    trainer.start_training(
        dataset=train_data,
        output_dir=args.output_dir,
        eval_dataset=eval_data,
        prompt_field=args.prompt_field,
        chosen_field=args.chosen_field,
        rejected_field=args.rejected_field,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
    )

    logger.info("=" * 60)
    logger.info("DPO 训练完成！")
    logger.info(f"模型已保存到: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
