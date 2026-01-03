#!/bin/bash

# Multi-GPU Training Launch Script for SFTTrainer
# 使用 Hugging Face Accelerate 进行分布式训练

set -e

MODEL_NAME_OR_PATH="/Users/xiniuyiliao/Desktop/code/models/Qwen2.5-7B-Instruct"
TRAIN_DATA_DIR="./src/training/data/sft_data"
OUTPUT_DIR="./outputs/sft/qwen2.5-7B"
MAX_SEQ_LENGTH=2048
PER_DEVICE_TRAIN_BATCH_SIZE=4
PER_DEVICE_EVAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=1e-5
NUM_EPOCHS=3
LOGGING_STEPS=10
SAVE_STEPS=500
STRATEGY="single"
NUM_GPUS=1
FINETUNING_TYPE="lora"
LORA_RANK=64
LORA_ALPHA=16
LORA_DROPOUT=0.05

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name_or_path)
            MODEL_NAME_OR_PATH="$2"
            shift 2
            ;;
        --train_data_dir)
            TRAIN_DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max_seq_length)
            MAX_SEQ_LENGTH="$2"
            shift 2
            ;;
        --per_device_train_batch_size)
            PER_DEVICE_TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --per_device_eval_batch_size)
            PER_DEVICE_EVAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --gradient_accumulation_steps)
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --logging_steps)
            LOGGING_STEPS="$2"
            shift 2
            ;;
        --save_steps)
            SAVE_STEPS="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --do_eval)
            DO_EVAL="--do_eval"
            shift
            ;;
        --eval_strategy)
            EVAL_STRATEGY="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --finetuning_type)
            FINETUNING_TYPE="$2"
            shift 2
            ;;
        --lora_rank)
            LORA_RANK="$2"
            shift 2
            ;;
        --lora_alpha)
            LORA_ALPHA="$2"
            shift 2
            ;;
        --lora_dropout)
            LORA_DROPOUT="$2"
            shift 2
            ;;
        --use_fp16)
            USE_FP16="--use_fp16"
            shift
            ;;
        --use_bf16)
            USE_BF16="--use_bf16"
            shift
            ;;
        --use_4bit)
            USE_4BIT="--use_4bit"
            shift
            ;;
        --use_8bit)
            USE_8BIT="--use_8bit"
            shift
            ;;
        --gradient_checkpointing)
            GRADIENT_CHECKPOINTING="--gradient_checkpointing"
            shift
            ;;
        --help)
            echo "使用方法: $0 [选项]"
            echo "选项:"
            echo "  --model_name_or_path <model>              模型名称或路径"
            echo "  --train_data_dir <dir>                    训练数据目录"
            echo "  --output_dir <dir>                        输出目录"
            echo "  --max_seq_length <length>                 最大序列长度"
            echo "  --per_device_train_batch_size <bs>        训练批次大小"
            echo "  --per_device_eval_batch_size <bs>         验证批次大小"
            echo "  --gradient_accumulation_steps <g>         梯度累积步数"
            echo "  --learning_rate <lr>                      学习率"
            echo "  --num_epochs <epochs>                     训练轮数"
            echo "  --logging_steps <steps>                   日志步数"
            echo "  --save_steps <steps>                      保存步数"
            echo "  --strategy <strategy>                     分布式策略: auto, ddp, fsdp, deepspeed, single"
            echo "  --num_gpus <gpus>                         GPU数量"
            echo "  --finetuning_type <type>                  微调类型: lora, full"
            echo "  --lora_rank <rank>                        LoRA rank"
            echo "  --lora_alpha <alpha>                      LoRA alpha"
            echo "  --lora_dropout <dropout>                  LoRA dropout"
            echo "  --use_fp16                                使用FP16精度"
            echo "  --use_bf16                                使用BF16精度"
            echo "  --use_4bit                                使用4-bit量化"
            echo "  --use_8bit                                使用8-bit量化"
            echo "  --gradient_checkpointing                  使用梯度检查点"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true

LOG_FILE="$OUTPUT_DIR/training_log_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "SFT Training (Supervised Fine-Tuning)"
echo "=========================================="
echo "模型: $MODEL_NAME_OR_PATH"
echo "训练数据: $TRAIN_DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "GPU数量: $NUM_GPUS"
echo "分布式策略: $STRATEGY"
echo "微调类型: $FINETUNING_TYPE"
echo "LoRA配置: rank=$LORA_RANK, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT"
echo "=========================================="

COMMON_ARGS="--model_name_or_path $MODEL_NAME_OR_PATH \
--train_data_dir $TRAIN_DATA_DIR \
--output_dir $OUTPUT_DIR \
--max_seq_length $MAX_SEQ_LENGTH \
--per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
--per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
--gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
--learning_rate $LEARNING_RATE \
--num_epochs $NUM_EPOCHS \
--logging_steps $LOGGING_STEPS \
--save_steps $SAVE_STEPS \
--finetuning_type $FINETUNING_TYPE \
--lora_rank $LORA_RANK \
--lora_alpha $LORA_ALPHA \
--lora_dropout $LORA_DROPOUT \
--distributed_strategy $STRATEGY \
--report_to tensorboard \
--warmup_ratio 0.03 \
--weight_decay 0.01 \
$USE_FP16 \
$USE_BF16 \
$USE_4BIT \
$USE_8BIT \
$GRADIENT_CHECKPOINTING \
$DO_EVAL \
--eval_strategy $EVAL_STRATEGY"

case $STRATEGY in
    ddp|DDP)
        echo "使用 DDP (Distributed Data Parallel) 策略..."
        CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch \
            --nproc_per_node=$NUM_GPUS \
            --nnodes=1 \
            --node_rank=0 \
            --master_port=29500 \
            src/training/run_trainer/run_sft_trainer.py \
            $COMMON_ARGS \
            --find_unused_parameters False \
            --dataloader_num_workers 0 \
            2>&1 | tee "$LOG_FILE"
        ;;
        
    fsdp|FSDP)
        echo "使用 FSDP (Fully Sharded Data Parallel) 策略..."
        export CUDA_VISIBLE_DEVICES="0,1,2,3"
        
        accelerate launch \
            --multi_gpu \
            --num_processes $NUM_GPUS \
            --distributed_type multi_gpu \
            --fsdp "full_shard auto_wrap" \
            --mixed_precision fp16 \
            src/training/run_trainer/run_sft_trainer.py \
            $COMMON_ARGS \
            2>&1 | tee "$LOG_FILE"
        ;;
        
    deepspeed|DEEPSPEED)
        echo "使用 DeepSpeed ZeRO 策略..."
        export CUDA_VISIBLE_DEVICES="0,1,2,3"
        
        accelerate launch \
            --multi_gpu \
            --num_processes $NUM_GPUS \
            --distributed_type multi_gpu \
            --deepspeed \
            --mixed_precision fp16 \
            src/training/run_trainer/run_sft_trainer.py \
            $COMMON_ARGS \
            2>&1 | tee "$LOG_FILE"
        ;;
        
    single|SINGLE|cpu|CPU|"")
        echo "使用单GPU/CPU模式..."
        
        python src/training/run_trainer/run_sft_trainer.py \
            $COMMON_ARGS \
            --dataloader_num_workers 0 \
            2>&1 | tee "$LOG_FILE"
        ;;
        
    *)
        echo "不支持的策略: $STRATEGY"
        echo "支持的策略: ddp, fsdp, deepspeed, single"
        exit 1
        ;;
esac

echo "=========================================="
echo "SFT 训练完成！"
echo "模型已保存到: $OUTPUT_DIR"
echo "日志文件: $LOG_FILE"
echo "=========================================="
