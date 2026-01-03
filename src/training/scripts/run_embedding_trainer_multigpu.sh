#!/bin/bash

# Multi-GPU Training Launch Script for EmbeddingTrainer
# 使用 Hugging Face Accelerate 进行分布式训练

set -e

# 默认配置
MODEL_NAME_OR_PATH="/Users/xiniuyiliao/Desktop/code/models/Qwen3-Embedder-0.6B"
TRAIN_DATA_DIR="./src/training/data/embedding_data"
OUTPUT_DIR="./outputs/embedding/qwen3-embedder"
MAX_SEQ_LENGTH=512
PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=2e-5
NUM_EPOCHS=3
LOGGING_STEPS=5
SAVE_STEPS=100
STRATEGY="single"
NUM_GPUS=1
LOSS_TYPE="mnr"
TEMP=0.01
MARGIN=0.3
NORMALIZED=true

# 解析命令行参数
# 只要命令行参数个数大于 0，就持续循环，逐个处理所有传入的参数
while [[ $# -gt 0 ]]; do
    # 使用 case 语句匹配当前第一个参数（$1）
    case $1 in
        # 如果当前参数是 --model_name_or_path
        --model_name_or_path)
            # 将下一个参数（$2）的值赋给变量 MODEL_NAME_OR_PATH
            MODEL_NAME_OR_PATH="$2"
            # 把参数指针整体向右移动两位，即跳过刚刚处理过的“选项”和“值”，进入下一轮循环
            shift 2
            ;;
        # 如果当前参数是 --train_data_dir
        --train_data_dir)
            # 将下一个参数（$2）的值赋给变量 TRAIN_DATA_DIR
            TRAIN_DATA_DIR="$2"
            # 同样移动两位，继续处理后续参数
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
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --loss_type)
            LOSS_TYPE="$2"
            shift 2
            ;;
        --temperature)
            TEMP="$2"
            shift 2
            ;;
        --margin)
            MARGIN="$2"
            shift 2
            ;;
        --normalized)
            NORMALIZED="$2"
            shift 2
            ;;
        --help)
            echo "使用方法: $0 [选项]"
            echo "选项:"
            echo "  --model_name_or_path <model>        模型名称或路径 (默认: Qwen/Qwen3-Embedder-0.6B)"
            echo "  --train_data_dir <dir>              训练数据目录 (默认: ./src/training/data/embedding_data)"
            echo "  --output_dir <dir>                  输出目录 (默认: ./outputs/embedding/qwen3-embedder)"
            echo "  --max_seq_length <length>           最大序列长度 (默认: 512)"
            echo "  --per_device_train_batch_size <bs>  每个设备的训练批次大小 (默认: 2)"
            echo "  --gradient_accumulation_steps <g>   梯度累积步数 (默认: 4)"
            echo "  --learning_rate <lr>                学习率 (默认: 2e-5)"
            echo "  --num_epochs <epochs>               训练轮数 (默认: 3)"
            echo "  --logging_steps <steps>             日志记录步数 (默认: 5)"
            echo "  --save_steps <steps>                模型保存步数 (默认: 100)"
            echo "  --strategy <strategy>               分布式策略: auto, ddp, fsdp, deepspeed, single (默认: single)"
            echo "  --num_gpus <gpus>                   GPU数量 (默认: 1)"
            echo "  --loss_type <loss>                  损失类型: softmax, mnr, cosine, contrastive, info_nce (默认: softmax)"
            echo "  --temperature <temp>                Softmax温度参数 (默认: 0.01)"
            echo "  --margin <margin>                   对比损失边界值 (默认: 0.3)"
            echo "  --normalized <bool>                 是否归一化 embedding (默认: true)"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 导出环境变量
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true

# 设置日志文件
LOG_FILE="$OUTPUT_DIR/training_log_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "Embedding Training Test"
echo "=========================================="
echo "模型: $MODEL_NAME_OR_PATH"
echo "训练数据: $TRAIN_DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "GPU数量: $NUM_GPUS"
echo "分布式策略: $STRATEGY"
echo "损失类型: $LOSS_TYPE"
echo "温度参数: $TEMP"
echo "归一化: $NORMALIZED"
echo "=========================================="

# 根据分布式策略选择启动命令
case $STRATEGY in
    ddp|DDP)
        echo "使用 DDP (Distributed Data Parallel) 策略..."
        CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch \
            --nproc_per_node=$NUM_GPUS \
            --nnodes=1 \
            --node_rank=0 \
            --master_port=29500 \
            src/training/run_trainer/run_embedding_trainer.py \
            --model_name_or_path "$MODEL_NAME_OR_PATH" \
            --train_data_dir "$TRAIN_DATA_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --max_seq_length $MAX_SEQ_LENGTH \
            --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
            --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
            --learning_rate $LEARNING_RATE \
            --num_train_epochs $NUM_EPOCHS \
            --logging_steps $LOGGING_STEPS \
            --save_steps $SAVE_STEPS \
            --loss_type $LOSS_TYPE \
            --temperature $TEMP \
            --margin $MARGIN \
            --distributed_strategy ddp \
            --find_unused_parameters False \
            --report_to tensorboard \
            --warmup_steps 10 \
            --weight_decay 0.01 \
            --gradient_checkpointing \
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
            src/training/run_trainer/run_embedding_trainer.py \
            --model_name_or_path "$MODEL_NAME_OR_PATH" \
            --train_data_dir "$TRAIN_DATA_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --max_seq_length $MAX_SEQ_LENGTH \
            --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
            --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
            --learning_rate $LEARNING_RATE \
            --num_train_epochs $NUM_EPOCHS \
            --logging_steps $LOGGING_STEPS \
            --save_steps $SAVE_STEPS \
            --loss_type $LOSS_TYPE \
            --temperature $TEMP \
            --margin $MARGIN \
            --distributed_strategy fsdp \
            --find_unused_parameters False \
            --report_to tensorboard \
            --warmup_steps 10 \
            --weight_decay 0.01 \
            --gradient_checkpointing \
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
            src/training/run_trainer/run_embedding_trainer.py \
            --model_name_or_path "$MODEL_NAME_OR_PATH" \
            --train_data_dir "$TRAIN_DATA_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --max_seq_length $MAX_SEQ_LENGTH \
            --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
            --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
            --learning_rate $LEARNING_RATE \
            --num_train_epochs $NUM_EPOCHS \
            --logging_steps $LOGGING_STEPS \
            --save_steps $SAVE_STEPS \
            --loss_type $LOSS_TYPE \
            --temperature $TEMP \
            --margin $MARGIN \
            --distributed_strategy deepspeed \
            --report_to tensorboard \
            --warmup_steps 10 \
            --weight_decay 0.01 \
            --gradient_checkpointing \
            2>&1 | tee "$LOG_FILE"
        ;;
        
    single|SINGLE|cpu|CPU|"")
        echo "使用单GPU/CPU模式..."
        
        if [ "$NORMALIZED" = "true" ]; then
            NORMALIZED_ARG="--normalized true"
        else
            NORMALIZED_ARG=""
        fi
        
        python src/training/run_trainer/run_embedding_trainer.py \
            --model_name_or_path "$MODEL_NAME_OR_PATH" \
            --train_data_dir "$TRAIN_DATA_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --max_seq_length $MAX_SEQ_LENGTH \
            --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
            --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
            --learning_rate $LEARNING_RATE \
            --num_train_epochs $NUM_EPOCHS \
            --logging_steps $LOGGING_STEPS \
            --save_steps $SAVE_STEPS \
            --loss_type $LOSS_TYPE \
            --temperature $TEMP \
            --margin $MARGIN \
            $NORMALIZED_ARG \
            --distributed_strategy single \
            --report_to none \
            --warmup_steps 10 \
            --weight_decay 0.01 \
            --gradient_checkpointing \
            --dataloader_num_workers 0 \
            2>&1 | tee "$LOG_FILE"      # 2>&1 把标准错误重定向到标准输出；| 把输出传给 tee；tee 同时把日志写入文件并显示到终端
        ;;
        
    *)
        echo "不支持的策略: $STRATEGY"
        echo "支持的策略: ddp, fsdp, deepspeed, single"
        exit 1
        ;;
esac

echo "=========================================="
echo "训练完成！"
echo "模型已保存到: $OUTPUT_DIR"
echo "日志文件: $LOG_FILE"
echo "=========================================="
