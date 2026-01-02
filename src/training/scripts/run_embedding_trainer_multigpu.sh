#!/bin/bash

# Multi-GPU Training Launch Script for EmbeddingTrainer
# 使用 Hugging Face Accelerate 进行分布式训练

set -e

# 默认配置
MODEL_NAME_OR_PATH="Qwen/Qwen3-Embedder-1.5B"
TRAIN_DATA_DIR="./data/embedding/train"
OUTPUT_DIR="./outputs/embedding/qwen3-embedder"
MAX_SEQ_LENGTH=512
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=2e-5
NUM_EPOCHS=3
LOGGING_STEPS=10
SAVE_STEPS=500
STRATEGY="auto"
NUM_GPUS=4
LOSS_TYPE="softmax"
TEMP=0.01
MARGIN=0.3

# 解析命令行参数
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
        --help)
            echo "使用方法: $0 [选项]"
            echo "选项:"
            echo "  --model_name_or_path <model>        模型名称或路径 (默认: Qwen/Qwen3-Embedder-1.5B)"
            echo "  --train_data_dir <dir>              训练数据目录 (默认: ./data/embedding/train)"
            echo "  --output_dir <dir>                  输出目录 (默认: ./outputs/embedding/qwen3-embedder)"
            echo "  --max_seq_length <length>           最大序列长度 (默认: 512)"
            echo "  --per_device_train_batch_size <bs>  每个设备的训练批次大小 (默认: 4)"
            echo "  --gradient_accumulation_steps <g>   梯度累积步数 (默认: 8)"
            echo "  --learning_rate <lr>                学习率 (默认: 2e-5)"
            echo "  --num_epochs <epochs>               训练轮数 (默认: 3)"
            echo "  --logging_steps <steps>             日志记录步数 (默认: 10)"
            echo "  --save_steps <steps>                模型保存步数 (默认: 500)"
            echo "  --strategy <strategy>               分布式策略: auto, ddp, fsdp, deepspeed (默认: auto)"
            echo "  --num_gpus <gpus>                   GPU数量 (默认: 4)"
            echo "  --loss_type <loss>                  损失类型: softmax, mnr, cosine, contrastive, info_nce (默认: softmax)"
            echo "  --temperature <temp>                Softmax温度参数 (默认: 0.01)"
            echo "  --margin <margin>                   对比损失边界值 (默认: 0.3)"
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
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true

# 设置日志文件
LOG_FILE="$OUTPUT_DIR/training_log_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "Embedding Multi-GPU Training"
echo "=========================================="
echo "模型: $MODEL_NAME_OR_PATH"
echo "训练数据: $TRAIN_DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "GPU数量: $NUM_GPUS"
echo "分布式策略: $STRATEGY"
echo "损失类型: $LOSS_TYPE"
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
            src/training/run_embedding_trainer.py \
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
            --warmup_steps 100 \
            --weight_decay 0.01 \
            --fp16 \
            --gradient_checkpointing \
            --dataloader_num_workers 4 \
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
            src/training/run_embedding_trainer.py \
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
            --warmup_steps 100 \
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
            src/training/run_embedding_trainer.py \
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
            --warmup_steps 100 \
            --weight_decay 0.01 \
            --gradient_checkpointing \
            2>&1 | tee "$LOG_FILE"
        ;;
        
    auto|AUTO|"")
        echo "使用自动检测策略..."
        export CUDA_VISIBLE_DEVICES="0,1,2,3"
        
accelerate launch \
            --multi_gpu \
            --num_processes $NUM_GPUS \
            --distributed_type multi_gpu \
            --mixed_precision fp16 \
            src/training/run_embedding_trainer.py \
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
            --distributed_strategy auto \
            --report_to tensorboard \
            --warmup_steps 100 \
            --weight_decay 0.01 \
            --gradient_checkpointing \
            2>&1 | tee "$LOG_FILE"
        ;;
        
    *)
        echo "不支持的策略: $STRATEGY"
        echo "支持的策略: ddp, fsdp, deepspeed, auto"
        exit 1
        ;;
esac

echo "=========================================="
echo "训练完成！"
echo "模型已保存到: $OUTPUT_DIR"
echo "日志文件: $LOG_FILE"
echo "=========================================="
