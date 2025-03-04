# 数据预处理脚本示例  
python -m llama_factory.train.preprocess \
  --stage sft \
  --dataset_dir data/raw/medical_qa \
  --output_dir data/processed/sft \
  --template qwen \
  --max_seq_length 2048 \
  --split_ratio 0.9,0.1,0.0  