import os
import json
import socket

from omegaconf import OmegaConf
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator
import sys
sys.path.append(str(Path(__file__).parent.parent))


from config.settings import settings
from training.trainer.base_trainer import BaseTrainer



def main():
    # 设置配置文件路径
    base_trainer_yaml_path = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "yamls", 
        "base_trainer_config.yaml"
    )
    
    print(f"Loading config from: {base_trainer_yaml_path}")
    base_trainer_config = OmegaConf.load(base_trainer_yaml_path)
    
    # 设置训练和验证数据路径
    train_data_path = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "data", 
        "sft_data", 
        "sft_train_data.json"
    )
    
    valid_data_path = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "data", 
        "sft_data", 
        "sft_valid_data.json"
    )
    
    print(f"Loading training data from: {train_data_path}")
    print(f"Loading validation data from: {valid_data_path}")
    
    # 加载训练和验证数据
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(valid_data_path, 'r', encoding='utf-8') as f:
        valid_data = json.load(f)
    
    print(f"Loaded {len(train_data)} training samples")
    print(f"Loaded {len(valid_data)} validation samples")

    run_base_trainer(base_trainer_config, train_data, valid_data)





def run_base_trainer(base_trainer_config, train_data, valid_data):
    trainer = BaseTrainer(base_trainer_config)

    # 启动训练
    trainer.start_training(
        dataset=train_data,
        output_dir=base_trainer_config.output_dir if hasattr(base_trainer_config, 'output_dir') else './output',
        messages_field="messages",
        eval_dataset=valid_data
    )





if __name__ == "__main__":
    main()
