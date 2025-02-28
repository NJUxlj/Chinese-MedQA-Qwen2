import torch
import os

MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-1.5B"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOKENIZER_PATH = MODEL_PATH




DEEPSPEED_CONFIG_PATH = "./src/config/ds_config.json"
OUTPUT_DIR = "./output"


SFT_MODEL_NAME = "qwen2_cmed_sft"
SFT_MODEL_PATH = os.path.join("../../output", SFT_MODEL_NAME)


DPO_MODEL_NAME = "qwen2_cmed_dpo"
DPO_MODEL_PATH = ""


SFT_DPO_MODEL_NAME = ""
SFT_DPO_MODEL_PATH = ""