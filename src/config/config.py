import torch

MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-1.5B"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOKENIZER_PATH = MODEL_PATH