import torch
import torch.nn as nn



class DPOTrainer(nn.Module):
    def __init__(self, model, tokenizer, device):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer