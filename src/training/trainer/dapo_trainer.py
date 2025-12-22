from typing import List, Optional, Union, Dict, Any
from pathlib import Path
from datasets import Dataset
import torch
import torch.nn.functional as F
import sys, os
import uuid
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    DataCollatorForSeq2Seq
)
from accelerate import Accelerator, DistributedDataParallelKwargs

sys.path.append(str(Path(__file__).parent.parent.parent))
from trainer.base_trainer import BaseTrainer
from config.training_config import DAPOTrainingConfig
from utils.logger import setup_logger
from utils.dataproto import DataProto

from verl.utils.profiler import marked_timmer


'''
DAPO 的训练样本中有 3 个字段： id, messages, ground_true_answer
- messages 中的前 n-1 轮， 就是传统意义上的 prompt
'''


class DAPOTrainer(BaseTrainer):
    def __init__(
        self, 
        config: DAPOTrainingConfig,
        val_reward_fn):
        super().__init__(config)


    

    def prepare_dataset(
        self,
        dataset: Union[Dataset, List[Dict]],
        messages_field: str = "messages",
        max_seq_length: Optional[int] = None
        ):
        pass




    def tokenize_dataset(self):
        pass




    def start_training(self):
        

        # 处理数据集



        # 训练




        # 保存最终模型权重
        pass



    def train(
        self,
        actor_model:AutoModelForCausalLM,
        reference_model:AutoModelForCausalLM,
        old_model:AutoModelForCausalLM,
        train_dataset,
        eval_dataset):

        train_dataloader = None
        eval_dataloader = None

        self.global_steps = 0
        # 假设当前有 n 张 GPU
        self.total_training_steps = self.config.num_train_epochs * len(train_dataset) // self.config.per_device_train_batch_size
        
        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0

        for epoch in range(self.config.num_train_epochs):
            for step, batch_dict in enumerate(train_dataloader):


                metrics = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1

                gen_batch = new_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps







                # 训练步骤
                self.global_steps += 1
                num_prompt_in_batch += 1

                # 更新进度条
                progress_bar.update(1)



    def _load_checkpoint(self, checkpoint_path: str):
        """
        从指定路径加载模型检查点
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=self.config.trust_remote_code
        )



    def _validate(self):
        pass