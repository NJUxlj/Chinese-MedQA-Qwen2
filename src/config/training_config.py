import logging
import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from typing import List, Optional, Any, Callable, Union, Dict
from pydantic import BaseModel, Field, validator
from config.base_config import BaseConfig




class BaseTrainingConfig(BaseConfig):
    model_name_or_path: str = Field(default = "Qwen/Qwen3-14B", description="模型名称或路径")

    max_seq_length: int = Field(default = 2048, description="最大序列长度")
    use_gradient_checkpointing: bool = Field(default = True, description="是否使用梯度检查点")
    device_map: Union[str, Dict[str, Any]] = Field(default = "auto", description="设备映射")

    use_4bit: bool = Field(default = False, description="是否使用 4bit 量化")
    use_fp16: bool = Field(default = True, description="是否使用 FP16 精度")
    use_bf16: bool = Field(default = False, description="是否使用 BF16 精度")
    trust_remote_code: bool = Field(default = True, description="是否信任远程代码")

    output_dir: str = Field(default = "output", description="模型权重输出目录")
    num_train_epochs: int = Field(default = 3, description="训练轮数")
    per_device_train_batch_size: int = Field(default = 4, description="训练批次大小")
    per_device_eval_batch_size: int = Field(default = 4, description="验证批次大小")
    gradient_accumulation_steps: int = Field(default = 1, description="梯度累积步数")
    learning_rate: float = Field(default = 2e-5, description="学习率")
    logging_steps: int = Field(default = 10, description="日志步数")
    save_steps: int = Field(default = 500, description="保存步数")

    save_total_limit: int = Field(default = 2, description="保存总限制")
    warmup_ratio: float = Field(default = 0.03, description="预热比例")
    lr_scheduler_type: str = Field(default = "cosine", description="学习率调度器类型")
    report_to: str = Field(default = "none", description="报告到")
    eval_strategy: str = Field(default = "steps", description="评估策略")
    eval_steps: Optional[int] = Field(default = 50, description="评估步数")
    load_best_model_at_end: bool = Field(default = False, description="是否在结束时加载最佳模型")
    metric_for_best_model: str = Field(default = "eval_loss", description="用于最佳模型评估的指标")
    greater_is_better: bool = Field(default = False, description="指标是否越大越好")
    max_grad_norm: float = Field(default = 1.0, description="最大梯度范数")
    dataloader_pin_memory: bool = Field(default = False, description="是否固定数据加载器内存")
    remove_unused_columns: bool = Field(default = False, description="是否移除未使用的列")
    do_train: bool = Field(default = True, description="是否进行训练")
    do_eval: bool = Field(default = False, description="是否进行评估")

    dataloader_num_workers: int = Field(default = 4, description="数据加载器工作线程数")
    @validator('device_map')
    def validate_device_map(cls, v):
        '''
        - "auto" - 自动设备映射
        - "balanced" - 平衡分配到所有GPU
        - "balanced_low_0" - 平衡分配但优先使用编号较低的GPU
        - "sequential" - 顺序分配
        - "cpu" - 使用CPU
        - "cuda" - 使用CUDA设备
        - "cuda:0" , "cuda:1" , ..., "cuda:9" - 使用指定编号的GPU

        '''
        # 如果是字符串，检查是否为预定义的有效值
        if isinstance(v, str):
            valid_string_values = {
                "auto", "balanced", "balanced_low_0", "sequential",
                "cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3", 
                "cuda:4", "cuda:5", "cuda:6", "cuda:7", "cuda:8", "cuda:9"
            }
            if v not in valid_string_values:
                raise ValueError(
                    f"device_map 字符串值必须是以下之一: {', '.join(sorted(valid_string_values))}, "
                    f"但得到了 '{v}'"
                )
        
        # 如果是字典，检查字典的键值
        elif isinstance(v, dict):
            valid_device_keys = {"cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3", 
                               "cuda:4", "cuda:5", "cuda:6", "cuda:7", "cuda:8", "cuda:9"}
            valid_layer_keys = {"embeddings", "encoder", "decoder", "lm_head", "transformer"}
            
            for key, value in v.items():
                # 检查设备值是否有效
                if value not in valid_device_keys:
                    raise ValueError(
                        f"device_map 字典中的设备值必须是以下之一: {', '.join(sorted(valid_device_keys))}, "
                        f"但 '{key}' 的值 '{value}' 无效"
                    )
                # 检查层名称是否为字符串
                if not isinstance(key, str):
                    raise ValueError(f"device_map 字典的键必须是字符串，但得到了 {type(key).__name__}: {key}")
        
        # 如果既不是字符串也不是字典，抛出错误
        else:
            raise ValueError(f"device_map 必须是字符串或字典，但得到了 {type(v).__name__}")
        
        return v




class RolloutConfig(BaseConfig):
    n: int = Field(default = 1, description="Rollout 次数")


class ActorRolloutRefConfig(BaseConfig):
    rollout: 'RolloutConfig' = Field(default_factory=RolloutConfig, description="Rollout 配置")












class DPOTrainingConfig(BaseTrainingConfig):
    pass




class DAPOTrainingConfig(BaseTrainingConfig):
    val_before_train: bool = Field(default = True, description="是否在训练前评估")
    val_only: bool = Field(default = False, description="是否仅进行评估")


    actor_rollout_ref: ActorRolloutRefConfig = Field(default_factory=ActorRolloutRefConfig(), description="Actor 回滚引用配置")












class GSPOTrainingConfig(BaseTrainingConfig):
    pass




class SFTTrainingConfig(BaseTrainingConfig):
    lora_rank: int = Field(default = 64, description="LoRA rank")
    lora_alpha: int = Field(default = 16, description="LoRA alpha")
    lora_dropout: float = Field(default = 0.05, description="LoRA dropout")
    target_modules: Optional[List[str]] = Field(default = None, description="目标模块")
    



