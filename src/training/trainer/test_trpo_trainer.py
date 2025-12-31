#!/usr/bin/env python3
"""
TRPOTrainer 测试脚本
使用 GPT-2 模型测试 TRPO (Trust Region Policy Optimization) 训练功能
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from training.trainer.trpo_trainer import TRPOTrainer
from config.training_config import TRPOTrainingConfig


from dotenv import load_dotenv
load_dotenv()

LOCAL_MODEL_PATH = os.getenv('LOCAL_MODEL_PATH')

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """设置日志记录器"""
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger


def load_trpo_dataset(data_path: str) -> list:
    """加载 TRPO 训练数据"""
    logger = setup_logger("data_loader")
    
    if not os.path.exists(data_path):
        logger.error(f"数据文件不存在: {data_path}")
        return []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"成功加载 {len(data)} 个 TRPO 训练样本")
    return data


def create_test_dataset() -> list:
    """创建用于测试的小型数据集"""
    return [
        {
            "prompt": "请计算：15 + 27 等于多少？",
            "chosen": "15 + 27 = 42。计算过程：5+7=12（个位），写2进1；1+2+1=4（十位），结果是42。",
            "rejected": "15 + 27 = 42"
        },
        {
            "prompt": "请计算：8 × 7 等于多少？",
            "chosen": "8 × 7 = 56。根据乘法口诀'七八五十六'，即8个7相加等于56。",
            "rejected": "8 × 7 = 56"
        },
        {
            "prompt": "请计算：144 ÷ 12 等于多少？",
            "chosen": "144 ÷ 12 = 12。12 × 12 = 144，所以商是12。",
            "rejected": "144 ÷ 12 = 12"
        },
        {
            "prompt": "请计算：3² + 4² 等于多少？",
            "chosen": "3² + 4² = 9 + 16 = 25。这是一个经典的勾股数组合。",
            "rejected": "3² + 4² = 25"
        },
        {
            "prompt": "请计算：2⁵ 等于多少？",
            "chosen": "2⁵ = 32。2的5次方表示2×2×2×2×2=32。",
            "rejected": "2⁵ = 32"
        }
    ]


def prepare_trpo_dataset(dataset: list, tokenizer, max_length: int = 128) -> dict:
    """准备 TRPO 数据集用于训练
    
    Args:
        dataset: 原始数据集
        tokenizer: 分词器
        max_length: 最大序列长度
    
    Returns:
        包含 input_ids, attention_mask, chosen_ids, rejected_ids 的字典
    """
    logger = setup_logger("data_preparation")
    
    prepared_data = {
        "input_ids": [],
        "attention_mask": [],
        "chosen_ids": [],
        "rejected_ids": [],
        "prompt": [],
        "chosen": [],
        "rejected": []
    }
    
    for item in dataset:
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        chosen_text = prompt + chosen
        rejected_text = prompt + rejected
        
        chosen_encoding = tokenizer(
            chosen_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        rejected_encoding = tokenizer(
            rejected_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        prompt_encoding = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        prepared_data["input_ids"].append(prompt_encoding["input_ids"].squeeze())
        prepared_data["attention_mask"].append(prompt_encoding["attention_mask"].squeeze())
        prepared_data["chosen_ids"].append(chosen_encoding["input_ids"].squeeze())
        prepared_data["rejected_ids"].append(rejected_encoding["input_ids"].squeeze())
        prepared_data["prompt"].append(prompt)
        prepared_data["chosen"].append(chosen)
        prepared_data["rejected"].append(rejected)
    
    logger.info(f"数据准备完成: {len(dataset)} 个样本")
    return prepared_data


def test_initialization():
    """测试 1: TRPOTrainer 初始化"""
    logger = setup_logger("test_initialization")
    logger.info("=" * 60)
    logger.info("测试 1: TRPOTrainer 初始化")
    logger.info("=" * 60)
    
    try:
        config = TRPOTrainingConfig(
            model_name_or_path=LOCAL_MODEL_PATH,
            max_seq_length=128,
            learning_rate=1e-4,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            logging_steps=10,
            save_steps=100,
            output_dir="./test_output/trpo_test",
            max_kl=0.01,
            cg_damping=0.1,
            cg_iterations=10,
            gamma=0.99,
            gae_lambda=0.95,
        )
        
        trainer = TRPOTrainer(
            config=config,
            finetuning_type="lora",
            use_deepspeed=False
        )
        
        logger.info("✓ TRPOTrainer 初始化成功")
        logger.info(f"  - 模型路径: {config.model_name_or_path}")
        logger.info(f"  - 微调类型: lora")
        logger.info(f"  - 最大序列长度: {config.max_seq_length}")
        logger.info(f"  - 学习率: {config.learning_rate}")
        logger.info(f"  - max_kl: {config.max_kl}")
        logger.info(f"  - cg_damping: {config.cg_damping}")
        logger.info(f"  - cg_iterations: {config.cg_iterations}")
        logger.info(f"  - gamma: {config.gamma}")
        logger.info(f"  - gae_lambda: {config.gae_lambda}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tokenizer_loading():
    """测试 2: Tokenizer 加载"""
    logger = setup_logger("test_tokenizer")
    logger.info("\n" + "=" * 60)
    logger.info("测试 2: Tokenizer 加载")
    logger.info("=" * 60)
    
    try:
        config = TRPOTrainingConfig(
            model_name_or_path=LOCAL_MODEL_PATH,
            max_seq_length=128,
        )
        
        trainer = TRPOTrainer(config=config)
        trainer._load_tokenizer()
        
        logger.info("✓ Tokenizer 加载成功")
        logger.info(f"  - 词汇表大小: {len(trainer.tokenizer)}")
        logger.info(f"  - 特殊令牌: pad_token={trainer.tokenizer.pad_token}, eos_token={trainer.tokenizer.eos_token}")
        
        test_text = "15 + 27 = ?"
        tokens = trainer.tokenizer.encode(test_text)
        logger.info(f"  - 测试文本: '{test_text}'")
        logger.info(f"  - Token IDs: {tokens}")
        logger.info(f"  - Token 数量: {len(tokens)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Tokenizer 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """测试 3: 模型加载"""
    logger = setup_logger("test_model_loading")
    logger.info("\n" + "=" * 60)
    logger.info("测试 3: 模型加载")
    logger.info("=" * 60)
    
    try:
        config = TRPOTrainingConfig(
            model_name_or_path=LOCAL_MODEL_PATH,
            max_seq_length=128,
        )
        
        trainer = TRPOTrainer(config=config)
        trainer._setup_accelerator()
        trainer._load_tokenizer()
        trainer._load_model()
        
        logger.info("✓ 模型加载成功")
        logger.info(f"  - 模型类型: {type(trainer.model).__name__}")
        
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in trainer.model.parameters())
        logger.info(f"  - 总参数量: {total_params:,}")
        logger.info(f"  - 可训练参数量: {trainable_params:,}")
        
        if hasattr(trainer, 'accelerator'):
            logger.info(f"  - 设备: {trainer.accelerator.device}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reference_model():
    """测试 4: 参考模型加载"""
    logger = setup_logger("test_reference_model")
    logger.info("\n" + "=" * 60)
    logger.info("测试 4: 参考模型加载")
    logger.info("=" * 60)
    
    try:
        config = TRPOTrainingConfig(
            model_name_or_path=LOCAL_MODEL_PATH,
            max_seq_length=128,
        )
        
        trainer = TRPOTrainer(config=config)
        trainer._setup_accelerator()
        trainer._load_tokenizer()
        trainer._load_model()
        trainer._load_reference_model()
        
        logger.info("✓ 参考模型加载成功")
        logger.info(f"  - 参考模型类型: {type(trainer.ref_model).__name__}")
        
        ref_params = sum(p.numel() for p in trainer.ref_model.parameters())
        logger.info(f"  - 参考模型参数量: {ref_params:,}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 参考模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_preparation():
    """测试 5: 数据准备"""
    logger = setup_logger("test_data_preparation")
    logger.info("\n" + "=" * 60)
    logger.info("测试 5: 数据准备")
    logger.info("=" * 60)
    
    try:
        config = TRPOTrainingConfig(
            model_name_or_path=LOCAL_MODEL_PATH,
            max_seq_length=128,
        )
        
        trainer = TRPOTrainer(config=config)
        trainer._load_tokenizer()
        
        data_path = "/Users/xiniuyiliao/Desktop/code/Chinese-MedQA-Qwen2/src/training/data/trpo_data/trpo_train.json"
        
        if os.path.exists(data_path):
            dataset = load_trpo_dataset(data_path)
        else:
            logger.warning(f"使用测试数据集代替: {data_path}")
            dataset = create_test_dataset()
        
        prepared_data = prepare_trpo_dataset(dataset, trainer.tokenizer, max_length=128)
        
        logger.info("✓ 数据准备成功")
        logger.info(f"  - 样本数量: {len(prepared_data['prompt'])}")
        logger.info(f"  - input_ids 形状: {prepared_data['input_ids'][0].shape}")
        logger.info(f"  - chosen_ids 形状: {prepared_data['chosen_ids'][0].shape}")
        
        sample_idx = 0
        logger.info(f"\n  样本示例 (idx={sample_idx}):")
        logger.info(f"    Prompt: {prepared_data['prompt'][sample_idx]}")
        logger.info(f"    Chosen 长度: {len(prepared_data['chosen'][sample_idx])} 字符")
        logger.info(f"    Rejected 长度: {len(prepared_data['rejected'][sample_idx])} 字符")
        
        return prepared_data
        
    except Exception as e:
        logger.error(f"✗ 数据准备失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_kl_divergence():
    """测试 6: KL 散度计算"""
    logger = setup_logger("test_kl_divergence")
    logger.info("\n" + "=" * 60)
    logger.info("测试 6: KL 散度计算")
    logger.info("=" * 60)
    
    try:
        config = TRPOTrainingConfig(
            model_name_or_path=LOCAL_MODEL_PATH,
            max_seq_length=128,
        )
        
        trainer = TRPOTrainer(config=config)
        trainer._setup_accelerator()
        trainer._load_tokenizer()
        trainer._load_model()
        trainer._load_reference_model()
        
        dataset = create_test_dataset()
        prepared_data = prepare_trpo_dataset(dataset, trainer.tokenizer, max_length=128)
        
        input_ids = torch.stack(prepared_data["input_ids"]).to(trainer.accelerator.device)
        attention_mask = torch.stack(prepared_data["attention_mask"]).to(trainer.accelerator.device)
        chosen_ids = torch.stack(prepared_data["chosen_ids"]).to(trainer.accelerator.device)
        
        kl_div = trainer._compute_kl_divergence(
            trainer.model,
            trainer.ref_model,
            input_ids,
            attention_mask,
            chosen_ids
        )
        
        logger.info("✓ KL 散度计算成功")
        logger.info(f"  - KL 散度值: {kl_div.item():.6f}")
        logger.info(f"  - 输入形状: {input_ids.shape}")
        logger.info(f"  - 响应长度: {chosen_ids.shape[1]}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ KL 散度计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gae_computation():
    """测试 7: GAE (广义优势估计) 计算"""
    logger = setup_logger("test_gae")
    logger.info("\n" + "=" * 60)
    logger.info("测试 7: GAE 计算")
    logger.info("=" * 60)
    
    try:
        config = TRPOTrainingConfig(
            model_name_or_path=LOCAL_MODEL_PATH,
            max_seq_length=128,
        )
        
        trainer = TRPOTrainer(config=config)
        trainer._setup_accelerator()
        
        batch_size = 4
        seq_len = 20
        
        rewards = torch.randn(batch_size, seq_len).to(trainer.accelerator.device)
        values = torch.randn(batch_size, seq_len + 1).to(trainer.accelerator.device)
        masks = torch.ones(batch_size, seq_len).to(trainer.accelerator.device)
        
        advantages, returns = trainer._compute_gae(
            rewards,
            values,
            masks,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda
        )
        
        logger.info("✓ GAE 计算成功")
        logger.info(f"  - 奖励形状: {rewards.shape}")
        logger.info(f"  - 价值形状: {values.shape}")
        logger.info(f"  - 优势形状: {advantages.shape}")
        logger.info(f"  - 回报形状: {returns.shape}")
        logger.info(f"  - 优势均值: {advantages.mean().item():.4f}")
        logger.info(f"  - 回报均值: {returns.mean().item():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ GAE 计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_conjugate_gradient():
    """测试 8: 共轭梯度法"""
    logger = setup_logger("test_conjugate_gradient")
    logger.info("\n" + "=" * 60)
    logger.info("测试 8: 共轭梯度法")
    logger.info("=" * 60)
    
    try:
        config = TRPOTrainingConfig(
            model_name_or_path=LOCAL_MODEL_PATH,
            max_seq_length=128,
        )
        
        trainer = TRPOTrainer(config=config)
        trainer._setup_accelerator()
        
        dim = 10
        A = torch.randn(dim, dim).to(trainer.accelerator.device)
        A = A @ A.T + torch.eye(dim).to(trainer.accelerator.device)
        b = torch.randn(dim).to(trainer.accelerator.device)
        
        def Ax(v):
            return A @ v
        
        x = trainer._conjugate_gradient(Ax, b, max_iter=20, tol=1e-10)
        
        residual = torch.norm(A @ x - b)
        
        logger.info("✓ 共轭梯度法计算成功")
        logger.info(f"  - 矩阵维度: {dim}x{dim}")
        logger.info(f"  - 解向量形状: {x.shape}")
        logger.info(f"  - 残差范数: {residual.item():.10f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 共轭梯度法计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_log_probability():
    """测试 9: 对数概率计算"""
    logger = setup_logger("test_log_prob")
    logger.info("\n" + "=" * 60)
    logger.info("测试 9: 对数概率计算")
    logger.info("=" * 60)
    
    try:
        config = TRPOTrainingConfig(
            model_name_or_path=LOCAL_MODEL_PATH,
            max_seq_length=128,
        )
        
        trainer = TRPOTrainer(config=config)
        trainer._setup_accelerator()
        trainer._load_tokenizer()
        trainer._load_model()
        
        dataset = create_test_dataset()
        prepared_data = prepare_trpo_dataset(dataset, trainer.tokenizer, max_length=128)
        
        input_ids = torch.stack(prepared_data["input_ids"]).to(trainer.accelerator.device)
        attention_mask = torch.stack(prepared_data["attention_mask"]).to(trainer.accelerator.device)
        chosen_ids = torch.stack(prepared_data["chosen_ids"]).to(trainer.accelerator.device)
        
        log_probs = trainer._compute_log_probs(
            trainer.model,
            input_ids,
            attention_mask,
            chosen_ids
        )
        
        logger.info("✓ 对数概率计算成功")
        logger.info(f"  - Log probs 形状: {log_probs.shape}")
        logger.info(f"  - Log probs 均值: {log_probs.mean().item():.4f}")
        logger.info(f"  - Log probs 范围: [{log_probs.min().item():.4f}, {log_probs.max().item():.4f}]")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 对数概率计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_training_loop():
    """测试 10: 完整训练循环（模拟）"""
    logger = setup_logger("test_training_loop")
    logger.info("\n" + "=" * 60)
    logger.info("测试 10: 完整训练循环（模拟）")
    logger.info("=" * 60)
    
    try:
        config = TRPOTrainingConfig(
            model_name_or_path=LOCAL_MODEL_PATH,
            max_seq_length=128,
            learning_rate=1e-4,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            max_kl_div=0.01,
            cg_damping=0.1,
            cg_iterations=10,
            gamma=0.99,
            gae_lambda=0.95,
        )
        
        trainer = TRPOTrainer(config=config)
        trainer._setup_accelerator()
        trainer._load_tokenizer()
        trainer._load_model()
        trainer._load_reference_model()
        
        dataset = create_test_dataset()
        prepared_data = prepare_trpo_dataset(dataset, trainer.tokenizer, max_length=128)
        
        input_ids = torch.stack(prepared_data["input_ids"]).to(trainer.accelerator.device)
        attention_mask = torch.stack(prepared_data["attention_mask"]).to(trainer.accelerator.device)
        chosen_ids = torch.stack(prepared_data["chosen_ids"]).to(trainer.accelerator.device)
        rejected_ids = torch.stack(prepared_data["rejected_ids"]).to(trainer.accelerator.device)
        
        logger.info("开始模拟训练循环...")
        
        num_epochs = 1
        log_probs_chosen_old = trainer._compute_log_probs(
            trainer.model, input_ids, attention_mask, chosen_ids
        )
        log_probs_rejected_old = trainer._compute_log_probs(
            trainer.model, input_ids, attention_mask, rejected_ids
        )
        
        advantages = log_probs_chosen_old - log_probs_rejected_old
        
        logger.info(f"  - 初始优势均值: {advantages.mean().item():.4f}")
        
        kl_div_before = trainer._compute_kl_divergence(
            trainer.model, trainer.ref_model, input_ids, attention_mask, chosen_ids
        )
        logger.info(f"  - 训练前 KL 散度: {kl_div_before.item():.6f}")
        
        logger.info("✓ 训练循环模拟成功")
        logger.info(f"  - 训练轮数: {num_epochs}")
        logger.info(f"  - 批次大小: {config.per_device_train_batch_size}")
        logger.info(f"  - 样本数量: {len(dataset)}")
        logger.info(f"  - 序列长度: {config.max_seq_length}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 训练循环失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference():
    """测试 11: 推理功能"""
    logger = setup_logger("test_inference")
    logger.info("\n" + "=" * 60)
    logger.info("测试 11: 推理功能")
    logger.info("=" * 60)
    
    try:
        config = TRPOTrainingConfig(
            model_name_or_path=LOCAL_MODEL_PATH,
            max_seq_length=128,
        )
        
        trainer = TRPOTrainer(config=config)
        trainer._setup_accelerator()
        trainer._load_tokenizer()
        trainer._load_model()
        
        prompt = "请计算：15 + 27 = "
        
        inputs = trainer.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(trainer.accelerator.device) for k, v in inputs.items()}
        
        trainer.model.eval()
        with torch.no_grad():
            outputs = trainer.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=trainer.tokenizer.eos_token_id
            )
        
        generated_text = trainer.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info("✓ 推理功能测试成功")
        logger.info(f"  - 输入: {prompt}")
        logger.info(f"  - 生成: {generated_text}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_real_data():
    """测试 12: 使用真实 TRPO 数据训练"""
    logger = setup_logger("test_real_data")
    logger.info("\n" + "=" * 60)
    logger.info("测试 12: 使用真实 TRPO 数据")
    logger.info("=" * 60)
    
    try:
        data_path = "/Users/xiniuyiliao/Desktop/code/Chinese-MedQA-Qwen2/src/training/data/trpo_data/trpo_train.json"
        
        if not os.path.exists(data_path):
            logger.warning(f"数据文件不存在，跳过此测试: {data_path}")
            return True
        
        dataset = load_trpo_dataset(data_path)
        
        config = TRPOTrainingConfig(
            model_name_or_path=LOCAL_MODEL_PATH,
            max_seq_length=256,
            learning_rate=1e-4,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            max_kl_div=0.01,
            cg_damping=0.1,
            cg_iterations=10,
            gamma=0.99,
            gae_lambda=0.95,
            output_dir="./test_output/trpo_real_data",
        )
        
        trainer = TRPOTrainer(config=config)
        trainer._setup_accelerator()
        trainer._load_tokenizer()
        trainer._load_model()
        trainer._load_reference_model()
        
        prepared_data = prepare_trpo_dataset(dataset, trainer.tokenizer, max_length=256)
        
        logger.info("✓ 真实数据测试准备完成")
        logger.info(f"  - 数据路径: {data_path}")
        logger.info(f"  - 样本数量: {len(dataset)}")
        logger.info(f"  - 最大序列长度: {config.max_seq_length}")
        
        sample_prompts = [prepared_data["prompt"][i] for i in range(min(3, len(dataset)))]
        logger.info(f"\n  样本提示词示例:")
        for i, p in enumerate(sample_prompts):
            logger.info(f"    {i+1}. {p[:50]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 真实数据测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lora_finetuning():
    """测试 13: LoRA 微调"""
    logger = setup_logger("test_lora")
    logger.info("\n" + "=" * 60)
    logger.info("测试 13: LoRA 微调")
    logger.info("=" * 60)
    
    try:
        config = TRPOTrainingConfig(
            model_name_or_path=LOCAL_MODEL_PATH,
            max_seq_length=128,
            learning_rate=1e-4,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            max_kl_div=0.01,
            cg_damping=0.1,
            cg_iterations=10,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            lora_target_modules=["c_attn", "c_proj"],
            output_dir="./test_output/trpo_lora",
        )
        
        trainer = TRPOTrainer(
            config=config,
            finetuning_type="lora"
        )
        
        trainer._setup_accelerator()
        trainer._load_tokenizer()
        trainer._load_model()
        
        logger.info("✓ LoRA 微调初始化成功")
        logger.info(f"  - LoRA r: {config.lora_r}")
        logger.info(f"  - LoRA alpha: {config.lora_alpha}")
        logger.info(f"  - LoRA dropout: {config.lora_dropout}")
        logger.info(f"  - 目标模块: {config.lora_target_modules}")
        
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in trainer.model.parameters())
        logger.info(f"  - 可训练参数占比: {trainable_params/total_params*100:.2f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ LoRA 微调测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_validation():
    """测试 14: 配置验证"""
    logger = setup_logger("test_config_validation")
    logger.info("\n" + "=" * 60)
    logger.info("测试 14: 配置验证")
    logger.info("=" * 60)
    
    tests_passed = 0
    tests_total = 0
    
    test_cases = [
        ("max_kl > 0", lambda: TRPOTrainingConfig(
            model_name_or_path=LOCAL_MODEL_PATH,
            max_seq_length=128,
            max_kl_div=0.01
        )),
        ("gamma in (0,1]", lambda: TRPOTrainingConfig(
            model_name_or_path=LOCAL_MODEL_PATH,
            max_seq_length=128,
            gamma=0.99
        )),
        ("gae_lambda in (0,1]", lambda: TRPOTrainingConfig(
            model_name_or_path=LOCAL_MODEL_PATH,
            max_seq_length=128,
            gae_lambda=0.95
        )),
    ]
    
    for name, config_factory in test_cases:
        tests_total += 1
        try:
            config = config_factory()
            logger.info(f"  ✓ {name}: 通过")
            tests_passed += 1
        except Exception as e:
            logger.error(f"  ✗ {name}: 失败 - {e}")
    
    invalid_cases = [
        ("max_kl <= 0", lambda: TRPOTrainingConfig(
            model_name_or_path=LOCAL_MODEL_PATH,
            max_seq_length=128,
            max_kl=-0.01
        )),
        ("gamma <= 0", lambda: TRPOTrainingConfig(
            model_name_or_path=LOCAL_MODEL_PATH,
            max_seq_length=128,
            gamma=0
        )),
    ]
    
    for name, config_factory in invalid_cases:
        tests_total += 1
        try:
            config = config_factory()
            logger.error(f"  ✗ {name}: 应该有异常但没有")
        except ValueError:
            logger.info(f"  ✓ {name}: 正确抛出异常")
            tests_passed += 1
        except Exception as e:
            logger.error(f"  ✗ {name}: 异常类型错误 - {e}")
    
    logger.info(f"✓ 配置验证测试完成: {tests_passed}/{tests_total} 通过")
    return tests_passed == tests_total


def cleanup():
    """清理测试输出"""
    logger = setup_logger("cleanup")
    logger.info("\n" + "=" * 60)
    logger.info("清理测试输出")
    logger.info("=" * 60)
    
    test_dirs = [
        "./test_output/trpo_test",
        "./test_output/trpo_lora",
        "./test_output/trpo_real_data",
    ]
    
    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            try:
                import shutil
                shutil.rmtree(dir_path)
                logger.info(f"  ✓ 已清理: {dir_path}")
            except Exception as e:
                logger.error(f"  ✗ 清理失败 {dir_path}: {e}")


def main():
    """主函数"""
    logger = setup_logger("main")
    logger.info("=" * 60)
    logger.info("TRPOTrainer 测试套件")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    all_passed = True
    
    tests = [
        ("初始化测试", test_initialization),
        ("Tokenizer 加载测试", test_tokenizer_loading),
        ("模型加载测试", test_model_loading),
        ("参考模型测试", test_reference_model),
        ("数据准备测试", lambda: test_data_preparation() is not None),
        ("KL 散度测试", test_kl_divergence),
        ("GAE 计算测试", test_gae_computation),
        ("共轭梯度法测试", test_conjugate_gradient),
        ("对数概率测试", test_log_probability),
        ("训练循环测试", test_full_training_loop),
        ("推理测试", test_inference),
        ("真实数据测试", test_with_real_data),
        ("LoRA 微调测试", test_lora_finetuning),
        ("配置验证测试", test_config_validation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results[test_name] = passed
            if not passed:
                all_passed = False
        except Exception as e:
            logger.error(f"测试 '{test_name}' 异常: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
            all_passed = False
    
    logger.info("\n" + "=" * 60)
    logger.info("测试总结")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        logger.info(f"  {status}: {test_name}")
    
    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    logger.info(f"\n总计: {passed_count}/{total_count} 测试通过")
    
    if all_passed:
        logger.info("\n✓ 所有测试通过！")
    else:
        logger.info("\n✗ 部分测试失败")
    
    logger.info(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
