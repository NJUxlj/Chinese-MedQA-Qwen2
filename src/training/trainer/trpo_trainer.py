import os
import torch
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from datasets import Dataset
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from transformers.trainer_utils import get_last_checkpoint
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import DistributedType
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import setup_logger
from config.training_config import TRPOTrainingConfig




class TRPOTrainer:
    """
    TRPO (Trust-Region Policy Optimization) Trainer
    
    支持：
    1. 全量微调 (Full Fine-tuning)
    2. LoRA 微调 (Low-Rank Adaptation)
    3. 4-bit/8-bit 量化
    4. DeepSpeed 加速
    5. 多卡分布式训练 (Accelerate)
    6. LoRA adapter 保存和合并
    """
    
    def __init__(
        self,
        config: TRPOTrainingConfig,
        finetuning_type: str = "lora",
        use_deepspeed: bool = False,
        deepspeed_config: Optional[Dict] = None
    ):
        """
        初始化 TRPO 训练器
        
        Args:
            config: 训练配置
            finetuning_type: 微调类型 ("lora" 或 "full")
            use_deepspeed: 是否使用 DeepSpeed
            deepspeed_config: DeepSpeed 配置文件路径或配置字典
        """
        self.config = config
        self.finetuning_type = finetuning_type
        self.use_deepspeed = use_deepspeed
        self.deepspeed_config = deepspeed_config
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.accelerator = None
        self.lora_config = None
        self.ref_model = None
        self.reward_model = None
        self.value_head = None
        
        self.logger = setup_logger(self.__class__.__name__, level="INFO")
        
        self._validate_config()
    
    def _validate_config(self):
        """验证配置"""
        if self.finetuning_type not in ["lora", "full"]:
            raise ValueError(f"finetuning_type 必须是 'lora' 或 'full'，但得到了 '{self.finetuning_type}'")
        
        if self.use_deepspeed and self.config.use_4bit:
            self.logger.warning("DeepSpeed 与 4-bit 量化可能存在兼容性问题，请谨慎使用")
        
        if self.config.max_kl <= 0:
            raise ValueError("max_kl 必须大于 0")
        
        if self.config.gamma <= 0 or self.config.gamma > 1:
            raise ValueError("gamma 必须在 (0, 1] 范围内")
        
        if self.config.gae_lambda <= 0 or self.config.gae_lambda > 1:
            raise ValueError("gae_lambda 必须在 (0, 1] 范围内")
    
    def _setup_accelerator(self):
        """设置 Accelerate 分布式训练环境"""
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=False
        )
        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs],
            deepspeed_plugin=self.deepspeed_config if self.use_deepspeed else None
        )
        self.logger.info(f"Accelerator 设备: {self.accelerator.device}")
        if self.accelerator.is_main_process:
            self.logger.info("分布式训练初始化完成")
    
    def _load_tokenizer(self):
        """加载 tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,
            use_fast=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.logger.info(f"Tokenizer 加载完成，词汇表大小: {len(self.tokenizer)}")
    
    def _load_model(self):
        """加载模型"""
        if self.config.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif self.config.use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            quantization_config = None
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if self.config.use_4bit or self.config.use_8bit else torch.float32,
            device_map="auto" if (self.config.use_4bit or self.config.use_8bit) else None,
            trust_remote_code=True
        )
        
        if not (self.config.use_4bit or self.config.use_8bit):
            self.model = self.model.to(self.accelerator.device if self.accelerator else self.config.device)
        
        if self.finetuning_type == "lora":
            self.lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules
            )
            self.model = get_peft_model(self.model, self.lora_config)
            self.model.print_trainable_parameters()
        
        self.logger.info(f"模型加载完成，微调类型: {self.finetuning_type}")
    
    def _load_reference_model(self):
        """加载参考模型（用于 KL 散度计算）"""
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.logger.info("参考模型加载完成")
    
    def _load_reward_model(self, reward_model_path: str):
        """加载奖励模型
        
        Args:
            reward_model_path: 奖励模型路径
        """
        self.reward_model = AutoModelForCausalLM.from_pretrained(
            reward_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.reward_model.eval()
        for param in self.reward_model.parameters():
            param.requires_grad = False
        self.logger.info("奖励模型加载完成")
    
    def _init_value_head(self):
        """初始化价值头（用于critic网络）"""
        hidden_size = self.model.config.hidden_size
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        self.value_head = self.value_head.to(self.accelerator.device if self.accelerator else self.config.device)
        self.logger.info("价值头初始化完成")
    
    def _compute_log_probs(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_ids: torch.Tensor
    ) -> torch.Tensor:
        """计算响应部分的对数概率
        
        Args:
            model: 语言模型
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            response_ids: 响应 token IDs
        
        Returns:
            response 部分每个位置的对数概率
        """
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            response_start = attention_mask.sum(dim=1) - response_ids.size(1)
            log_probs = torch.log_softmax(logits, dim=-1)
            
            log_probs_response = []
            for i in range(logits.size(0)):
                start = response_start[i].item()
                response_log_probs = []
                for t in range(response_ids.size(1)):
                    token_id = response_ids[i, t].item()
                    lp = log_probs[i, start + t, token_id].item()
                    response_log_probs.append(lp)
                log_probs_response.append(response_log_probs)
            
            return torch.tensor(log_probs_response, device=logits.device)
    
    def _compute_kl_divergence(
        self,
        model: torch.nn.Module,
        ref_model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_ids: torch.Tensor
    ) -> torch.Tensor:
        """计算当前模型与参考模型之间的 KL 散度
        
        Args:
            model: 当前策略模型
            ref_model: 参考模型
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            response_ids: 响应 token IDs
        
        Returns:
            平均 KL 散度
        """
        model.eval()
        ref_model.eval()
        
        with torch.no_grad():
            ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits
            ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
        
        model.train()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        
        response_start = attention_mask.sum(dim=1) - response_ids.size(1)
        kl_list = []
        
        for i in range(logits.size(0)):
            start = response_start[i].item()
            p = log_probs[i, start:start + response_ids.size(1)]
            q = ref_log_probs[i, start:start + response_ids.size(1)]
            kl = torch.nn.functional.kl_div(
                p, q,
                reduction='none',
                log_target=False
            ).sum(dim=-1)
            kl_list.append(kl)
        
        return torch.stack(kl_list).mean()
    
    def _compute_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_ids: torch.Tensor
    ) -> torch.Tensor:
        """计算奖励分数
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            response_ids: 响应 token IDs
        
        Returns:
            每个样本的奖励分数
        """
        self.reward_model.eval()
        
        with torch.no_grad():
            rewards = []
            for i in range(input_ids.size(0)):
                full_ids = torch.cat([input_ids[i], response_ids[i]], dim=0)
                full_mask = torch.cat([attention_mask[i], torch.ones_like(response_ids[i])], dim=0)
                
                outputs = self.reward_model(input_ids=full_ids.unsqueeze(0), attention_mask=full_mask.unsqueeze(0))
                reward = outputs.logits[0, -1].item()
                rewards.append(reward)
        
        return torch.tensor(rewards, device=input_ids.device)
    
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> tuple:
        """计算广义优势估计 (Generalized Advantage Estimation)
        
        Args:
            rewards: 奖励序列 [batch_size, seq_len]
            values: 价值估计序列 [batch_size, seq_len]
            masks: 有效位置掩码 [batch_size, seq_len]
            gamma: 折扣因子
            gae_lambda: GAE lambda 参数
        
        Returns:
            advantages: 优势估计 [batch_size, seq_len]
            returns: 回报估计 [batch_size, seq_len]
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        batch_size, seq_len = rewards.size()
        
        for i in range(batch_size):
            gae = 0
            returns_i = 0
            for t in reversed(range(seq_len)):
                if masks[i, t] == 0:
                    continue
                mask = 1 if t == seq_len - 1 or masks[i, t + 1] == 0 else gamma
                delta = rewards[i, t] + gamma * values[i, t + 1] * mask - values[i, t]
                gae = delta + gamma * gae_lambda * mask * gae
                advantages[i, t] = gae
                returns_i = advantages[i, t] + values[i, t]
                returns[i, t] = returns_i
        
        return advantages, returns
    
    def _conjugate_gradient(
        self,
        Ax: callable,
        b: torch.Tensor,
        max_iter: int = 10,
        tol: float = 1e-10
    ) -> torch.Tensor:
        """共轭梯度法求解线性方程 Ax = b
        
        TRPO 使用共轭梯度法来近似求解 Fisher 信息矩阵的逆与梯度的乘积，
        避免直接计算和存储大的 Hessian 矩阵。
        
        Args:
            Ax: 矩阵向量乘法的函数 (输入向量 -> 输出向量)
            b: 右端向量
            max_iter: 最大迭代次数
            tol: 收敛 tolerance
        
        Returns:
            方程的解向量 x
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        
        rsold = torch.dot(r, r)
        
        for _ in range(max_iter):
            Ap = Ax(p)
            
            alpha = rsold / torch.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            
            rsnew = torch.dot(r, r)
            
            if torch.sqrt(rsnew) < tol:
                break
            
            beta = rsnew / rsold
            p = r + beta * p
            p = p.to(device=x.device)
            
            rsold = rsnew
        
        return x
    
    def _fisher_vector_product(
        self,
        params: torch.nn.Parameter,
        damp: float = 0.1
    ) -> callable:
        """构建 Fisher 信息矩阵向量乘积函数
        
        Args:
            params: 模型参数
            damp: 阻尼系数，用于数值稳定性
        
        Returns:
            Fisher 向量乘积函数
        """
        def fisher_vector_product(v):
            params_grad = torch.autograd.grad(
                self._compute_kl_divergence(
                    self.model,
                    self.ref_model,
                    self.current_input_ids,
                    self.current_attention_mask,
                    self.current_response_ids
                ),
                params,
                create_graph=True,
                retain_graph=True
            )
            
            flat_grad = torch.cat([grad.view(-1) for grad in params_grad])
            
            fisher_product = torch.autograd.grad(
                torch.dot(flat_grad, v),
                params,
                create_graph=False,
                retain_graph=True
            )
            
            flat_fisher = torch.cat([grad.view(-1) for grad in fisher_product])
            
            return flat_fisher + damp * v
        
        return fisher_vector_product
    
    def _line_search(
        self,
        policy_update: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        max_backtracks: int = 10,
        reduction_factor: float = 0.5,
        accept_ratio: float = 0.1
    ) -> bool:
        """线搜索找到合适的步长
        
        在找到自然梯度方向后，使用线搜索确定步长，
        确保更新后的策略满足 KL 散度约束。
        
        Args:
            policy_update: 参数量更新方向
            old_log_probs: 更新前的对数概率
            advantages: 优势估计
            max_backtracks: 最大回溯次数
            reduction_factor: 步长缩减因子
            accept_ratio: 接受更新的奖励比例阈值
        
        Returns:
            是否成功找到可接受的步长
        """
        current_params = [p.clone() for p in self.model.parameters()]
        
        old_loss = self._compute_policy_loss(old_log_probs, advantages)
        
        step_size = 1.0
        
        for _ in range(max_backtracks):
            for i, param in enumerate(self.model.parameters()):
                param.data = current_params[i] + step_size * policy_update[i]
            
            with torch.no_grad():
                new_log_probs = self._compute_log_probs(
                    self.model,
                    self.current_input_ids,
                    self.current_attention_mask,
                    self.current_response_ids
                )
            
            kl_div = torch.mean(torch.exp(new_log_probs - old_log_probs) - 1 - new_log_probs + old_log_probs)
            
            if kl_div.item() < self.config.max_kl * 1.5:
                new_loss = self._compute_policy_loss(new_log_probs, advantages)
                improvement = old_loss - new_loss
                
                if improvement > 0 or accept_ratio > 0.1:
                    return True
            
            step_size *= reduction_factor
        
        for i, param in enumerate(self.model.parameters()):
            param.data = current_params[i]
        
        return False
    
    def _compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor
    ) -> torch.Tensor:
        """计算策略损失
        
        Args:
            log_probs: 对数概率
            advantages: 优势估计
        
        Returns:
            策略损失
        """
        policy_loss = -torch.mean(log_probs * advantages)
        return policy_loss
    
    def _update_policy(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_ids: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor
    ) -> Dict[str, float]:
        """执行 TRPO 策略更新
        
        TRPO 的核心更新步骤：
        1. 计算策略梯度
        2. 使用共轭梯度法计算自然梯度方向
        3. 使用线搜索确定步长
        4. 更新策略参数
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            response_ids: 响应 token IDs
            old_log_probs: 更新前的对数概率
            advantages: 优势估计
        
        Returns:
            更新统计信息
        """
        self.model.train()
        self.ref_model.eval()
        
        self.current_input_ids = input_ids
        self.current_attention_mask = attention_mask
        self.current_response_ids = response_ids
        
        params = list(self.model.parameters())
        
        def kl_divergence_fn():
            return self._compute_kl_divergence(
                self.model,
                self.ref_model,
                input_ids,
                attention_mask,
                response_ids
            )
        
        kl_before = kl_divergence_fn()
        
        flat_params = torch.cat([p.view(-1) for p in params])
        flat_grad = torch.autograd.grad(
            kl_before,
            params,
            create_graph=True,
            retain_graph=True
        )
        flat_grad = torch.cat([g.view(-1) for g in flat_grad])
        
        def Ax(v):
            fisher_grad = torch.autograd.grad(
                torch.dot(flat_grad, v),
                params,
                create_graph=False,
                retain_graph=True
            )
            flat_fisher = torch.cat([g.view(-1) for g in fisher_grad])
            return flat_fisher + self.config.cg_damping * v
        
        natural_gradient = self._conjugate_gradient(Ax, flat_grad, max_iter=self.config.cg_iterations)
        
        natural_gradient = natural_gradient / (torch.norm(natural_gradient) + 1e-8) * self.config.max_kl
        
        policy_update = []
        idx = 0
        for param in params:
            numel = param.numel()
            policy_update.append(natural_gradient[idx:idx + numel].view(param.shape))
            idx += numel
        
        self.model.zero_grad()
        
        with torch.no_grad():
            old_loss = self._compute_policy_loss(old_log_probs, advantages)
        
        success = self._line_search(policy_update, old_log_probs, advantages)
        
        if not success:
            self.logger.warning("线搜索未能找到有效步长，跳过此次更新")
            return {"success": False, "kl_before": kl_before.item(), "status": "line_search_failed"}
        
        kl_after = kl_divergence_fn()
        
        self.model.zero_grad()
        
        return {
            "success": True,
            "kl_before": kl_before.item(),
            "kl_after": kl_after.item(),
            "status": "updated"
        }
    
    def _update_value_function(
        self,
        states: torch.Tensor,
        attention_mask: torch.Tensor,
        returns: torch.Tensor,
        epochs: int = 5,
        lr: float = 1e-3
    ) -> float:
        """更新价值函数（Critic）
        
        Args:
            states: 状态序列
            attention_mask: 注意力掩码
            returns: 回报估计
            epochs: 训练轮数
            lr: 学习率
        
        Returns:
            价值损失
        """
        self.value_head.train()
        
        optimizer = torch.optim.Adam(self.value_head.parameters(), lr=lr)
        
        total_loss = 0
        
        for _ in range(epochs):
            optimizer.zero_grad()
            
            with torch.no_grad():
                last_hidden_state = self.model(
                    input_ids=states,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                ).hidden_states[-1]
            
            values = self.value_head(last_hidden_state).squeeze(-1)
            
            mask = attention_mask.bool()
            value_loss = torch.nn.functional.mse_loss(
                values[mask],
                returns[mask],
                reduction='mean'
            )
            
            value_loss.backward()
            optimizer.step()
            
            total_loss += value_loss.item()
        
        return total_loss / epochs
    
    def _prepare_batch(self, batch: Dict) -> Dict:
        """准备训练批次
        
        Args:
            batch: 原始数据批次
        
        Returns:
            处理后的批次
        """
        input_ids = batch["input_ids"].to(self.accelerator.device if self.accelerator else self.config.device)
        attention_mask = batch["attention_mask"].to(self.accelerator.device if self.accelerator else self.config.device)
        
        if "response_ids" in batch:
            response_ids = batch["response_ids"].to(self.accelerator.device if self.accelerator else self.config.device)
        else:
            response_length = self.config.response_length
            response_ids = input_ids[:, -response_length:]
            attention_mask_for_response = torch.ones_like(response_ids)
            attention_mask = torch.cat([attention_mask, attention_mask_for_response], dim=1)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_ids": response_ids
        }
    
    def _generate_responses(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """使用当前策略生成响应
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
        
        Returns:
            生成的响应 token IDs
        """
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.response_length,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                device=self.accelerator.device if self.accelerator else None
            )
        
        response_start = attention_mask.sum(dim=1)
        responses = []
        for i in range(outputs.size(0)):
            response = outputs[i, response_start[i]:]
            if response.size(0) < self.config.response_length:
                padding = torch.full(
                    (self.config.response_length - response.size(0),),
                    self.tokenizer.eos_token_id,
                    device=outputs.device
                )
                response = torch.cat([response, padding])
            responses.append(response[:self.config.response_length])
        
        return torch.stack(responses)
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: Optional[str] = None,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate_actor: float = 1e-5,
        learning_rate_critic: float = 1e-3,
        max_kl: float = 0.01,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ppo_epochs: int = 4,
        log_interval: int = 1,
        save_interval: int = 5,
        **kwargs
    ):
        """执行 TRPO 训练
        
        Args:
            train_dataset: 训练数据集
            eval_dataset: 评估数据集
            output_dir: 输出目录
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate_actor: Actor (策略) 学习率
            learning_rate_critic: Critic (价值) 学习率
            max_kl: 最大 KL 散度
            gamma: 折扣因子
            gae_lambda: GAE lambda 参数
            ppo_epochs: PPO/TRPO 更新次数
            log_interval: 日志打印间隔
            save_interval: 模型保存间隔
        """
        self._setup_accelerator()
        self._load_tokenizer()
        self._load_model()
        self._init_value_head()
        
        if self.accelerator.is_main_process:
            self.logger.info("=" * 60)
            self.logger.info("开始 TRPO 训练")
            self.logger.info(f"训练数据集大小: {len(train_dataset)}")
            self.logger.info(f"批大小: {batch_size}")
            self.logger.info(f"训练轮数: {epochs}")
            self.logger.info("=" * 60)
        
        self.model, self.value_head = self.accelerator.prepare(
            self.model, self.value_head
        )
        
        optimizer = torch.optim.Adam([
            {"params": self.model.parameters(), "lr": learning_rate_actor},
            {"params": self.value_head.parameters(), "lr": learning_rate_critic}
        ])
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: self._collate_fn(x),
            num_workers=4,
            pin_memory=True
        )
        
        train_loader = self.accelerator.prepare(train_loader)
        
        global_step = 0
        best_eval_reward = float('-inf')
        
        for epoch in range(epochs):
            self.model.eval()
            self.value_head.eval()
            self.ref_model.eval()
            
            epoch_rewards = []
            epoch_kl = []
            
            for step, batch in enumerate(train_loader):
                batch = self._prepare_batch(batch)
                
                response_ids = self._generate_responses(
                    batch["input_ids"],
                    batch["attention_mask"]
                )
                
                batch["response_ids"] = response_ids
                
                with torch.no_grad():
                    rewards = self._compute_rewards(
                        batch["input_ids"],
                        batch["attention_mask"],
                        response_ids
                    )
                    
                    ref_log_probs = self._compute_log_probs(
                        self.ref_model,
                        batch["input_ids"],
                        batch["attention_mask"],
                        response_ids
                    )
                
                self.model.train()
                self.value_head.train()
                
                with torch.enable_grad():
                    for ppo_step in range(ppo_epochs):
                        current_log_probs = self._compute_log_probs(
                            self.model,
                            batch["input_ids"],
                            batch["attention_mask"],
                            response_ids
                        )
                        
                        outputs = self.model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            output_hidden_states=True
                        )
                        
                        last_hidden = outputs.hidden_states[-1]
                        values = self.value_head(last_hidden).squeeze(-1)
                        
                        response_start = batch["attention_mask"].sum(dim=1) - response_ids.size(1)
                        values_response = []
                        for i in range(values.size(0)):
                            start = response_start[i].item()
                            values_response.append(values[i, start:start + response_ids.size(1)])
                        values_response = torch.stack(values_response)
                        
                        response_masks = torch.ones_like(response_ids, dtype=torch.float32)
                        
                        advantages, returns = self._compute_gae(
                            rewards.unsqueeze(-1).expand_as(values_response),
                            values_response,
                            response_masks,
                            gamma=gamma,
                            gae_lambda=gae_lambda
                        )
                        
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                        
                        policy_result = self._update_policy(
                            batch["input_ids"],
                            batch["attention_mask"],
                            response_ids,
                            current_log_probs,
                            advantages
                        )
                        
                        if policy_result.get("success", False):
                            states_for_value = torch.cat([
                                batch["input_ids"],
                                response_ids
                            ], dim=1)
                            masks_for_value = torch.cat([
                                batch["attention_mask"],
                                response_masks
                            ], dim=1)
                            
                            self._update_value_function(
                                states_for_value,
                                masks_for_value,
                                returns,
                                epochs=5,
                                lr=learning_rate_critic
                            )
                        
                        if ppo_step == ppo_epochs - 1:
                            kl = self._compute_kl_divergence(
                                self.model,
                                self.ref_model,
                                batch["input_ids"],
                                batch["attention_mask"],
                                response_ids
                            )
                            epoch_kl.append(kl.item())
                
                epoch_rewards.extend(rewards.cpu().numpy())
                
                global_step += 1
                
                if global_step % log_interval == 0 and self.accelerator.is_main_process:
                    avg_reward = sum(epoch_rewards) / len(epoch_rewards)
                    avg_kl = sum(epoch_kl) / len(epoch_kl) if epoch_kl else 0
                    self.logger.info(
                        f"Epoch {epoch + 1}/{epochs} | "
                        f"Step {global_step} | "
                        f"Avg Reward: {avg_reward:.4f} | "
                        f"Avg KL: {avg_kl:.6f}"
                    )
                
                if global_step % save_interval == 0 and self.accelerator.is_main_process:
                    self.save_model(output_dir or self.config.output_dir, global_step)
            
            if self.accelerator.is_main_process:
                avg_epoch_reward = sum(epoch_rewards) / len(epoch_rewards)
                self.logger.info(f"Epoch {epoch + 1} 完成 | 平均奖励: {avg_epoch_reward:.4f}")
        
        if self.accelerator.is_main_process:
            self.save_model(output_dir or self.config.output_dir, global_step)
            self.logger.info("训练完成!")
    
    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """数据整理函数
        
        Args:
            batch: 批次数据列表
        
        Returns:
            整理后的批次字典
        """
        input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
        attention_mask = torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long)
        
        if "response_ids" in batch[0]:
            response_ids = torch.tensor([item["response_ids"] for item in batch], dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "response_ids": response_ids
            }
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    
    def save_model(self, output_dir: str, step: int = 0):
        """保存模型
        
        Args:
            output_dir: 输出目录
            step: 当前步数
        """
        output_path = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(output_path, exist_ok=True)
        
        if self.finetuning_type == "lora":
            self.model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
        else:
            self.accelerator.unwrap_model(self.model).save_pretrained(
                output_path,
                is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save
            )
            self.tokenizer.save_pretrained(output_path)
        
        if self.accelerator.is_main_process:
            self.logger.info(f"模型已保存到: {output_path}")
    
    def merge_and_unload(self, output_dir: str):
        """合并 LoRA adapter 并保存
        
        Args:
            output_dir: 输出目录
        """
        if self.finetuning_type != "lora":
            self.logger.warning("只有 LoRA 微调支持合并操作")
            return
        
        if isinstance(self.model, PeftModel):
            self.model = self.model.merge_and_unload()
            self.logger.info("LoRA adapter 已合并到基础模型")
        
        self.save_model(output_dir)
    
    def evaluate(self, eval_dataset: Dataset, batch_size: int = 8) -> Dict:
        """评估模型
        
        Args:
            eval_dataset: 评估数据集
            batch_size: 批次大小
        
        Returns:
            评估指标
        """
        self.model.eval()
        self.ref_model.eval()
        
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: self._collate_fn(x)
        )
        
        total_reward = 0
        total_kl = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = self._prepare_batch(batch)
                
                response_ids = self._generate_responses(
                    batch["input_ids"],
                    batch["attention_mask"]
                )
                
                rewards = self._compute_rewards(
                    batch["input_ids"],
                    batch["attention_mask"],
                    response_ids
                )
                
                kl = self._compute_kl_divergence(
                    self.model,
                    self.ref_model,
                    batch["input_ids"],
                    batch["attention_mask"],
                    response_ids
                )
                
                total_reward += rewards.sum().item()
                total_kl += kl.item()
                num_batches += 1
        
        return {
            "eval_reward": total_reward / len(eval_dataset),
            "eval_kl_divergence": total_kl / num_batches
        }
