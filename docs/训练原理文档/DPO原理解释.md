# Direct Preference Optimization（DPO）原理详解

## 一、背景与动机

### 1.1 大模型对齐的重要性

大型语言模型（LLM）在预训练阶段学习了海量的世界知识，具备了强大的语言理解和生成能力。然而，预训练模型直接用于实际应用时往往存在以下问题：

- **回答风格不可控**：可能过于简短、机械，或不符合人类交流习惯
- **价值观偏差**：可能生成有害、偏见或不当内容
- **指令遵循能力弱**：难以准确理解和执行用户复杂指令
- **安全性问题**：可能被诱导生成恶意内容

因此，在模型训练的最后阶段需要一个"对齐"（Alignment）过程，使模型输出符合人类价值观、偏好和实际需求。这个过程也被称为"人类偏好对齐"或"RLHF阶段"。

### 1.2 传统 RLHF 的局限性

传统的人类反馈强化学习（Reinforcement Learning from Human Feedback，RLHF）方法由以下三个阶段组成：

**阶段一：监督微调（Supervised Fine-Tuning，SFT）**
- 使用高质量标注数据对预训练模型进行微调
- 让模型学会遵循指令和生成符合格式的文本
- 数据形式：`(prompt, response)` 对

**阶段二：奖励模型训练（Reward Model Training）**
- 收集人类偏好标注数据，形式为 `(prompt, chosen_response, rejected_response)`
- 训练一个奖励模型（Reward Model）来预测人类对回答的偏好程度
- 奖励模型通常是一个分类器，输出回答被评为"好"的概率

**阶段三：策略优化（Policy Optimization）**
- 使用强化学习算法（通常是 PPO，Proximal Policy Optimization）优化原模型
- 目标：最大化奖励模型给出的分数，同时约束模型不要偏离 SFT 模型太远
- 需要引入 KL 散度惩罚项，防止策略突变

**RLHF 的主要问题：**

| 问题类型 | 具体表现 |
|---------|---------|
| **训练复杂** | 需要管理多个模型（策略模型、价值模型、奖励模型、参考模型） |
| **计算资源密集** | PPO 需要大量的策略采样和价值估计 |
| **训练不稳定** | 强化学习天然存在方差大、波动明显的问题 |
| **超参数敏感** | KL 系数、PPO -clip 参数等难以调优 |
| **奖励黑客攻击** | 模型可能学会"欺骗"奖励模型而非真正改善输出质量 |

### 1.3 DPO 的革命性突破

2023年，斯坦福大学等机构的研究者提出了 Direct Preference Optimization（直接偏好优化），其核心思想是：**跳过奖励模型的训练环节，直接通过偏好数据优化策略模型**。

DPO 的创新之处在于发现了一个关键的数学等价关系：从 RLHF 的优化目标出发，通过数学推导，可以将复杂的强化学习问题转化为简单的二分类问题（对比"好回答"和"坏回答"）。这使得对齐训练变得像监督学习一样简单直接。

---

## 二、数学原理深度解析

### 2.1 偏好数据的表示

DPO 使用的偏好数据与 RLHF 相同，每条样本包含三元组：

$$\mathcal{D} = \{(x_i, y_w^i, y_l^i)\}_{i=1}^{N}$$

其中：
- $x$：输入提示（prompt），代表用户的问题或指令
- $y_w$：人类偏好的回答（winner/chosen），由标注者评选为更好的回答
- $y_l$：人类不偏好的回答（loser/rejected），由标注者评选为较差的回答

### 2.2 从 RLHF 到 DPO 的数学推导

**RLHF 的优化目标**

标准 RLHF 使用带 KL 散度约束的 PPO 优化：

$$\max_{\pi_\theta} \mathbb{E}_{(x,y)\sim\pi_\theta} [r(x,y)] - \beta \cdot \text{KL}(\pi_\theta(\cdot|x) || \pi_{\text{ref}}(\cdot|x))$$

其中：
- $\pi_\theta$：当前策略模型（正在训练的模型）
- $\pi_{\text{ref}}$：参考模型（通常是 SFT 后的模型，参数固定）
- $r(x,y)$：奖励函数（由奖励模型给出）
- $\beta$：KL 散度的权重系数

**引入配分函数**

对于任意奖励函数 $r(x,y)$，我们可以定义一个"奖励对齐"的隐式概率分布：

$$P^*(y|x) = \frac{\exp(r(x,y))}{\sum_{y'}\exp(r(x,y'))}$$

这个形式类似于 softmax 归一化，其中分母 $Z(x) = \sum_{y'}\exp(r(x,y'))$ 被称为**配分函数**（Partition Function）。

**关键洞察：奖励函数的等价表示**

研究者发现，DPO 使用的偏好损失函数可以直接推导出奖励函数的等价形式：

$$r(x,y) = \beta \cdot \log\left(\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\right)$$

这个等式的含义非常深刻：**回答的"奖励分数"等价于该回答在当前策略下的概率与在参考策略下概率的比值的对数**。

**推导偏好概率**

基于上述奖励函数定义，两个回答的偏好概率比为：

$$\frac{P(y_w \succ y_l | x)}{P(y_l \succ y_w | x)} = \frac{\exp(r(x,y_w))}{\exp(r(x,y_l))} = \exp(r(x,y_w) - r(x,y_l))$$

代入奖励函数的等价表示：

$$= \exp\left(\beta \cdot \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \cdot \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

$$= \left(\frac{\pi_\theta(y_w|x) / \pi_{\text{ref}}(y_w|x)}{\pi_\theta(y_l|x) / \pi_{\text{ref}}(y_l|x)}\right)^\beta$$

### 2.3 DPO 损失函数

**交叉熵损失形式**

将偏好问题建模为二分类问题，使用交叉熵损失：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma(\beta \cdot (\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}))\right]$$

其中 $\sigma(z) = \frac{1}{1+e^{-z}}$ 是 sigmoid 函数。

**简化的直观理解**

损失函数可以简化为：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma(\beta \cdot (\Delta_{\text{policy}} - \Delta_{\text{ref}}))\right]$$

其中：
- $\Delta_{\text{policy}} = \log\pi_\theta(y_w|x) - \log\pi_\theta(y_l|x)$：当前策略对好回答 vs 坏回答的对数概率差
- $\Delta_{\text{ref}} = \log\pi_{\text{ref}}(y_w|x) - \log\pi_{\text{ref}}(y_l|x)$：参考策略对好回答 vs 坏回答的对数概率差

**损失函数的直观解释**

| 情况 | $\Delta_{\text{policy}} - \Delta_{\text{ref}}$ | $\sigma(\cdot)$ | $\log\sigma(\cdot)$ | 梯度方向 |
|------|----------------------------------------------|-----------------|-------------------|---------|
| 当前策略正确偏好好回答 | 越大越好 | 接近1 | 接近0 | 梯度小 |
| 当前策略错误偏好坏回答 | 越小（负） | 接近0 | 负值大 | 梯度大 |

**物理意义**

- 当 $\pi_\theta$ 对 $y_w$ 的概率高于 $\pi_{\text{ref}}$，且对 $y_l$ 的概率低于 $\pi_{\text{ref}}$ 时，损失降低
- $\beta$ 参数控制学习的"激进程度"：$\beta$ 越大，对偏好差异越敏感，训练越保守
- KL 散度约束自然内嵌在概率比的形式中，防止策略偏离参考模型太远

### 2.4 KL 散度的隐式约束

DPO 损失函数的一个重要特性是**隐式包含了 KL 散度约束**。这可以通过以下方式理解：

$$\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} = \text{KL}(\pi_\theta(\cdot|x) || \pi_{\text{ref}}(\cdot|x)) \text{ 的某种展开形式}$$

更精确地说，DPO 的优化目标等价于：

$$\mathcal{L} = -\mathbb{E}\left[\log\sigma(\beta \cdot \Delta)\right] + \beta \cdot \text{KL}(\pi_\theta || \pi_{\text{ref}})$$

这意味着：
- 增大 $\beta$ 会增强 KL 惩罚效果，限制策略偏离参考模型的程度
- 减小 $\beta$ 会使策略更容易偏离参考模型，学习更"激进"的偏好

---

## 三、与其他方法的对比

### 3.1 RLHF（PPO）vs DPO

| 维度 | RLHF（PPO） | DPO |
|------|------------|-----|
| **模型数量** | 4个（Actor、Critic、Reward、Ref） | 2个（Policy、Ref） |
| **奖励模型** | 需要单独训练 | 不需要 |
| **优化算法** | PPO（策略梯度） | 直接梯度下降（交叉熵） |
| **训练稳定性** | 低（强化学习方差大） | 高（类似监督学习） |
| **计算复杂度** | 高（需要大量采样） | 低（单次前向传播） |
| **超参数数量** | 多（clip、gamma、lamda等） | 少（主要是 $\beta$） |
| **奖励黑客攻击风险** | 存在 | 较低 |
| **适用场景** | 复杂奖励结构 | 偏好对齐 |

### 3.2 SFT vs DPO

| 维度 | SFT（监督微调） | DPO（偏好优化） |
|------|----------------|----------------|
| **数据形式** | `(prompt, good_response)` | `(prompt, good_response, bad_response)` |
| **学习目标** | 模仿正确答案 | 区分优劣回答 |
| **知识来源** | 单一正确答案 | 偏好对比信息 |
| **对齐能力** | 有限（只能学到一种答案） | 更强（学到为什么好/为什么差） |
| **幻觉风险** | 较高（可能学到错误模式） | 较低（对比学习更鲁棒） |

### 3.3 DPO 与其他偏好优化方法

| 方法 | 提出时间 | 特点 |
|------|---------|------|
| **RLHF+PPO** | 2022 | 传统方法，功能强但复杂 |
| **DPO** | 2023 | 简单稳定，无需奖励模型 |
| **KTO** | 2023 | 只需正负样本，不需要成对数据 |
| **IPO** | 2023 | 解决 DPO 的过拟合问题 |
| **GRPO** | 2024 | 群体相对策略优化，无需价值模型 |

---

## 四、实战：DPOTrainer 代码解析

### 4.1 整体架构

```python
class DPOTrainer(BaseTrainer):
    """
    DPO（直接偏好优化）训练器
    
    核心组件：
    1. 策略模型（Policy Model）：正在训练的主模型
    2. 参考模型（Reference Model）：固定的对比基准
    3. DPO 损失计算：核心优化逻辑
    """
```

### 4.2 初始化与配置

```python
def __init__(
    self,
    config: DPOTrainingConfig,
    finetuning_type: str = "lora",
    beta: float = 0.1,
    use_deepspeed: bool = False,
    deepspeed_config: Optional[Union[str, Dict]] = None
):
```

**关键参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `beta` | float | 0.1 | DPO 温度参数，控制偏好学习的强度。<br>β 越大，对偏好差异越敏感，训练越保守 |
| `finetuning_type` | str | "lora" | 微调类型："lora"（参数高效微调）或 "full"（全参数微调） |
| `use_deepspeed` | bool | False | 是否使用 DeepSpeed 加速（适用于多GPU/大模型） |

**β 参数的影响：**

| β 值 | 效果 | 适用场景 |
|------|------|---------|
| 0.01-0.05 | 激进学习，策略变化大 | 初始对齐、偏好差异明显时 |
| 0.1-0.3 | 平衡（推荐初值） | 大多数场景 |
| 0.5-1.0 | 保守学习，策略变化小 | 精细调优、避免过拟合 |

### 4.3 参考模型的创建与管理

```python
def _create_ref_model(self) -> AutoModelForCausalLM:
    """
    创建参考模型（Reference Model）
    
    参考模型的作用：
    - 提供一个固定的基准，用于计算概率比
    - 确保训练过程中模型不会偏离原始策略太远
    - 不参与梯度计算（requires_grad=False）
    
    实现方式：
    - 深拷贝当前模型参数
    - 移动到相同设备
    - 设置 eval 模式
    """
    ref_model = AutoModelForCausalLM.from_pretrained(
        self.config.model_name_or_path,
        trust_remote_code=self.config.trust_remote_code,
        torch_dtype=self._get_dtype(),
        device_map=self.config.device_map if torch.cuda.is_available() else None,
    )
    
    # 同步参数并冻结
    ref_model.load_state_dict(self.model.state_dict())
    ref_model.requires_grad_(False)
    ref_model.eval()
    
    return ref_model
```

**为什么需要参考模型？**

1. **KL 散度约束**：通过概率比 $\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 隐式限制策略偏离程度
2. **稳定训练**：提供一致的对比基准，避免策略"漂移"
3. **保留能力**：确保模型不会在优化偏好时遗忘预训练知识

**参考模型的同步策略：**

在训练过程中，需要定期同步参考模型和策略模型的参数：

```python
class DPOCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        """每个 epoch 开始时同步参考模型"""
        if self.ref_model is not None:
            model = kwargs.get("model")
            if model is not None:
                self.ref_model.load_state_dict(model.state_dict())
```

### 4.4 DPO 损失计算

```python
def _get_log_probs(
    self,
    logits: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    计算给定 logits 和标签的对数概率
    
    计算过程：
    1. 对 logits 在 vocab 维度上计算 log_softmax
    2. 使用 torch.gather 根据 labels 索引收集对应的对数概率
    
    Args:
        logits: 模型输出的 logits，shape = (batch_size, seq_len, vocab_size)
        labels: 目标标签，shape = (batch_size, seq_len)
    
    Returns:
        每个位置的对数概率，shape = (batch_size, seq_len)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
    return log_probs

def _compute_dpo_loss(
    self,
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_labels: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    计算 DPO 损失函数
    
    损失函数：
    L = -log(σ(β * (Δ_log_prob_chosen - Δ_log_prob_rejected)))
    
    其中：
    - Δ_log_prob_chosen = log(π_policy(y_w|x)) - log(π_ref(y_w|x))
    - Δ_log_prob_rejected = log(π_policy(y_l|x)) - log(π_ref(y_l|x))
    """
    # 1. 计算策略模型的对数概率
    chosen_log_probs = self._get_log_probs(policy_logits, chosen_labels)
    rejected_log_probs = self._get_log_probs(policy_logits, rejected_labels)
    
    # 2. 计算参考模型的对数概率（无梯度）
    with torch.no_grad():
        ref_chosen_log_probs = self._get_log_probs(ref_logits, chosen_labels)
        ref_rejected_log_probs = self._get_log_probs(ref_logits, rejected_labels)
    
    # 3. 计算奖励差异
    policy_chosen_reward = self.beta * (chosen_log_probs - ref_chosen_log_probs)
    policy_rejected_reward = self.beta * (rejected_log_probs - ref_rejected_log_probs)
    
    # 4. 计算损失
    logits = policy_chosen_reward - policy_rejected_reward
    loss = -F.logsigmoid(logits)
    
    # 5. 应用注意力掩码
    if attention_mask is not None:
        loss = (loss * attention_mask).sum() / attention_mask.sum()
    
    return loss
```

**损失计算的物理图像：**

```
y_w (好回答)      y_l (坏回答)
     |                 |
     v                 v
π_θ -----> log_prob_a     π_θ -----> log_prob_b
     |                 |
     v                 v
π_ref ----> ref_log_a    π_ref ---> ref_log_b
     |                 |
     +---- (a - ref_a) - (b - ref_b) ----+
                   |
                   v
              sigmoid(z)
                   |
                   v
              -log(sigmoid(z))
```

### 4.5 偏好数据集处理

```python
def prepare_dataset(
    self,
    dataset: Union[Dataset, List[Dict]],
    prompt_field: str = "prompt",
    chosen_field: str = "chosen",
    rejected_field: str = "rejected",
    max_prompt_length: Optional[int] = None,
    max_response_length: Optional[int] = None
) -> Dataset:
    """
    准备 DPO 偏好数据集
    
    数据格式：
    {
        "prompt": "用户问题",
        "chosen": "偏好的回答",
        "rejected": "不偏好的回答"
    }
    
    处理流程：
    1. 过滤无效样本
    2. 拼接 prompt + response 并分词
    3. 构建输入和标签
    """
```

**数据格式示例：**

```json
{
  "prompt": "我最近经常头痛，早上起床时太阳穴胀痛，是什么原因？",
  "chosen": "晨起头痛可能由多种原因引起：1.紧张性头痛...（详细专业的回答）",
  "rejected": "头痛可能是睡眠不足导致的，建议多休息。（简单敷衍的回答）"
}
```

---

## 五、训练超参数与最佳实践

### 5.1 推荐的超参数配置

| 参数 | 推荐范围 | 说明 |
|------|---------|------|
| **学习率** | 1e-6 ~ 5e-5 | DPO 通常比 SFT 使用更低的学习率 |
| **批次大小** | 4 ~ 16 | 根据显存调整，越大越稳定 |
| **训练轮数** | 1 ~ 5 epochs | DPO 收敛较快，过训练易过拟合 |
| **β** | 0.1 ~ 0.5 | 偏好强度系数，初始推荐 0.1 |
| **warmup_ratio** | 0.1 ~ 0.2 | 学习率预热比例 |
| **max_seq_length** | 1024 ~ 2048 | 根据任务和显存调整 |

### 5.2 学习率与批次大小的关系

增加批次大小通常需要相应增加学习率：

$$\text{new_lr} = \text{base_lr} \times \frac{\text{new_batch_size}}{\text{base_batch_size}}$$

例如，如果批次大小从 4 增加到 16（放大 4 倍），学习率也应相应放大。

### 5.3 β 参数调优策略

| β 值 | 训练行为 | 适用情况 |
|------|---------|---------|
| 0.05 | 学习激进，策略变化大 | 偏好差异小、初始对齐 |
| 0.1 | 平衡（推荐起点） | 大多数场景 |
| 0.3 | 学习保守，变化小 | 精细调优、避免过拟合 |
| 0.5+ | 极其保守 | 防止灾难性遗忘 |

**调优建议：**
- 从 β=0.1 开始
- 如果训练波动大，增加 β
- 如果学习效果不明显，减小 β
- 观察验证集上的偏好准确率（policy 对 chosen 的得分是否高于 rejected）

### 5.4 内存优化技术

**梯度检查点（Gradient Checkpointing）**

```python
if self.config.use_gradient_checkpointing:
    self.model.gradient_checkpointing_enable()
```

以计算换内存，将显存占用降低约 30-50%。

**LoRA 参数高效微调**

```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,                    # LoRA rank
    lora_alpha=16,           # LoRA alpha
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

使用 LoRA 可以将可训练参数量减少 90% 以上。

**DeepSpeed ZeRO 优化**

```json
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 8,
  "optim": {
    "type": "AdamW",
    "params": {
      "lr": 0.0001
    }
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu"
    }
  }
}
```

DeepSpeed ZeRO 可以实现：
- 梯度分片存储
- 优化器状态分片
- 模型参数分片

### 5.5 常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| **损失不下降** | 学习率太低/数据问题 | 提高学习率，检查数据格式 |
| **训练崩溃** | 学习率太高/梯度爆炸 | 降低学习率，添加梯度裁剪 |
| **过拟合** | 训练轮数太多/β太小 | 减少 epochs，增加 β |
| **偏好准确率低** | β设置不当/数据质量差 | 调整 β，检查标注质量 |
| **显存不足** | 模型太大/批次太大 | 启用梯度检查点，使用 LoRA |

---

## 六、数据准备与质量控制

### 6.1 偏好数据的质量要求

**标注原则：**
- **明确偏好**：chosen 和 rejected 之间的差异应该清晰可辨
- **一致性**：相同 prompt 的偏好判断应该一致
- **多样性**：覆盖不同类型的问题和回答风格
- **准确性**：避免标注错误

**数据质量检查：**

```python
def _is_valid_preference_sample(
    self,
    prompt: Any,
    chosen: Any,
    rejected: Any
) -> bool:
    """检查偏好样本的有效性"""
    if not prompt or not chosen or not rejected:
        return False
    
    if not isinstance(prompt, str) or not isinstance(chosen, str) or not isinstance(rejected, str):
        return False
    
    # 检查回答长度差异是否合理
    if len(chosen) < 10 or len(rejected) < 10:
        return False
    
    return True
```

### 6.2 偏好数据的规模建议

| 场景 | 推荐样本数 | 说明 |
|------|-----------|------|
| 演示/测试 | 100-1000 | 验证流程正确性 |
| 轻量级训练 | 1,000-10,000 | 快速迭代实验 |
| 生产训练 | 10,000-100,000 | 标准的偏好对齐数据量 |
| 大规模训练 | 100,000+ | 需要高质量标注 |

### 6.3 医疗领域 DPO 数据的特点

医疗领域的偏好数据有其特殊性：

**chosen 回答应具备：**
- 医学准确性（基于循证医学）
- 完整性（涵盖诊断、治疗、注意事项）
- 专业性（使用规范医学术语）
- 可操作性（给出具体建议）
- 安全性（避免误导性信息）

**rejected 回答的特征：**
- 信息不完整或缺失关键点
- 过于笼统或敷衍
- 可能引起误导的信息
- 缺少必要的警示信息

---

## 七、代码实现注意事项

### 7.1 数值稳定性

```python
# 使用 log_softmax 而不是 softmax + log，避免数值溢出
log_probs = F.log_softmax(logits, dim=-1)

# 使用 logsigmoid 避免数值问题
loss = -F.logsigmoid(logits)
```

### 7.2 分布式训练

```python
def _initialize_accelerator(self):
    """初始化 Accelerator（用于分布式训练）"""
    kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,
        broadcast_buffers=False
    )
    
    self.accelerator = Accelerator(
        kwargs_handlers=[kwargs],
    )
```

### 7.3 模型保存

```python
def _save_model(self, output_dir: str):
    """保存模型"""
    if self.accelerator.is_main_process:
        if self.finetuning_type == "lora":
            self.model.save_pretrained(
                output_dir,
                safe_serialization=True,
                save_adapter_config=True
            )
        else:
            self.trainer.save_model(output_dir)
```

---

## 八、总结与展望

### 8.1 DPO 的核心优势

1. **简洁性**：无需训练奖励模型，将 RLHF 三步简化为一步
2. **稳定性**：使用标准交叉熵损失，训练过程类似监督学习
3. **高效性**：计算资源需求约为 RLHF 的 1/3
4. **效果可靠**：在对话、摘要、代码生成等任务上达到或超越 RLHF

### 8.2 DPO 的局限性与改进方向

| 局限性 | 潜在改进 |
|-------|---------|
| 需要成对偏好数据 | KTO 等方法支持单样本正负反馈 |
| 可能过拟合偏好数据 | IPO 等方法引入正则化 |
| 对 β 参数敏感 | 自适应 β 调整策略 |
| 难以处理复杂奖励结构 | 与其他对齐技术结合使用 |

### 8.3 实践建议

1. **从 SFT 开始**：先用高质量数据做监督微调，建立基础能力
2. **数据质量优先**：偏好数据的质量比数量更重要
3. **从小规模实验**：先用小数据验证流程，再扩展到大规模
4. **监控关键指标**：偏好准确率、KL 散度、损失曲线
5. **渐进式调参**：从保守参数开始，逐步探索最优配置

### 8.4 参考资源

- **原始论文**：Direct Preference Optimization: Your Language Model is Already a Reward Model（Stanford, 2023）
- **开源实现**：TRL（Transformer Reinforcement Learning）库
- **社区实践**：Hugging Face Hub 上的 DPO 微调示例

---

## 附录：关键公式速查表

| 公式 | 含义 |
|------|------|
| $r(x,y) = \beta \log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ | 奖励函数的等价表示 |
| $\mathcal{L}_{\text{DPO}} = -\mathbb{E}[\log\sigma(\beta \cdot \Delta)]$ | DPO 损失函数 |
| $\Delta = (\log\pi_\theta(y_w|x) - \log\pi_\theta(y_l|x)) - (\log\pi_{\text{ref}}(y_w|x) - \log\pi_{\text{ref}}(y_l|x))$ | 偏好差异 |
| $\sigma(z) = \frac{1}{1+e^{-z}}$ | Sigmoid 函数 |
| $\text{KL}(P\|Q) = \sum_i P_i \log\frac{P_i}{Q_i}$ | KL 散度 |
