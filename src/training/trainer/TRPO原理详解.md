# TRPO（信赖域策略优化）算法详解

## 目录

1. [TRPO 简介](#1-trpo-简介)
2. [策略梯度回顾](#2-策略梯度回顾)
3. [TRPO 核心原理](#3-trpo-核心原理)
4. [关键数学推导](#4-关键数学推导)
5. [算法实现细节](#5-算法实现细节)
6. [代码对照说明](#6-代码对照说明)
7. [在语言模型中的应用](#7-在语言模型中的应用)
8. [参考文献](#8-参考文献)

---

## 1. TRPO 简介

### 1.1 什么是 TRPO

TRPO（Trust Region Policy Optimization，信赖域策略优化）是由 John Schulman 等人于 2015 年在论文《Trust Region Policy Optimization》中提出的强化学习优化算法[1]。TRPO 通过限制策略更新的幅度，确保策略在优化过程中的稳定性和单调改进。

### 1.2 TRPO 的核心思想

TRPO 的核心思想可以用一句话概括：**在信赖域内进行策略优化**。

- **信赖域（Trust Region）**：参数空间中的一个区域，在这个区域内，我们对优化问题的近似是可靠的
- **策略优化**：通过调整策略网络的参数来最大化期望回报
- **信任区域约束**：通过 KL 散度限制新旧策略之间的差异，确保每次更新都是"安全"的

### 1.3 为什么需要 TRPO

在深度强化学习中，策略梯度方法面临以下挑战：

1. **样本效率低**：每次更新只能使用少量样本
2. **训练不稳定**：策略更新可能导致性能大幅下降
3. **超参数敏感**：学习率的选择对训练影响很大

TRPO 通过以下方式解决这些问题：

- 使用**自然梯度**替代普通梯度，考虑参数空间的曲率信息
- 通过 KL 散度约束限制更新幅度，保证单调改进
- 使用**共轭梯度法**高效求解优化问题

### 1.4 TRPO 与 PPO 的关系

| 特性 | TRPO | PPO（近端策略优化） |
|------|------|---------------------|
| 约束方式 | KL 散度硬约束 | 概率比剪切 |
| 计算复杂度 | 较高（需要Fisher信息矩阵） | 较低 |
| 实现难度 | 较复杂 | 较简单 |
| 稳定性 | 非常好 | 很好 |
| 适用场景 | 需要稳定收敛的情况 | 需要简单实现的场景 |

PPO 可以看作是 TRPO 的一种简化近似，保留了 TRPO 的核心思想但大幅降低了计算复杂度[2]。

---

## 2. 策略梯度回顾

### 2.1 马尔可夫决策过程（MDP）

强化学习问题通常建模为马尔可夫决策过程：

$$M = (S, A, P, R, \gamma)$$

其中：
- $S$：状态空间
- $A$：动作空间
- $P(s'|s, a)$：状态转移概率
- $R(s, a)$：奖励函数
- $\gamma \in [0, 1]$：折扣因子

### 2.2 策略梯度定理

策略梯度定理给出了期望回报对策略参数的梯度：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A^{\pi_\theta}(s_t, a_t)\right]$$

其中：
- $\pi_\theta$：参数化策略
- $\tau$：轨迹 $(s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$
- $A^{\pi_\theta}(s_t, a_t)$：优势函数

### 2.3 策略梯度的问题

标准策略梯度方法存在以下问题：

1. **更新步长难以确定**：
   - 步长太小：收敛速度慢
   - 步长太大：可能导致策略崩溃

2. **方差较高**：
   - 蒙特卡洛估计本身具有较高方差
   - 需要使用基线（baseline）来降低方差

3. **无法利用曲率信息**：
   - 普通梯度只考虑一阶信息
   - 参数空间的曲率信息对于确定合适的更新步长非常重要

---

## 3. TRPO 核心原理

### 3.1 优化目标

TRPO 的优化目标是：

$$\max_\theta \mathbb{E}_{s \sim \rho_\theta, a \sim \pi_\theta}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} \cdot A_{\theta_{old}}(s, a)\right]$$

$$\text{s.t.} \quad \mathbb{E}_{s \sim \rho_\theta}\left[KL(\pi_{\theta_{old}}(\cdot|s) || \pi_\theta(\cdot|s))\right] \leq \delta$$

其中：
- $\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$：重要性采样比率（概率比）
- $A_{\theta_{old}}(s, a)$：使用旧策略计算的优势
- $KL(\cdot || \cdot)$：KL 散度
- $\delta$：KL 散度上限（信赖域大小）
- $\rho_\theta$：状态访问频率

### 3.2 信赖域的含义

信赖域是参数空间中的一个区域，在这个区域内：

1. **近似可靠**：使用二阶泰勒展开近似目标函数是准确的
2. **约束有效**：KL 散度约束能够有效限制策略变化
3. **单调改进**：每次更新都能保证期望回报不下降

### 3.3 自然梯度

普通梯度：

$$\theta_{new} = \theta + \alpha \cdot \nabla_\theta J(\theta)$$

自然梯度：

$$\theta_{new} = \theta + \alpha \cdot F^{-1} \cdot \nabla_\theta J(\theta)$$

其中 $F$ 是 Fisher 信息矩阵：

$$F = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot \nabla_\theta \log \pi_\theta(a|s)^T]$$

自然梯度的特点：
- 考虑了参数空间的黎曼几何结构
- 对参数化方式不变（坐标无关）
- 更好地反映策略的真实变化程度

### 3.4 共轭梯度法

TRPO 使用共轭梯度法来求解以下问题：

$$F \cdot x = \nabla_\theta J(\theta)$$

其中 $x$ 就是自然梯度方向 $F^{-1} \cdot \nabla_\theta J(\theta)$。

**共轭梯度法的优势**：

1. **避免显式计算 $F$**：直接计算 $F \cdot v$ 而不需要存储整个矩阵
2. **迭代求解**：只需要 $O(n)$ 次迭代（$n$ 是参数数量）
3. **数值稳定**：对于大规模问题更加稳定

**算法步骤**：

```
x = 0
r = b - A*x  # 残差
p = r        # 搜索方向

while ||r|| > tol and iter < max_iter:
    Ap = A*p
    alpha = (r^T * r) / (p^T * Ap)
    x = x + alpha * p
    r = r - alpha * Ap
    beta = (r_new^T * r_new) / (r_old^T * r_old)
    p = r + beta * p
```

### 3.5 线搜索

找到梯度方向后，需要确定合适的步长 $\alpha$：

$$\theta_{new} = \theta + \alpha \cdot x$$

线搜索的目标是找到满足以下条件的最小 $\alpha$：

1. **满足 KL 约束**：$KL(\pi_\theta || \pi_{\theta_{new}}) \leq \delta$
2. **改善目标函数**：$J(\theta_{new}) > J(\theta)$

**回溯线搜索算法**：

```
step_size = 1.0
reduction_factor = 0.5
max_backtracks = 10

for i in range(max_backtracks):
    theta_candidate = theta + step_size * x
    if KL_divergence(theta_candidate) < delta * 1.5:
        if improvement > 0 or accept_ratio > threshold:
            return step_size
    step_size *= reduction_factor
return None  # 未找到合适的步长
```

---

## 4. 关键数学推导

### 4.1 重要性采样

在 TRPO 中，我们使用重要性采样来使用旧策略的数据更新新策略：

$$\mathbb{E}_{\pi_\theta}[f(s, a)] = \mathbb{E}_{\pi_{\theta_{old}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} \cdot f(s, a)\right]$$

这使得我们可以使用离线数据进行更新，提高样本效率。

### 4.2 优势函数的估计

**蒙特卡洛优势估计**：

$$A^{MC}_t = \sum_{l=0}^{\infty} \gamma^l r_{t+l} - V(s_t)$$

**广义优势估计（GAE）**[3]：

GAE 通过调节参数 $\lambda$ 平衡偏差和方差：

$$A^{GAE}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

**递归计算**：

```
gae = 0
for t in reversed(range(T)):
    delta = r[t] + gamma * V[t+1] - V[t]
    gae = delta + gamma * lambda * gae
    A[t] = gae
```

### 4.3 Fisher 信息矩阵

Fisher 信息矩阵定义为：

$$F = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot \nabla_\theta \log \pi_\theta(a|s)^T]$$

性质：

1. **半正定**：$F \succeq 0$
2. **KL 散度的 Hessian**：$F = \nabla^2_\theta KL(\pi_{\theta_{old}} || \pi_\theta) |_{\theta=\theta_{old}}$
3. **信息度量**：$F$ 衡量了策略对参数的敏感程度

### 4.4 共轭梯度与 Fisher 矩阵

在 TRPO 中，我们不需要显式计算 $F$，而是计算 Fisher 向量积：

$$F \cdot v = \nabla_\theta (\nabla_\theta \log \pi_\theta(a|s)^T v)$$

通过两次自动微分即可计算：
1. 第一次计算 $\nabla_\theta KL$
2. 第二次计算 $\nabla_\theta ((\nabla_\theta KL)^T v)$

### 4.5 自然梯度的含义

自然梯度 $F^{-1} \cdot \nabla_\theta J$ 表示在参数空间中"等效"的更新方向：

- 普通梯度：参数空间中的欧几里得距离
- 自然梯度：策略空间中的 KL 距离

这意味着自然梯度保证了策略变化的程度相同，而不管参数的量级如何。

---

## 5. 算法实现细节

### 5.1 TRPO 算法伪代码

```
Initialize: policy πθ, value function Vφ, hyperparameters

for iteration = 1, 2, ...:
    # 收集经验
    rollouts = collect_rollouts(πθ)
    
    # 计算回报和优势
    returns = compute_returns(rollouts.rewards, γ)
    advantages = compute_gae(
        rewards=rollouts.rewards,
        values=rollouts.values,
        γ=γ,
        λ=λ
    )
    
    # 归一化优势
    advantages = (advantages - mean) / (std + ε)
    
    # 更新价值函数
    for _ in epochs_value:
        Vφ = update_value_function(Vφ, rollouts.states, returns)
    
    # 更新策略（TRPO 核心）
    for _ in epochs_policy:
        # 计算梯度
        grads = compute_policy_gradient(πθ, rollouts, advantages)
        
        # Fisher 向量积函数
        def fisher_vector_product(v):
            return compute_fisher_product(πθ, rollouts, v)
        
        # 共轭梯度求解
        natural_gradient = conjugate_gradient(
            A=fisher_vector_product,
            b=grads,
            max_iter=cg_iterations
        )
        
        # 线搜索
        success = line_search(
            πθ,
            natural_gradient,
            max_kl
        )
        
        if not success:
            break
```

### 5.2 关键实现组件

#### 5.2.1 共轭梯度实现

```python
def _conjugate_gradient(self, Ax, b, max_iter=10, tol=1e-10):
    """
    共轭梯度法求解线性方程 Ax = b
    
    Args:
        Ax: 矩阵向量乘法的函数
        b: 右端向量
        max_iter: 最大迭代次数
        tol: 收敛 tolerance
    
    Returns:
        解向量 x
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
        rsold = rsnew
    
    return x
```

#### 5.2.2 Fisher 向量积实现

```python
def _fisher_vector_product(self, params, damp=0.1):
    """
    构建 Fisher 信息矩阵向量乘积函数
    
    Fisher 向量积 = ∇θ(KL) 与向量 v 的二阶导数
    """
    def fisher_vector_product(v):
        # 第一次自动微分：计算 KL 散度的梯度
        kl = self._compute_kl_divergence(...)
        kl_grad = torch.autograd.grad(kl, params, create_graph=True)
        flat_grad = torch.cat([g.view(-1) for g in kl_grad])
        
        # 第二次自动微分：计算 Fisher 向量积
        fisher_product = torch.autograd.grad(
            torch.dot(flat_grad, v),
            params,
            create_graph=False
        )
        flat_fisher = torch.cat([g.view(-1) for g in fisher_product])
        
        # 添加阻尼项提高数值稳定性
        return flat_fisher + damp * v
    
    return fisher_vector_product
```

#### 5.2.3 线搜索实现

```python
def _line_search(self, policy_update, old_log_probs, advantages):
    """
    线搜索找到合适的步长
    
    在找到自然梯度方向后，使用线搜索确定步长，
    确保更新后的策略满足 KL 散度约束。
    """
    current_params = [p.clone() for p in self.model.parameters()]
    step_size = 1.0
    
    for _ in range(max_backtracks):
        # 应用更新
        for i, param in enumerate(self.model.parameters()):
            param.data = current_params[i] + step_size * policy_update[i]
        
        # 检查 KL 约束
        new_log_probs = self._compute_log_probs(...)
        kl_div = torch.mean(
            torch.exp(new_log_probs - old_log_probs) - 1 
            - new_log_probs + old_log_probs
        )
        
        if kl_div < max_kl * 1.5:
            new_loss = self._compute_policy_loss(new_log_probs, advantages)
            improvement = old_loss - new_loss
            
            if improvement > 0:
                return True
        
        # 回溯步长
        step_size *= reduction_factor
    
    # 恢复原始参数
    for i, param in enumerate(self.model.parameters()):
        param.data = current_params[i]
    
    return False
```

#### 5.2.4 GAE 优势估计实现

```python
def _compute_gae(self, rewards, values, masks, gamma=0.99, gae_lambda=0.95):
    """
    计算广义优势估计 (Generalized Advantage Estimation)
    
    GAE 通过参数 λ 平衡 TD 误差的不同估计：
    - λ = 0：高偏差（单步 TD 误差）
    - λ = 1：低方差（蒙特卡洛回报）
    """
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    
    batch_size, seq_len = rewards.size()
    
    for i in range(batch_size):
        gae = 0
        for t in reversed(range(seq_len)):
            if masks[i, t] == 0:
                continue
            
            # TD 误差
            mask = 1 if t == seq_len - 1 or masks[i, t + 1] == 0 else gamma
            delta = rewards[i, t] + gamma * values[i, t + 1] * mask - values[i, t]
            
            # GAE 累积
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[i, t] = gae
            returns[i, t] = advantages[i, t] + values[i, t]
    
    return advantages, returns
```

---

## 6. 代码对照说明

### 6.1 TRPOTrainer 类结构

| 方法名 | 功能 | 对应算法组件 |
|--------|------|--------------|
| `__init__` | 初始化训练器 | - |
| `_setup_accelerator` | 设置分布式训练 | 基础设施 |
| `_load_model` | 加载策略模型 | 策略 πθ |
| `_load_reference_model` | 加载参考模型 | 旧策略 πθ_old |
| `_load_reward_model` | 加载奖励模型 | 奖励函数 R |
| `_init_value_head` | 初始化价值函数 | 价值网络 Vφ |
| `_compute_log_probs` | 计算对数概率 | 策略输出 |
| `_compute_kl_divergence` | 计算 KL 散度 | 信任域约束 |
| `_compute_rewards` | 计算奖励 | 奖励信号 |
| `_compute_gae` | 计算优势 | GAE 估计 |
| `_conjugate_gradient` | 共轭梯度求解 | 自然梯度方向 |
| `_line_search` | 线搜索 | 步长选择 |
| `_update_policy` | 策略更新 | TRPO 更新 |
| `_update_value_function` | 价值函数更新 | Critic 更新 |
| `_generate_responses` | 生成响应 | 经验收集 |
| `train` | 主训练循环 | 整体算法 |

### 6.2 关键配置参数

```python
@dataclass
class TRPOConfig:
    # TRPO 特有参数
    max_kl: float = 0.01          # KL 散度上限（信赖域大小）
    cg_iterations: int = 10       # 共轭梯度迭代次数
    cg_damping: float = 0.1       # Fisher 矩阵阻尼系数
    
    # GAE 参数
    gamma: float = 0.99           # 折扣因子
    gae_lambda: float = 0.95      # GAE lambda 参数
    
    # 优化参数
    learning_rate_actor: float = 1e-5  # 策略学习率
    learning_rate_critic: float = 1e-3 # 价值学习率
    
    # 训练参数
    ppo_epochs: int = 4           # 策略更新轮数
    value_epochs: int = 5         # 价值更新轮数
    batch_size: int = 4           # 批次大小
```

### 6.3 训练流程图

```
┌─────────────────────────────────────────────────────────────┐
│                        TRPO 训练流程                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐    ┌─────────────┐    ┌───────────────────┐   │
│  │ 加载数据 │ -> │ 生成响应    │ -> │ 计算奖励分数      │   │
│  └─────────┘    └─────────────┘    └───────────────────┘   │
│                                              │              │
│                                              v              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    经验收集阶段                      │   │
│  │  (并行执行多次 PPO/TRPO 更新)                       │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │                                                     │   │
│  │  ┌───────────────┐    ┌───────────────────────┐    │   │
│  │  │ 计算对数概率  │ -> │ 计算优势函数 (GAE)    │    │   │
│  │  └───────────────┘    └───────────────────────┘    │   │
│  │                                              │       │   │
│  │                                              v       │   │
│  │  ┌────────────────────────────────────────────────┐│   │
│  │  │              TRPO 策略更新                     ││   │
│  │  ├────────────────────────────────────────────────┤│   │
│  │  │                                                ││   │
│  │  │  1. 计算 KL 散度梯度                           ││   │
│  │  │           |                                    ││   │
│  │  │           v                                    ││   │
│  │  │  2. Fisher 向量积函数                         ││   │
│  │  │           |                                    ││   │
│  │  │           v                                    ││   │
│  │  │  3. 共轭梯度法求解自然梯度                     ││   │
│  │  │           |                                    ││   │
│  │  │           v                                    ││   │
│  │  │  4. 线搜索确定步长                             ││   │
│  │  │           |                                    ││   │
│  │  │           v                                    ││   │
│  │  │  5. 更新策略参数                               ││   │
│  │  │                                                ││   │
│  │  └────────────────────────────────────────────────┘│   │
│  │                                              │       │   │
│  │                                              v       │   │
│  │  ┌────────────────────────────────────────────────┐│   │
│  │  │              更新价值函数 (Critic)             ││   │
│  │  └────────────────────────────────────────────────┘│   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                              │              │
│                                              v              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   日志与保存                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.4 分布式训练支持

TRPOTrainer 使用 Hugging Face Accelerate 库支持分布式训练：

```python
def _setup_accelerator(self):
    """设置 Accelerate 分布式训练环境"""
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=False
    )
    self.accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        deepspeed_plugin=self.deepspeed_config if self.use_deepspeed else None
    )
```

**支持的训练模式**：

1. **单卡训练**：直接在单个 GPU 上运行
2. **数据并行**：多卡数据并行（DataParallel）
3. **分布式数据并行**：多节点分布式训练（DDP）
4. **DeepSpeed**：使用 DeepSpeed 进行 ZeRO 优化和流水线并行

### 6.5 量化支持

支持 4-bit 和 8-bit 量化以减少显存占用：

```python
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
```

---

## 7. 在语言模型中的应用

### 7.1 RLHF 中的 TRPO

在基于人类反馈的强化学习（RLHF）中，TRPO 可以用于优化语言模型使其符合人类偏好：

```
┌─────────────────────────────────────────────────────────────┐
│                     RLHF 训练流程                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐               │
│  │  SFT 模型       │ -> │ 奖励模型训练    │               │
│  │  (监督微调)     │    │  (人类偏好数据) │               │
│  └─────────────────┘    └─────────────────┘               │
│                                   │                        │
│                                   v                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              策略优化 (TRPO/PPO)                     │   │
│  │                                                     │   │
│  │  状态 s: 输入提示 (prompt)                          │   │
│  │  动作 a: 生成的响应 (response)                       │   │
│  │  奖励 r: 奖励模型的分数                              │   │
│  │  约束: KL(π_policy || π_reference) ≤ δ              │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                   │                        │
│                                   v                        │
│  ┌─────────────────┐                                     │
│  │  对齐后的模型   │                                     │
│  └─────────────────┘                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 语言模型特有的考虑

#### 7.2.1 状态和动作空间

- **状态**：输入序列（prompt）的 token 序列
- **动作**：输出的 token 序列
- **序列长度**：可达数千个 token

#### 7.2.2 奖励设计

在语言模型 RLHF 中，奖励通常由以下部分组成：

$$R(s, a) = R_{RM}(s, a) - \lambda \cdot KL(\pi_\theta(a|s) || \pi_{ref}(a|s))$$

其中：
- $R_{RM}$：奖励模型的分数
- $\pi_{ref}$：参考策略（通常是 SFT 模型）
- $\lambda$：KL 惩罚系数

#### 7.2.3 挑战

1. **长序列**：需要处理长文本的策略梯度计算
2. **高维动作空间**：词表大小通常是数万
3. **稀疏奖励**：只有生成完整句子后才能获得奖励
4. **计算效率**：大模型的梯度计算开销很大

### 7.3 代码示例：完整的 TRPO 训练

```python
from trainer.trpo_trainer import TRPOTrainer
from config.training_config import TRPOTrainingConfig
from datasets import load_from_disk

# 1. 配置参数
config = TRPOTrainingConfig(
    model_name_or_path="Qwen/Qwen2-7B-Instruct",
    reward_model_path="./reward_model_checkpoints",
    max_seq_length=2048,
    response_length=512,
    max_kl=0.02,
    gamma=0.99,
    gae_lambda=0.95,
    cg_iterations=10,
    cg_damping=0.1,
    learning_rate_actor=1e-5,
    learning_rate_critic=1e-3,
    use_4bit=True,
    lora_r=64,
    lora_alpha=16
)

# 2. 创建训练器
trainer = TRPOTrainer(
    config=config,
    finetuning_type="lora",
    use_deepspeed=False
)

# 3. 加载数据集
train_dataset = load_from_disk("./processed_data/train")
eval_dataset = load_from_disk("./processed_data/eval")

# 4. 加载参考模型和奖励模型
trainer._load_reference_model()
trainer._load_reward_model(config.reward_model_path)

# 5. 开始训练
trainer.train(
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    output_dir="./trpo_checkpoints",
    epochs=3,
    batch_size=4,
    max_kl=0.02,
    gamma=0.99,
    gae_lambda=0.95,
    ppo_epochs=4,
    log_interval=10,
    save_interval=100
)

# 6. 保存模型
trainer.merge_and_unload("./final_model")
```

### 7.4 训练数据格式详解

在基于人类反馈的强化学习（RLHF）和TRPO训练中，数据格式的设计直接影响训练效果。本节将详细介绍奖励模型训练数据和TRPO算法训练数据的主流格式规范。

#### 7.4.1 奖励模型训练数据格式

奖励模型（Reward Model，RM）的训练依赖于高质量的人类偏好数据。这类数据的核心特点是包含对同一提示的不同响应之间的比较信息[10]。主流数据集通常采用偏好对（Preference Pair）的形式，每个样本包含一个提示和多个候选响应，以及人类标注的偏好标签。

**标准JSON格式示例**：

```json
{
  "prompt": "请解释一下什么是人工智能？",
  "chosen": "人工智能（Artificial Intelligence，简称AI）是指计算机系统能够执行通常需要人类智能才能完成的任务。这包括学习、推理、问题解决、知识理解、语言识别等能力。人工智能可以分为弱人工智能和强人工智能两大类。",
  "rejected": "AI就是人工智能，很厉害的东西。",
  "score_chosen": 4.5,
  "score_rejected": 2.0,
  "label": "chosen",
  "metadata": {
    "generation_id": "gen_001",
    "model_source": "gpt-3.5-turbo",
    "annotator_id": "anno_123",
    "timestamp": "2024-01-15T10:30:00Z",
    "category": "helpfulness"
  }
}
```

**批量偏好数据格式（推荐用于大规模训练）**：

```json
{
  "dataset_name": "preference_pairs_train",
  "version": "1.0",
  "total_samples": 250000,
  "splits": {
    "train": 200000,
    "validation": 25000,
    "test": 25000
  },
  "samples": [
    {
      "id": "sample_000001",
      "prompt_id": "prompt_001",
      "prompt": "患者男性，65岁，胸痛2小时，心电图显示ST段抬高，应该如何处理？",
      "responses": [
        {
          "response_id": "resp_A",
          "text": "根据心电图表现，考虑急性心肌梗死。建议立即进行急诊PCI手术，同时嚼服阿司匹林300mg，舌下含服硝酸甘油。建立静脉通路，监测生命体征。",
          "model": "medical-sft-v2",
          "metadata": {
            "response_length": 128,
            "generation_params": {
              "temperature": 0.7,
              "max_tokens": 512
            }
          }
        },
        {
          "response_id": "resp_B",
          "text": "胸痛可能是心脏问题，建议去医院检查一下。",
          "model": "baseline-v1",
          "metadata": {
            "response_length": 32,
            "generation_params": {
              "temperature": 0.7,
              "max_tokens": 512
            }
          }
        }
      ],
      "preference": {
        "winner": "resp_A",
        "loser": "resp_B",
        "confidence": 0.95,
        "reason": "回答A提供了完整的临床处理流程，符合医学指南标准；回答B过于简单，缺乏具体指导。"
      },
      "rubrics": {
        "medical_accuracy": {
          "chosen_score": 5.0,
          "rejected_score": 2.0,
          "weight": 0.4
        },
        "completeness": {
          "chosen_score": 4.5,
          "rejected_score": 1.5,
          "weight": 0.3
        },
        "safety": {
          "chosen_score": 5.0,
          "rejected_score": 3.0,
          "weight": 0.3
        }
      }
    }
  ]
}
```

**HuggingFace标准格式（HuggingFaceH4/stack-exchange-preferences）**[11]：

```json
{
  "question_id": 123456,
  "prompt": "<|prompter|>请告诉我如何预防流感？<|end|>\n<|assistant|>",
  "answer_a": "预防流感的方法包括：接种流感疫苗、保持良好的个人卫生习惯、避免与流感患者密切接触、保持健康的生活方式。",
  "answer_b": "多喝热水，注意保暖。",
  "voted_answer": "answer_a",
  "tstamp": 1640995200,
  "pool": "medicine",
  "segment": "health_prevention"
}
```

**OpenAI Summarize格式（openai/summarize_from_feedback）**[12]：

```json
{
  "info": {
    "task": "summarization",
    "domain": "news_articles",
    "split": "train"
  },
  "prompt": "美国疾病控制与预防中心（CDC）周四报告称，上周美国流感住院率首次超过10%，达到了自2010年以来同期的最高水平。CDC官员表示，今年的流感季节比往年提前到来，且病毒传播速度极快。",
  "summary": "美国流感住院率达到十年来最高水平，CDC报告称上周住院率首次超过10%。",
  "广州日报": "美国流感住院率创新高，CDC数据显示上周已超10%。",
  "user_rating": {
    "summary": 5,
    "广州日报": 3
  },
  "worker_id": "worker_001",
  "work_time_ms": 45000
}
```

**Anthropic HH-RLHF格式**[13]：

```json
{
  "conversation": [
    {
      "role": "human",
      "content": "请帮我写一封辞职信。"
    },
    {
      "role": "assistant",
      "content": "我可以帮您写辞职信。请问您辞职的原因是什么？工作多长时间了？"
    },
    {
      "role": "human",
      "content": "我因为家庭原因需要辞职，已经工作两年了。"
    }
  ],
  "responses": {
    "chosen": {
      "text": "以下是辞职信模板：\n\n尊敬的领导：\n您好！\n首先，感谢公司给予我两年的工作机会。由于家庭原因，我不得不辞去目前的工作。经过慎重考虑，我决定于[日期]正式离职。\n感谢领导和同事们的帮助与支持，我会珍惜这段回忆。\n此致\n敬礼！\n[您的名字]\n[日期]",
      "rating": 5,
      "labels": {
        "helpfulness": 5,
        "harmlessness": 5,
        "honesty": 4
      }
    },
    "rejected": {
      "text": "你就写：领导，我不干了，后天就走。",
      "rating": 1,
      "labels": {
        "helpfulness": 1,
        "harmlessness": 3,
        "honesty": 5
      }
    }
  },
  "preference_type": "pairwise_comparison"
}
```

**数据字段详解**：

| 字段名 | 数据类型 | 必填 | 描述 |
|--------|----------|------|------|
| `prompt` | string | 是 | 输入提示，包含用户问题或任务描述 |
| `chosen` | string | 是 | 人类偏好的响应（正例） |
| `rejected` | string | 是 | 人类不偏好的响应（负例） |
| `score_chosen` | float | 否 | 正例的绝对评分（1-5分制） |
| `score_rejected` | float | 否 | 负例的绝对评分（1-5分制） |
| `label` | string | 否 | 标签标识，如"chosen"、"rejected" |
| `preference_order` | int | 否 | 偏好顺序，数字越小偏好程度越高 |
| `metadata` | object | 否 | 元数据，包含生成信息、标注信息等 |
| `rubrics` | object | 否 | 按评分维度的细粒度分数 |
| `conversation_history` | array | 否 | 多轮对话历史 |

**奖励模型训练的数据处理代码示例**：

```python
from datasets import load_dataset
import json

def create_preference_dataset(
    dataset_name: str = "carperai/openai_summarize_comparisons",
    split: str = "train",
    max_samples: int = None
) -> list:
    """
    从HuggingFace数据集创建偏好数据集
    
    Args:
        dataset_name: 数据集名称
        split: 数据集划分（train/validation/test）
        max_samples: 最大样本数
    
    Returns:
        处理后的偏好数据列表
    """
    dataset = load_dataset(dataset_name, split=split)
    pairs = []
    
    for sample in dataset:
        pair = {
            "prompt": sample["prompt"],
            "chosen": sample["chosen"],
            "rejected": sample["rejected"],
            "metadata": {
                "source": dataset_name,
                "split": split
            }
        }
        pairs.append(pair)
        
        if max_samples and len(pairs) >= max_samples:
            break
    
    return pairs

def convert_to_chat_format(pairs: list, system_prompt: str = None) -> list:
    """
    将偏好数据转换为对话格式
    
    适用于GPT-4、Qwen等支持对话格式的模型
    """
    chat_pairs = []
    for pair in pairs:
        chat_sample = {
            "conversations": [
                {
                    "from": "human",
                    "value": pair["prompt"]
                },
                {
                    "from": "assistant",
                    "value": pair["chosen"],
                    "type": "chosen"
                }
            ]
        }
        
        if system_prompt:
            chat_sample["system"] = system_prompt
            
        chat_pairs.append(chat_sample)
    
    return chat_pairs

# 使用示例
pairs = create_preference_dataset(
    dataset_name="HuggingFaceH4/stack-exchange-preferences",
    split="train",
    max_samples=10000
)
chat_data = convert_to_chat_format(
    pairs,
    system_prompt="你是一个专业、有帮助的医学助手。"
)

# 保存为JSONL格式
with open("preference_data.jsonl", "w", encoding="utf-8") as f:
    for item in chat_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
```

#### 7.4.2 TRPO算法训练数据格式

TRPO算法的训练数据通常以轨迹（Trajectory）的形式组织。每条轨迹记录了智能体与环境交互的完整序列，包括状态、动作、奖励等信息[14]。在语言模型的应用场景中，轨迹数据需要特殊设计以适应文本生成的特性。

**轨迹数据标准格式**：

```json
{
  "trajectory_id": "traj_20240115_001",
  "episode_id": "episode_042",
  "environment": "medical-qa-generation",
  "timestamp": "2024-01-15T14:30:00Z",
  "model_info": {
    "model_name": "Qwen2-7B-Instruct",
    "model_type": "policy_network",
    "checkpoint_path": "./checkpoints/step_1000"
  },
  "trajectory": {
    "states": [
      {
        "id": 0,
        "tokens": [151644, 8948, 123, 234, 456, 789],
        "text": "<|im_start|>user\n请介绍一下糖尿病的并发症。",
        "attention_mask": [1, 1, 1, 1, 1, 1, 1],
        "position_ids": [0, 1, 2, 3, 4, 5, 6],
        "meta": {
          "prompt_length": 28,
          "turn_type": "medical_inquiry"
        }
      }
    ],
    "actions": [
      {
        "token_id": 1024,
        "text": "糖",
        "log_prob": -2.345,
        "entropy": 4.567,
        "meta": {
          "is_new_token": true,
          "generation_step": 1
        }
      },
      {
        "token_id": 2048,
        "text": "尿",
        "log_prob": -1.234,
        "entropy": 3.891,
        "meta": {
          "is_new_token": true,
          "generation_step": 2
        }
      }
    ],
    "rewards": [
      {
        "step": 0,
        "value": 0.0,
        "type": "intermediate",
        "source": "reward_model"
      },
      {
        "step": 100,
        "value": 8.5,
        "type": "final",
        "source": "reward_model",
        "details": {
          "rm_score": 9.0,
          "kl_penalty": -0.5,
          "format_score": 1.0,
          "safety_score": 1.0
        }
      }
    ],
    "values": [
      {
        "step": 0,
        "value_estimate": 7.8,
        "return_estimate": 8.5,
        "advantage": -0.2
      }
    ],
    "log_probs": [
      -2.345,
      -1.234,
      -3.456,
      -2.789
    ]
  },
  "statistics": {
    "total_steps": 256,
    "total_reward": 8.5,
    "mean_reward": 0.0332,
    "reward_variance": 0.125,
    "episode_length": 256,
    "completion_status": "completed",
    "truncated": false,
    "timeout": false
  },
  "reference_info": {
    "reference_model": "Qwen2-7B-Instruct-SFT",
    "reference_log_probs": [-2.401, -1.189, -3.567, -2.801],
    "kl_divergence": 0.015
  }
}
```

**批量轨迹数据格式（用于离线TRPO训练）**：

```json
{
  "batch_info": {
    "batch_id": "batch_20240115_001",
    "num_trajectories": 64,
    "total_steps": 16384,
    "collection_time_ms": 120000,
    "policy_version": "v1.2.0"
  },
  "trajectories": [
    {
      "trajectory_id": "traj_001",
      "prompt": "患者女性，28岁，孕24周，血糖偏高如何处理？",
      "response": "孕24周血糖偏高需要重视。建议进行糖耐量试验明确诊断，必要时进行饮食控制和运动干预，定期监测血糖变化。",
      "response_tokens": [1200, 3400, 5678, 9012, 3456, 7890, 1234],
      "old_log_probs": [-1.2, -2.3, -1.5, -2.8, -1.9, -3.1, -2.4],
      "rewards": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 8.0],
      "advantages": [0.12, 0.11, 0.15, 0.09, 0.13, 0.14, 0.42],
      "value_targets": [7.8, 7.9, 7.7, 7.6, 7.8, 7.9, 8.5],
      "reference_log_probs": [-1.3, -2.4, -1.6, -2.9, -2.0, -3.2, -2.5],
      "KL": 0.018,
      "is_terminal": true
    }
  ],
  "batch_statistics": {
    "mean_reward": 0.85,
    "std_reward": 0.32,
    "mean_advantage": 0.15,
    "std_advantage": 0.08,
    "mean_KL": 0.012,
    "max_KL": 0.025,
    "mean_response_length": 128.5
  }
}
```

**优势估计数据格式（GAE计算结果）**：

```json
{
  "gae_info": {
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "normalize_advantages": true
  },
  "samples": [
    {
      "sample_id": "sample_001",
      "prompt": "高血压患者应该注意什么？",
      "response_tokens": [100, 200, 300, 400, 500, 600, 700],
      "rewards": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 9.5],
      "values": [7.2, 7.4, 7.3, 7.5, 7.6, 7.8, 9.5],
      "td_errors": [0.05, 0.08, -0.02, 0.12, 0.09, 0.15, 1.65],
      "advantages": [0.08, 0.09, 0.07, 0.11, 0.10, 0.13, 0.45],
      "returns": [7.28, 7.49, 7.37, 7.61, 7.70, 7.93, 9.50]
    }
  ]
}
```

**多轮对话轨迹格式**：

```json
{
  "multi_turn_trajectory": {
    "conversation_id": "conv_001",
    "num_turns": 3,
    "turns": [
      {
        "turn_id": 1,
        "role": "user",
        "content": "我有2型糖尿病，日常饮食应该注意什么？",
        "tokens": [151644, 1234, 5678, 9012, 3456, 7890],
        "reward": 0.0
      },
      {
        "turn_id": 2,
        "role": "assistant",
        "content": "2型糖尿病患者的饮食管理非常重要，主要包括以下几点：...",
        "tokens": [151645, 2345, 6789, 1234, 5678, 9012],
        "reward": 0.0
      },
      {
        "turn_id": 3,
        "role": "user",
        "content": "那我可以吃水果吗？比如苹果和香蕉？",
        "tokens": [151644, 3456, 7890, 1234, 5678, 9012],
        "reward": 0.0
      },
      {
        "turn_id": 4,
        "role": "assistant",
        "content": "糖尿病患者可以适量食用水果，但需要注意选择低糖水果和控制摄入量...",
        "tokens": [151645, 4567, 8901, 2345, 6789, 1234],
        "reward": 8.5,
        "reward_breakdown": {
          "medical_accuracy": 4.5,
          "completeness": 4.0,
          "safety": 5.0,
          "format": 5.0,
          "KL_penalty": -1.0
        }
      }
    ],
    "total_reward": 8.5,
    "completion_status": "completed"
  }
}
```

**数据字段详解**：

| 字段名 | 数据类型 | 必填 | 描述 |
|--------|----------|------|------|
| `trajectory_id` | string | 是 | 轨迹唯一标识符 |
| `episode_id` | string | 否 | 回合标识符 |
| `states` | array | 是 | 状态序列，包含token序列、文本、注意力掩码等 |
| `actions` | array | 是 | 动作序列，包含token ID、对数概率、熵等 |
| `rewards` | array | 是 | 奖励序列，包含每步奖励值、奖励类型等 |
| `values` | array | 否 | 价值估计序列，包含状态价值、优势值等 |
| `log_probs` | array | 是 | 旧策略的对数概率序列 |
| `old_log_probs` | array | 是 | 采样时的对数概率，用于重要性采样 |
| `advantages` | array | 否 | 优势值序列，用于策略梯度计算 |
| `returns` | array | 否 | 回报序列，用于价值函数训练 |
| `reference_log_probs` | array | 否 | 参考模型的对数概率，用于KL散度计算 |
| `KL` | float | 否 | KL散度值 |
| `is_terminal` | boolean | 否 | 是否为终止状态 |

**TRPO训练数据的预处理代码示例**：

```python
import torch
import json
from typing import List, Dict, Tuple
from datasets import Dataset

class TRPODataProcessor:
    """
    TRPO训练数据处理器
    
    功能：
    1. 从原始轨迹数据中提取训练样本
    2. 计算GAE优势估计
    3. 标准化优势值
    4. 转换为模型输入格式
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantages: bool = True
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool]
    ) -> Tuple[List[float], List[float]]:
        """
        计算广义优势估计（GAE）
        
        Args:
            rewards: 奖励序列
            values: 价值估计序列
            dones: 终止标志序列
        
        Returns:
            advantages: 优势值序列
            returns: 回报序列
        """
        advantages = []
        returns = []
        
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                gae = 0
            
            delta = rewards[t] + self.gamma * next_value * (1 - int(dones[t])) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - int(dones[t])) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            next_value = values[t]
        
        return advantages, returns
    
    def normalize(
        self,
        advantages: torch.Tensor
    ) -> torch.Tensor:
        """
        标准化优势值
        """
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages
    
    def process_trajectory(
        self,
        trajectory: Dict
    ) -> Dict:
        """
        处理单条轨迹数据
        
        提取训练所需的所有信息
        """
        rewards = trajectory["rewards"]
        values = trajectory.get("values", [0] * len(rewards))
        dones = [True] * (len(rewards) - 1) + [trajectory.get("is_terminal", True)]
        
        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = self.normalize(torch.tensor(advantages))
        
        return {
            "input_ids": trajectory["input_ids"],
            "attention_mask": trajectory["attention_mask"],
            "response_ids": trajectory["response_ids"],
            "old_log_probs": trajectory["old_log_probs"],
            "advantages": advantages.tolist(),
            "returns": returns,
            "reference_log_probs": trajectory.get("reference_log_probs", []),
            "KL": trajectory.get("KL", 0.0)
        }
    
    def create_training_dataset(
        self,
        trajectories: List[Dict],
        max_samples: int = None
    ) -> Dataset:
        """
        从轨迹列表创建训练数据集
        """
        processed_data = []
        
        for traj in trajectories:
            if max_samples and len(processed_data) >= max_samples:
                break
            
            processed = self.process_trajectory(traj)
            processed_data.append(processed)
        
        return Dataset.from_list(processed_data)
    
    def save_to_jsonl(
        self,
        trajectories: List[Dict],
        output_path: str
    ):
        """
        将轨迹数据保存为JSONL格式
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for traj in trajectories:
                processed = self.process_trajectory(traj)
                f.write(json.dumps(processed, ensure_ascii=False) + "\n")
    
    def load_from_jsonl(
        self,
        input_path: str,
        max_samples: int = None
    ) -> List[Dict]:
        """
        从JSONL文件加载轨迹数据
        """
        trajectories = []
        
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if max_samples and len(trajectories) >= max_samples:
                    break
                
                traj = json.loads(line.strip())
                trajectories.append(traj)
        
        return trajectories
```

#### 7.4.3 常用公开数据集

**偏好数据集资源**：

| 数据集名称 | 规模 | 特点 | 适用场景 |
|------------|------|------|----------|
| HuggingFaceH4/stack-exchange-preferences | 100K+ | StackExchange问答对 | 通用对话、指令微调 |
| carperai/openai_summarize_comparisons | 250K+ | 文本摘要偏好对 | 摘要任务对齐 |
| openai/summarize_from_feedback | 100K+ | 多维度评分 | 文本质量评估 |
| Anthropic/HH-RLHF | 170K+ | 有用性和无害性标注 | 对话安全对齐 |
| StanfordNLP/SHP | 100K+ | Reddit问答偏好 | 社区问答微调 |
| Intel/orca_dpo_pairs | 100K+ | GPT-4生成偏好对 | 知识密集型任务 |

**数据集下载和使用示例**：

```python
from datasets import load_dataset, concatenate_datasets

def load_preference_datasets():
    """
    加载并合并多个偏好数据集
    """
    datasets = []
    
    stack_exchange = load_dataset(
        "HuggingFaceH4/stack-exchange-preferences",
        split="train[:10000]"
    )
    stack_exchange = stack_exchange.rename_column("prompt", "input")
    stack_exchange = stack_exchange.rename_column("chosen", "positive")
    stack_exchange = stack_exchange.rename_column("rejected", "negative")
    datasets.append(stack_exchange)
    
    summarize = load_dataset(
        "carperai/openai_summarize_comparisons",
        split="train[:5000]"
    )
    summarize = summarize.rename_column("prompt", "input")
    summarize = summarize.rename_column("chosen", "positive")
    summarize = summarize.rename_column("rejected", "negative")
    datasets.append(summarize)
    
    combined = concatenate_datasets(datasets)
    return combined

def filter_by_quality(dataset, min_score: float = 3.0):
    """
    根据评分过滤低质量样本
    """
    def filter_fn(example):
        score = example.get("score", 0)
        return score >= min_score
    
    return dataset.filter(filter_fn)
```

#### 7.4.4 数据质量要求

**奖励模型数据质量标准**[15]：

| 维度 | 要求 | 说明 |
|------|------|------|
| 偏好一致性 | >95% | 同一pair不同标注者应有一致偏好 |
| 响应多样性 | >0.5 | 生成模型温度建议0.7-1.0 |
| 标注者可靠性 | >0.8 | 标注者内部一致性检验 |
| 覆盖度 | 多领域 | 应包含目标应用领域的样本 |
| 平衡性 | ≈1:1 | 正负例比例应大致平衡 |

**TRPO训练数据质量标准**：

| 维度 | 要求 | 说明 |
|------|------|------|
| 轨迹完整性 | 100% | 完整的状态-动作-奖励序列 |
| 奖励信噪比 | >0.5 | 奖励变化应显著 |
| 响应质量 | >3.5 | 生成响应应基本可用 |
| 长度分布 | 合理 | 响应长度应符合目标场景 |
| 采样覆盖度 | >0.8 | 避免过度集中于某些提示 |

---

## 8. 参考文献

### 8.1 原始论文

[1] Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). Trust Region Policy Optimization. *International Conference on Machine Learning (ICML)*.

[2] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv preprint arXiv:1707.06347*.

[3] Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. *International Conference on Learning Representations (ICLR)*.

### 8.2 实现参考

[4] OpenAI Spinning Up. "Trust Region Policy Optimization". https://spinningup.openai.com/

[5] Patrick Coady. "TRPO Implementation in TensorFlow". https://github.com/patrick-coady/trpo

[6] "从代码学习深度强化学习 - TRPO PyTorch版". CSDN. https://blog.csdn.net/weixin_43887510/article/details/148961558

### 8.3 延伸阅读

[7] Kakade, S., & Langford, J. (2002). Approximately Optimal Approximate Reinforcement Learning. *International Conference on Machine Learning (ICML)*.

[8] Williams, R. J. (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. *Machine Learning*.

[9] Mnih, V., et al. (2016). Asynchronous Methods for Deep Reinforcement Learning. *International Conference on Machine Learning (ICML)*.

### 8.4 资源链接

| 资源 | 链接 |
|------|------|
| TRPO 原始论文 | https://proceedings.mlr.press/v37/schulman15.html |
| PPO 论文 | https://arxiv.org/abs/1707.06347 |
| GAE 论文 | https://arxiv.org/abs/1506.02438 |
| OpenAI Spinning Up | https://spinningup.openai.com/ |
| Hugging Face Accelerate | https://huggingface.co/docs/accelerate |
| PEFT 文档 | https://huggingface.co/docs/peft |

---

## 附录

### A. 常用超参数设置

| 参数 | 推荐范围 | 说明 |
|------|----------|------|
| `max_kl` | 0.001 ~ 0.03 | KL 散度上限，越小更新越保守 |
| `gamma` | 0.99 ~ 0.9995 | 折扣因子，越大越关注长期回报 |
| `gae_lambda` | 0.9 ~ 0.99 | GAE 参数，越大方差越小 |
| `cg_iterations` | 5 ~ 20 | 共轭梯度迭代次数 |
| `cg_damping` | 0.01 ~ 0.5 | Fisher 矩阵阻尼，越大越稳定 |
| `learning_rate_actor` | 1e-6 ~ 1e-4 | 策略学习率 |
| `learning_rate_critic` | 1e-4 ~ 1e-2 | 价值学习率 |

### B. 故障排除

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| KL 散度不收敛 | `max_kl` 设置过大 | 减小 `max_kl` |
| 训练不稳定 | 学习率过高 | 降低学习率 |
| 共轭梯度不收敛 | 阻尼系数太小 | 增加 `cg_damping` |
| 显存不足 | 序列长度或批次太大 | 减小 `max_seq_length` 或 `batch_size` |
| 线搜索失败 | 更新方向错误 | 检查 Fisher 向量积计算 |

### C. 数学符号表

| 符号 | 含义 |
|------|------|
| $\theta$ | 策略参数 |
| $\pi_\theta$ | 参数化策略 |
| $J(\theta)$ | 期望回报目标函数 |
| $A(s, a)$ | 优势函数 |
| $V(s)$ | 状态价值函数 |
| $\gamma$ | 折扣因子 |
| $\lambda$ | GAE 参数 |
| $F$ | Fisher 信息矩阵 |
| $KL(p || q)$ | KL 散度 |
| $\delta$ | TD 误差 |

---

## 9. 训练数据格式规范

### 9.1 概述

在基于人类反馈的强化学习（RLHF）流程中，训练数据通常分为两个阶段：

1. **奖励模型训练阶段**：使用偏好对数据训练奖励模型，学习人类对回答质量的判断标准
2. **策略优化阶段**：使用轨迹数据训练策略模型（如 TRPO），优化生成策略以获得更高奖励

这两个阶段使用不同格式的数据文件，本节将详细说明其规范。

---

### 9.2 奖励模型训练数据（reward_train.json）

#### 9.2.1 数据格式

奖励模型训练数据采用标准的**偏好对（Preference Pair）**格式，每条样本包含一个问题和两个不同质量的回答：

```json
{
  "prompt": "问题文本",
  "chosen": "优质回答（人类偏好）",
  "rejected": "劣质回答（人类不偏好）"
}
```

#### 9.2.2 字段说明

| 字段名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `prompt` | string | 是 | 用户输入的问题或指令 |
| `chosen` | string | 是 | 人类标注者认为更好的回答 |
| `rejected` | string | 是 | 人类标注者认为较差的回答 |

#### 9.2.3 示例数据

```json
[
  {
    "prompt": "请计算：请问 15 + 27 等于多少？请写出计算过程。",
    "chosen": "15 + 27 的计算过程如下：\n\n**个位相加**：5 + 7 = 12，写 2，进位 1\n\n**十位相加**：1 + 2 = 3，再加上进位 1，等于 4\n\n**最终结果**：42\n\n因此，15 + 27 = **42**",
    "rejected": "15 + 27 = 42，直接心算就行，不用写过程。"
  },
  {
    "prompt": "请计算：请问 3² + 4² 等于多少？请说明计算步骤。",
    "chosen": "3² + 4² = 9 + 16 = 25\n\n**计算步骤**：\n\n**第一步：计算平方**\n- 3² = 3 × 3 = 9\n- 4² = 4 × 4 = 16\n\n**第二步：相加**\n- 9 + 16 = 25\n\n**验证**：\n这是一个经典的勾股数组合，3、4、5 构成直角三角形，满足 3² + 4² = 5² = 25\n\n因此，3² + 4² = **25**",
    "rejected": "3² + 4² = 9 + 16 = 25，直接算平方然后加起来就行。"
  }
]
```

#### 9.2.4 数据质量要求

- **偏好一致性**：chosen 回答应明显优于 rejected 回答
- **内容完整性**：回答应完整解答问题，避免中途截断
- **格式规范性**：JSON 格式正确，无转义错误
- **多样性**：问题类型和回答风格应具有多样性

---

### 9.3 TRPO 训练数据（trpo_train.json）

#### 9.3.1 数据格式

TRPO 训练数据采用**轨迹（Trajectory）**格式，每条样本包含完整的生成轨迹信息：

```json
{
  "prompt": "问题文本",
  "trajectory": {
    "tokens": [token_id_1, token_id_2, ...],
    "token_texts": ["token_str_1", "token_str_2", ...],
    "log_probs": [log_prob_1, log_prob_2, ...],
    "rewards": [reward_1, reward_2, ...],
    "values": [value_1, value_2, ...],
    "attention_mask": [mask_1, mask_2, ...]
  },
  "reference_score": 8.5,
  "metadata": {
    "model_source": "模型名称",
    "generation_config": {...}
  }
}
```

#### 9.3.2 轨迹字段说明

| 字段名 | 类型 | 必填 | 说明 | TRPO 作用 |
|--------|------|------|------|-----------|
| `tokens` | list[int] | 是 | 生成的所有 token ID 序列 | 计算 importance sampling ratio |
| `token_texts` | list[str] | 否 | token 对应的文本（调试用） | 日志和可视化 |
| `log_probs` | list[float] | 是 | 每个 token 的 log 概率 | 计算 KL 散度、策略更新 |
| `rewards` | list[float] | 是 | 每个时间步的即时奖励 | 计算 return 和 advantage |
| `values` | list[float] | 是 | critic 估计的状态价值 | 计算 GAE 优势估计 |
| `attention_mask` | list[int] | 是 | 注意力掩码 | 批量计算时忽略 padding |

#### 9.3.3 其他字段说明

| 字段名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `reference_score` | float | 否 | 奖励模型给出的整体评分 |
| `metadata` | dict | 否 | 元数据信息 |
| `metadata.model_source` | string | 否 | 生成轨迹的模型名称 |
| `metadata.generation_config` | dict | 否 | 生成参数配置 |

#### 9.3.4 示例数据

```json
[
  {
    "prompt": "请计算：请问 15 + 27 等于多少？请写出计算过程。",
    "trajectory": {
      "tokens": [13, 287, 158, 196, 338, 87, 25, 312, 87, 102, 87, 43, 198, 287, 76, 12],
      "token_texts": ["15", " ", "+", " ", "27", " ", "的", "计", "算", "过", "程", "如", "下", "：", "\n", "\n"],
      "log_probs": [-0.023, -2.891, -0.234, -3.456, -0.089, -1.234, -0.456, -0.789, -0.234, -1.567, -0.345, -0.678, -0.123, -0.456, -2.134, -1.892],
      "rewards": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0],
      "values": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.3, 1.3],
      "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    },
    "reference_score": 8.5,
    "metadata": {
      "model_source": "Qwen2-7B-Instruct-sft",
      "generation_config": {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "do_sample": true
      }
    }
  }
]
```

#### 9.3.5 奖励信号设计

在语言模型场景中，奖励信号通常来自以下来源：

| 奖励来源 | 说明 | 典型权重 |
|----------|------|----------|
| 奖励模型评分 | 训练好的奖励模型给出的质量分数 | 1.0 |
| 格式奖励 | 回答格式规范的奖励（如包含步骤说明） | 0.1-0.3 |
| KL 惩罚 | 防止策略偏离参考模型太远 | 0.01-0.1 |
| 长度奖励 | 控制回答长度在合理范围内 | 0.01-0.05 |
| 安全奖励 | 避免生成有害或不当内容 | -1.0 |

---

### 9.4 数据格式对比

| 特性 | reward_train.json | trpo_train.json |
|------|-------------------|-----------------|
| 格式类型 | 偏好对格式 | 轨迹格式 |
| 训练目标 | 奖励模型 | 策略模型（TRPO） |
| 主要字段 | prompt, chosen, rejected | prompt, trajectory, rewards |
| 数据来源 | 人类标注 | 模型生成 |
| 更新频率 | 静态数据 | 可动态生成 |

---

### 9.5 数据流程关系

```
┌─────────────────────────────┐      ┌─────────────────────────────┐
│     reward_train.json       │ ───▶ │      Reward Model           │
│     （偏好对格式）           │      │     （奖励模型训练）         │
│     10条样本                │      │     预测偏好概率            │
└─────────────────────────────┘      └─────────────────────────────┘
                                            │
                                            ▼
        ┌─────────────────────────────┐      ┌─────────────────────────────┐
        │     trpo_train.json         │ ───▶ │      TRPO Training          │
        │     （轨迹格式）             │      │     （策略优化训练）         │
        │     10条样本                │      │     优化策略参数            │
        └─────────────────────────────┘      └─────────────────────────────┘
```

**关键说明**：

1. **reward_train.json** 是奖励模型的训练数据，用于学习人类偏好
2. **trpo_train.json** 是 TRPO 的训练数据，包含完整的生成轨迹
3. 在实际训练中，trpo_train.json 的 trajectory 数据通常由策略模型实时生成
4. reward_model 的输出作为 TRPO 的奖励信号来源

---

### 9.6 注意事项

1. **数据一致性**：两个文件的 prompt 应保持一致，确保奖励模型和策略优化使用相同的评估标准

2. **数据规模**：
   - reward_train.json：通常需要数千到数万条偏好对
   - trpo_train.json：实际训练时可动态生成，每次迭代生成新轨迹

3. **数据质量**：
   - 偏好对应有明确的优劣区分
   - 轨迹数据应包含合理的奖励分布

4. **格式验证**：建议使用 JSON Schema 进行格式验证

---

*本文档最后更新于 2024 年 12 月*
