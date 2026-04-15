# Agent体系融合迭代方案

## 背景

当前项目存在两套并行的Agent系统：
- **MedicalAgent + BaseAgent**：单Agent架构，依赖`AgentManager`、`AgentFactory`，通过`rag_pipeline`获取上下文
- **MDAgents**：多Agent协作系统，包含复杂度分析、专家招募、共识机制等完整会诊流程

两套系统完全独立，API入口不统一，MDAgents缺少RAG能力。

## 目标

1. 将MedicalAgent融入MDAgents体系，**完全保留会诊机制**
2. 使用**AgentScope框架**替代现有Agent架构
3. 使用**REME记忆系统**管理对话上下文
4. **每个specialist回答前**调用`rag_pipeline`获取上下文
5. 对外仅暴露`api/services/mdagents_service.py`统一入口
6. 删除`AgentManager`、`AgentFactory`、`api/routers/agent.py`

---

## 技术选型

### 1. AgentScope框架

- **GitHub**: https://github.com/agentscope-ai/agentscope
- **文档**: https://doc.agentscope.io/
- **定位**：生产级、易用的Agent开发框架
- **要求**：Python 3.10+

**核心特性**：
- 内置`ReActAgent`、`DeepResearchAgent`等多种Agent类型
- 内置`MsgHub`多Agent编排
- 支持MCP/A2A协议
- 内置RAG、Pipeline、Planning等
- 支持分布式部署（K8s、Serverless）

**安装**：
```bash
# 锁定版本避免 2.0 破坏性变更
pip install agentscope==1.2.8

# 或从源码（锁定1.x分支）
git clone -b main https://github.com/agentscope-ai/agentscope.git
cd agentscope && git checkout tags/v1.2.8 -b stable-v1
pip install -e .
```

**⚠️ AgentScope 2.0 兼容性说明**：
- AgentScope 2.0 预计 2026-04 发布（[Roadmap](https://github.com/agentscope-ai/agentscope/discussions)）
- 2.0 可能存在 API 破坏性变更
- **本方案锁定版本为 1.2.8**，待 2.0 稳定后再考虑升级
- 升级前需完成完整的回归测试

**核心组件**：

| 组件 | 说明 |
|------|------|
| `ReActAgent` | 内置ReAct推理Agent，工具调用+推理循环 |
| `MsgHub` | 多Agent编排中枢，支持广播、轮询等模式 |
| `Toolkit` | 工具注册管理，`register_tool_function()`注册工具 |
| `InMemoryMemory` | 简单内存存储，短期会话记忆 |
| `ReMeInMemoryMemory` | REME压缩记忆，会话内存，自动管理长对话 |
| `LongTermMemory` | 长期记忆基类，可继承实现自定义 |

**基础用法**：
```python
from agentscope.agent import ReActAgent
from agentscope.model import DashScopeChatModel
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit, execute_python_code
import os, asyncio

async def main():
    toolkit = Toolkit()
    toolkit.register_tool_function(execute_python_code)

    agent = ReActAgent(
        name="assistant",
        sys_prompt="你是一个有帮助的助手",
        model=DashScopeChatModel(
            model_name="qwen-max",
            api_key=os.environ["DASHSCOPE_API_KEY"],
        ),
        memory=InMemoryMemory(),
        toolkit=toolkit,
    )

    msg = Msg(role="user", content="你好")
    response = await agent(msg)
```

---

### 2. REME记忆系统

- **GitHub**: https://github.com/agentscope-ai/ReMe
- **中文README**: https://github.com/agentscope-ai/ReMe/blob/main/README_ZH.md
- **定位**：专为AI Agent打造的记忆管理框架，解决context window有限和会话无状态两大核心问题

**核心能力**：
- **上下文压缩**：长对话自动浓缩成结构化摘要
- **长期记忆**：重要信息持久化到Markdown文件
- **工具输出压缩**：大工具输出外部存储+引用
- **自动触发**：监控token使用，主动压缩

**安装**：
```bash
# 锁定版本
pip install reme-ai==0.3.2 python-dotenv

# 基于文件的记忆系统（ReMeLight）
git clone https://github.com/agentscope-ai/ReMe.git
cd ReMe && git checkout tags/v0.3.2 -b stable
pip install -e ".[light]"
```

**环境变量**：

| 变量 | 说明 | 示例 |
|------|------|------|
| `LLM_API_KEY` | LLM API Key | `sk-xxx` |
| `LLM_BASE_URL` | LLM Base URL | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `EMBEDDING_API_KEY` | Embedding API Key（可选） | `sk-xxx` |
| `EMBEDDING_BASE_URL` | Embedding Base URL（可选） | `https://dashscope.aliyuncs.com/compatible-mode/v1` |

---

## REME两大记忆系统详解

### 系统一：ReMeLight（基于文件的记忆系统）

**核心类**：`ReMeLight`，提供完整的记忆管理能力

**文件结构**：
```
.reme/
├── MEMORY.md                    # 长期记忆，Agent Profile
└── memory/
    ├── 2025-02-12.md           # 每日日志
    └── ...
```

**核心方法**：

| 方法 | 功能 | 说明 |
|------|------|------|
| `check_context` | 上下文检查 | Token计数，判断是否需要压缩 |
| `compact_memory` | 对话压缩 | 用ReActAgent将历史压缩为结构化摘要 |
| `compact_tool_result` | 工具输出压缩 | 超长工具输出截断存文件，消息中保留引用 |
| `pre_reasoning_hook` | 推理前处理钩子 | 自动执行上述所有压缩步骤 |
| `summary_memory` | 长期记忆持久化 | 异步写入memory/*.md |
| `memory_search` | 记忆搜索 | 向量+BM25混合检索 |
| `get_in_memory_memory` | 会话内存实例 | 返回`ReMeInMemoryMemory` |
| `mark_messages_compressed` | 标记消息压缩 | 持久化到dialog/*.jsonl |

**压缩摘要结构**：

| 字段 | 说明 |
|------|------|
| `## Goal` | 用户目标 |
| `## Constraints` | 约束和偏好 |
| `## Progress` | 任务进度 |
| `## Key Decisions` | 关键决策 |
| `## Next Steps` | 下一步计划 |
| `## Critical Context` | 关键上下文（文件路径、函数名等） |

**ReMeInMemoryMemory**（会话内存）：
- 扩展AgentScope的`InMemoryMemory`
- Token感知内存管理
- 对话持久化到`dialog/YYYY-MM-DD.jsonl`

**集成到Agent推理流程**：
```
Agent → pre_reasoning_hook → compact_tool_result → check_context → compact_memory → memory_search
                                                     ↓ (超限)
                                            summary_memory → memory/*.md
```

---

### 系统二：ReMe（基于向量库的记忆系统）

**核心类**：`ReMe`，支持三种记忆类型统一管理

**记忆类型**：

| 类型 | 用途 |
|------|------|
| **个人记忆** | 用户偏好、习惯 |
| **任务/程序性记忆** | 任务执行经验、成功/失败模式 |
| **工具记忆** | 工具使用经验、参数优化 |

**核心方法**：

| 方法 | 功能 |
|------|------|
| `summarize_memory` | 从对话中提取并存储记忆 |
| `retrieve_memory` | 根据查询检索相关记忆 |
| `add_memory` | 手动添加记忆 |
| `get_memory` | 通过ID获取单条记忆 |
| `update_memory` | 更新记忆 |
| `delete_memory` | 删除记忆 |
| `list_memory` | 列出某类记忆 |

**基础用法**：
```python
import asyncio
from reme import ReMe

async def main():
    reme = ReMe(
        working_dir=".reme",
        default_llm_config={"backend": "openai", "model_name": "qwen3.5-plus"},
        default_embedding_model_config={"backend": "openai", "model_name": "text-embedding-v4", "dimensions": 1024},
        default_vector_store_config={"backend": "local"},
    )
    await reme.start()

    # 总结记忆
    result = await reme.summarize_memory(messages=messages, user_name="alice")

    # 检索记忆
    memories = await reme.retrieve_memory(query="Python编程", user_name="alice")

    await reme.close()
```

---

### ReMeInMemoryMemory（AgentScope集成用）

**用途**：自动working memory管理，解决"context rot"问题

**解决问题**：
- Context爆炸：50+工具调用导致context溢出
- 重复响应：模型在重复内容上"徘徊"
- 推理变慢：注意力分散在大量历史
- 焦点丢失：无法聚焦核心问题

**解决方案**：

| 策略 | 说明 |
|------|------|
| **Context Offloading** | 大工具输出外部存储+引用 |
| **Context Reduction** | LLM压缩工具结果 |
| **Smart Retention** | 保留近期消息保证连续性 |
| **Automatic Triggering** | 监控token使用主动触发 |

**配置参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | LLM | 必需 | 用于压缩的LLM |
| `working_summary_mode` | str | `"auto"` | `"compact"`, `"compress"`, `"auto"` |
| `max_total_tokens` | int | 20000 | 触发压缩的token阈值 |
| `max_tool_message_tokens` | int | 2000 | 单个工具消息最大token |
| `keep_recent_count` | int | 10 | 保留的近期消息数 |
| `compact_ratio_threshold` | float | 0.75 | 触发压缩的阈值比例 |
| `store_dir` | str | `"inmemory"` | 外部存储目录 |

**AgentScope集成用法**：
```python
import os
from dotenv import load_dotenv
load_dotenv()

from agentscope.agent import ReActAgent
from agentscope.model import DashScopeChatModel
from agentscope.memory import ReMeInMemoryMemory
from agentscope.tool import Toolkit

# 创建LLM
llm = DashScopeChatModel(
    model_name="qwen3-coder-30b-a3b-instruct",
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
)

# 创建REME会话内存（通过ReMeLight.get_in_memory_memory获取）
reme_light = ReMeLight(working_dir=".reme", model=llm)
short_term_memory = reme_light.get_in_memory_memory(
    working_summary_mode="auto",
    compact_ratio_threshold=0.75,
    max_total_tokens=20000,
    max_tool_message_tokens=2000,
    keep_recent_count=1,
)

# 使用async上下文管理器
async with short_term_memory:
    # 添加消息
    await short_term_memory.add(messages, allow_duplicates=True)

    # 创建带记忆的ReActAgent
    agent = ReActAgent(
        name="medical_agent",
        sys_prompt="你是一个医疗助手",
        model=llm,
        toolkit=toolkit,
        memory=short_term_memory,
        max_iters=20,
    )

    # 与Agent交互
    msg = Msg(role="user", content="医生，我头疼")
    response = await agent(msg)
```

---

## 迭代方案

### 第一阶段：基础设施重构

**目标**：建立基于AgentScope的新架构，保留MDAgents核心逻辑

**任务**：
1. [ ] 安装AgentScope和REME依赖（锁定版本：agentscope==1.2.8, reme-ai==0.3.2）
2. [ ] 创建新的`AgentScopeAgent`基类，封装AgentScope的`ReActAgent`
3. [ ] 将MDAgents中的6种Agent(pcc, specialist, moderator, recruiter, reviewer, integrator)迁移为AgentScope Agent
4. [ ] 设计并实现`MainController → AgentScope MsgHub`映射机制（详见下方设计）
5. [ ] 保留`MainController`、`ComplexityAnalyzer`、`CollaborationAllocator`、`ConsensusMechanism`核心逻辑
6. [ ] 将`mdagents_service.py`改写为使用AgentScope Agent
7. [ ] **废弃`api/routers/agent.py`路由**，从FastAPI注册中移除

**MainController → MsgHub 映射设计**：

```
┌─────────────────────────────────────────────────────────────────┐
│                        MainController                            │
│  (MDAgents核心协调器，主持会诊流程状态机)                          │
│                                                                   │
│  状态机: IDLE → ANALYZING → ALLOCATING → CONSULTING →           │
│         CONSENSUS → REPORTING → COMPLETED                        │
└─────────────────────────────────────────────────────────────────┘
              │
              │ 协调指令 (dispatch)
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        MsgHub                                    │
│  (AgentScope多Agent编排中枢)                                     │
│                                                                   │
│  角色映射:                                                       │
│  ├── MainController = MsgHub主持人 (host)                        │
│  ├── pcc          = 广播者 (broadcaster) - 发布问题给所有专家    │
│  ├── specialist   = 订阅者 (subscriber) - 接收任务并回复         │
│  ├── moderator    = 主持人 (moderator) - 引导讨论流程            │
│  ├── recruiter    = 观察者 (observer) - 监控并补充专家           │
│  ├── reviewer     = 评审者 (reviewer) - 评估回答质量            │
│  └── integrator   = 整合者 (integrator) - 汇总最终报告           │
│                                                                   │
│  消息模式:                                                       │
│  ├── 广播模式: pcc发布问题 → 所有specialist收到                   │
│  ├── 轮询模式: moderator依次询问专家意见                          │
│  └── 定向模式: recruiter单独联系特定专家                         │
└─────────────────────────────────────────────────────────────────┘
```

**关键映射规则**：

| MainController方法 | MsgHub操作 | 说明 |
|------------------|-----------|------|
| `analyze_complexity()` | 无（纯计算） | 保持不变 |
| `allocate_collaboration()` | `msg_hub.broadcast(role="specialist", msg)` | 分配任务广播 |
| `recruit_expert()` | `msg_hub.direct(agent_id, msg)` | 定向招募专家 |
| `conduct_consultation()` | `msg_hub.broadcast() → collect_response()` | 发起会诊 |
| `reach_consensus()` | `moderator.review(responses)` | 共识评审 |
| `generate_report()` | `integrator.synthesize()` | 生成报告 |

**验收标准**：
- [ ] `POST /api/mdagents/consult` 端点正常响应
- [ ] `GET /api/mdagents/health` 端点正常响应
- [ ] 6种Agent角色均能正常创建和通信
- [ ] 会诊流程状态机按 IDLE → ANALYZING → ALLOCATING → CONSULTING → CONSENSUS → REPORTING → COMPLETED 顺序执行

---

### 第二阶段：RAG集成与记忆系统

**目标**：在AgentScope Agent中集成RAG和REME记忆，RAG在每个specialist回答前触发

**任务**：
1. [ ] 为每个Agent添加`rag_pipeline`调用能力
2. [ ] 使用`ReMeInMemoryMemory`替换原有简单memory机制
3. [ ] **在每个specialist回答前自动调用`rag_pipeline.query()`获取上下文**
4. [ ] 配置`max_total_tokens`、`max_tool_message_tokens`等压缩参数
5. [ ] 保留对话历史的持久化能力

**RAG集成时机设计**：

```
会诊流程中的RAG触发点：

  ┌─────────────┐    broadcast question
  │     pcc     │──────────────────────────▶ specialist_1
  └─────────────┘                              │ RAG query
                                               ▼
                                         ┌───────────┐
                                         │  RAG检索  │──▶ 相关医学知识上下文
                                         └───────────┘
                                               │
  ┌─────────────┐    recruit expert              ▼
  │  recruiter  │────────────────────────▶ specialist_2
  └─────────────┘                              │ RAG query
                                               ▼
                                         ┌───────────┐
                                         │  RAG检索  │──▶ 相关医学知识上下文
                                         └───────────┘
                                               │
                                               ▼
                                    (specialist回答时携带RAG上下文)
```

**关键代码模式**：
```python
class MedicalSpecialist(ReActAgent):
    """继承AgentScope ReActAgent，每个回答前自动RAG"""

    def __init__(self, name: str, rag_pipeline: RAGPipeline, **kwargs):
        self.rag_pipeline = rag_pipeline
        super().__init__(name=name, **kwargs)

    async def respond(self, msg: Msg) -> Msg:
        # Step 1: RAG检索获取上下文（每个specialist回答前必触发）
        query = msg.content
        rag_context = self.rag_pipeline.build_context(query)

        # Step 2: 将RAG上下文注入消息
        augmented_msg = Msg(
            role="user",
            content=f"【相关医学知识】\n{rag_context}\n\n【用户问题】\n{query}"
        )

        # Step 3: 通过REME记忆管理层（自动压缩/持久化）
        await self.memory.add(augmented_msg)

        # Step 4: 调用pre_reasoning_hook自动管理记忆
        if hasattr(self.memory, 'pre_reasoning_hook'):
            await self.memory.pre_reasoning_hook()

        # Step 5: 执行Agent推理生成回答
        response = await super().respond(augmented_msg)

        # Step 6: 回答存入记忆
        await self.memory.add(response)

        return response

# 使用REME管理记忆（会话内存实例）
from reme.light import ReMeLight

reme_light = ReMeLight(
    working_dir=".reme",
    model=llm,
)
in_memory = reme_light.get_in_memory_memory(
    max_total_tokens=20000,
    max_tool_message_tokens=2000,
    keep_recent_count=10,
)
```

**验收标准**：
- [ ] 每个specialist在生成回答前，RAG上下文被正确注入
- [ ] 使用日志验证：`rag_pipeline.query()`在specialist.respond()时被调用
- [ ] 记忆压缩正常工作：连续10轮对话后，上下文被正确压缩
- [ ] 历史会话可被追溯：`.reme/dialog/`目录下有JSONL会话记录

---

### 第三阶段：接口统一与清理

**目标**：统一API入口，删除冗余代码

**任务**：
1. [ ] 删除`src/agent/agent_manager.py`
2. [ ] 删除`src/agent/agent_factory.py`
3. [ ] 删除`src/agent/base_agent.py`（被AgentScope替代）
4. [ ] 删除`src/agent/medical_agent.py`（功能已融合）
5. [ ] 删除`src/agent/multi_agent_system.py`（被MDAgents替代）
6. [ ] **删除`src/agent/tools/tool_manager.py`**（被AgentScope Toolkit替代）
7. [ ] **删除`src/agent/tools/`目录**（所有工具迁移到AgentScope Toolkit）
8. [ ] **删除`api/routers/agent.py`**（已废弃）
9. [ ] 从FastAPI主应用(`api/main.py`或类似)中移除`agent_router`的注册
10. [ ] 所有工具通过AgentScope `Toolkit.register_tool_function()`重新注册

**删除的模块清单**：

| 文件/目录 | 原因 |
|----------|------|
| `src/agent/agent_manager.py` | AgentScope自动管理Agent生命周期 |
| `src/agent/agent_factory.py` | 不再需要工厂模式 |
| `src/agent/base_agent.py` | 被AgentScope ReActAgent替代 |
| `src/agent/medical_agent.py` | 功能已融入MDAgents |
| `src/agent/multi_agent_system.py` | 被MDAgents替代 |
| `src/agent/tools/tool_manager.py` | 被AgentScope Toolkit替代 |
| `src/agent/tools/` 目录 | 所有工具迁移到Toolkit |
| `api/routers/agent.py` | 统一入口为mdagents路由 |

**验收标准**：
- [ ] 项目中不存在`AgentManager`、`AgentFactory`、`base_agent.py`、`medical_agent.py`、`agent_manager.py`、`agent_factory.py`、`agent.py`（路由）、`tool_manager.py`
- [ ] `src/agent/tools/`目录已删除
- [ ] `api/routers/mdagents.py`是唯一Agent相关路由
- [ ] `api/services/mdagents_service.py`是唯一Agent服务入口
- [ ] 访问`/api/agent/*`返回404

---

### 第四阶段：测试与文档

**目标**：确保功能完整，文档齐全

**任务**：
1. [ ] 更新`api/services/mdagents_service.py`中MedicalAgent到AgentScope Agent的映射说明
2. [ ] 更新README中Agent模块的架构说明
3. [ ] 编写AgentScope + REME使用指南
4. [ ] 确保所有现有测试通过
5. [ ] 测试多轮对话RAG检索正常
6. [ ] 测试长对话记忆压缩正常

**具体测试用例（必须全部通过）**：

| 编号 | 测试用例 | 验证方法 | 预期结果 |
|------|---------|---------|---------|
| T1 | 健康检查 | `GET /api/mdagents/health` | 返回 `{"status": "healthy"}` |
| T2 | 复杂度分析 | `POST /api/mdagents/analyze` + 简单问题 | 复杂度 = LOW |
| T3 | 单专家会诊 | `POST /api/mdagents/consult` + 低复杂度问题 | 单specialist参与，RAG上下文被注入 |
| T4 | 多专家会诊 | `POST /api/mdagents/consult` + 高复杂度问题 | ≥2个specialist参与 |
| T5 | RAG检索验证 | 在T3/T4中检查日志 | specialist.respond()前调用rag_pipeline.query() |
| T6 | 记忆压缩 | 连续15轮对话后检查`.reme/memory/` | 存在压缩后的.md文件 |
| T7 | 会话持久化 | 停止服务后重启，检查`.reme/dialog/` | 存在历史会话JSONL |
| T8 | 路由清理验证 | `GET /api/agent/...`（任意agent路由） | 返回404 |
| T9 | 会诊流程完整性 | 高复杂度问题走完整个流程 | 状态机经历所有状态并正常结束 |

---

## 架构对比

### 改造前
```
api/routers/
├── mdagents.py  →  api/services/mdagents_service.py  →  MainController
                                                              ↓
                                                      MDAgents Agents (6种)
                                                              ↓
                                                    api_integration (独立封装)

agent/ (独立体系)
├── agent_manager.py
├── agent_factory.py
├── base_agent.py
├── medical_agent.py
├── multi_agent_system.py
```

### 改造后
```
api/routers/
└── mdagents.py  →  api/services/mdagents_service.py  →  MainController
                                                              ↓
                                                        MsgHub (AgentScope)
                                                              ↓
                    ┌──────────────────────────────────────────┼──────────────────────────────────────────┐
                    ↓                                          ↑                                          ↓
           specialist_1 (ReActAgent)              specialist_N (ReActAgent)                  moderator (ReActAgent)
           ├── rag_pipeline.query()               ├── rag_pipeline.query()                    └── 引导讨论流程
           ├── ReMeInMemoryMemory                 ├── ReMeInMemoryMemory
           └── 回答前RAG触发                       └── 回答前RAG触发
```

---

## 关键设计决策

1. **保留MDAgents核心流程**：复杂度分析→协作分配→专家招募→共识达成→报告生成，这套流程不动

2. **AgentScope作为底层框架**：所有Agent继承自AgentScope的`ReActAgent`，享受其工具调用、记忆管理、多Agent编排等能力

3. **MainController主持，MsgHub执行**：MainController保持为状态机协调器，通过MsgHub广播/定向消息与各Agent通信

4. **RAG仅在specialist回答前触发**：
   - pcc广播问题：无需RAG（只是分发）
   - specialist收到任务并回答：**触发RAG**（注入医学知识上下文）
   - moderator引导讨论：无需RAG（流程控制）
   - integrator汇总报告：**触发RAG**（如有需要）

5. **REME解决记忆爆炸**：长对话自动压缩，避免context window溢出，使用`pre_reasoning_hook`钩子自动化处理

6. **统一服务入口**：所有Agent能力通过`mdagents_service.py`暴露，删除原有独立Agent入口

7. **版本锁定策略**：锁定 agentscope==1.2.8, reme-ai==0.3.2，避免 2.0 破坏性变更

8. **ToolManager → AgentScope Toolkit**：原`tool_manager.py`被AgentScope的`Toolkit.register_tool_function()`替代，所有工具需重新注册

---

## 风险与对策

| 风险 | 级别 | 对策 |
|------|------|------|
| AgentScope 2.0 API 破坏性变更 | **高** | 锁定版本 1.2.8，升级前完成完整回归测试 |
| REME压缩导致信息丢失 | **中** | 配置较大`max_total_tokens=20000`，保留足够上下文 |
| RAG引入延迟 | **低** | 异步调用，结果可缓存；仅在specialist回答前触发 |
| 多Agent协作复杂度高 | **中** | 保持MsgHub简单用法（广播/定向），逐步增加复杂性 |
| MainController与MsgHub职责重叠 | **中** | 明确MainController为状态机协调器，MsgHub为消息路由 |
| 与现有models.api_model冲突 | **中** | AgentScope的model接口与BaseGenerativeModel需做适配层 |
| 工具迁移工作量 | **低** | 工具数量有限，逐个迁移到Toolkit并测试 |

---

## 参考资料

| 资源 | 链接 |
|------|------|
| AgentScope GitHub | https://github.com/agentscope-ai/agentscope |
| AgentScope文档 | https://doc.agentscope.io/ |
| REME GitHub | https://github.com/agentscope-ai/ReMe |
| REME中文README | https://github.com/agentscope-ai/ReMe/blob/main/README_ZH.md |
| REME深度设计文档 | https://github.com/agentscope-ai/ReMe/wiki/ReMeV2-%E6%B7%B1%E5%BA%A6%E8%AE%BE%E8%AE%A1%E6%96%87%E6%A1%A3 |
| AgentScope记忆模块博客 | https://www.cnblogs.com/peacemaple/p/19514254 |
| REME Short-Term Memory示例 | https://github.com/agentscope-ai/agentscope/tree/main/examples/functionality/short_term_memory/reme |
| AgentScope Memory模块 | https://github.com/agentscope-ai/agentscope/tree/main/src/agentscope/memory |