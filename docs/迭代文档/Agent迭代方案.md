# Agent体系融合迭代方案（v2）

## 背景

当前项目存在两套并行的Agent系统：
- **单Agent体系**（`src/agent/base_agent.py` → `medical_agent.py` → `agent_factory.py` → `agent_manager.py`）：基于LangChain的单Agent架构，依赖`ToolManager`，支持plan/reflect/refine流程
- **MDAgents体系**（`src/agent/mdagents/`）：多Agent协作系统，包含复杂度分析、专家招募、共识机制等会诊流程

两套系统完全独立，存在以下核心问题：
1. MDAgents的`_get_agent_diagnosis()`是**硬编码stub**，不调用任何LLM
2. `ApiIntegrationManager`基于项目自有的`ApiModel`，不适合AgentScope生态
3. `MDAgentsService`使用`ThreadPoolExecutor`包裹`asyncio.run()`，存在事件循环冲突
4. 单Agent体系的`AgentFactory`引用了3个不存在的工具模块（`MedicalReferenceTool`、`ReActAgentTool`、`MedicalAssessmentTool`）
5. `MainController`没有真正的状态机实现，仅靠字符串字段`system_status`标记状态
6. RAG检索能力未接入MDAgents

## 目标

1. **删除整个单Agent体系**，保留MDAgents作为唯一Agent架构
2. 使用**AgentScope框架**重构MDAgents中的所有Agent
3. 所有LLM调用统一使用**DashScopeChatModel**，废弃`ApiIntegrationManager`和`ApiModel`依赖
4. 将`rag_pipeline`封装为**AgentScope Tool**，通过`Toolkit.register_tool_function()`注册
5. Agent间通信固定为**specialist → moderator → integrator**协议，通过**MsgHub**传递
6. 使用**REME记忆系统**（ReMeLight）管理对话上下文
7. `MDAgentsService`统一使用**asyncio**，废弃`ThreadPoolExecutor`
8. 在`MainController`中实现真正的**有限状态机**
9. 对外仅暴露`api/services/mdagents_service.py`统一入口

---

## 技术选型

### 1. AgentScope框架

- **GitHub**: https://github.com/agentscope-ai/agentscope
- **文档**: https://doc.agentscope.io/
- **要求**：Python 3.10+

**安装**：
```bash
pip install agentscope==1.2.8
```

**⚠️ AgentScope 2.0 兼容性说明**：
- AgentScope 2.0 预计 2026-04 发布（[Roadmap](https://github.com/agentscope-ai/agentscope/discussions)）
- 2.0 可能存在 API 破坏性变更
- **本方案锁定版本为 1.2.8**，待 2.0 稳定后再考虑升级
- 升级前需完成完整的回归测试

**本方案使用的核心组件**：

| 组件 | 说明 | 本项目用途 |
|------|------|-----------|
| `ReActAgent` | 内置ReAct推理Agent | 6种MDAgent的基类 |
| `MsgHub` | 多Agent编排中枢 | specialist → moderator → integrator 消息通信 |
| `Toolkit` | 工具注册管理 | 注册RAG检索工具 |
| `DashScopeChatModel` | 阿里云DashScope LLM接口 | **唯一**LLM调用方式 |
| `Msg` | 统一消息格式 | Agent间消息传递 |

### 2. REME记忆系统（ReMeLight）

- **GitHub**: https://github.com/agentscope-ai/ReMe

**安装**：
```bash
pip install reme-ai==0.3.2 python-dotenv
```

**本方案选择ReMeLight**（基于文件的记忆系统），而非基于向量库的ReMe。原因：
- 项目已有独立的向量检索体系（Milvus/FAISS），不需要REME再做向量检索
- ReMeLight的对话压缩和持久化能力是本项目需要的核心功能
- 更轻量，不引入额外的向量库依赖冲突

### 3. LLM 配置来源

Agent 的 LLM API 参数**统一从项目已有的 `config.yaml` 的 `agent` 节读取**，通过 `settings.agent` 访问：

```yaml
# src/config/config.yaml（已有，无需新增）
agent:
  max_iterations: 10
  tool_timeout: 30
  enable_auto_tool_choice: false
  model_name: ${MODEL_NAME}       # ← DashScopeChatModel 的 model_name
  api_key: ${API_KEY}             # ← DashScopeChatModel 的 api_key
  base_url: ${BASE_URL}           # ← DashScopeChatModel 的 base_url（用于兼容非 DashScope 端点）
  max_tokens: 2048
  temperature: 0.7
  timeout: 60
```

**配置读取链路**：
```
.env 中设置 MODEL_NAME / API_KEY / BASE_URL
    ↓ 环境变量替换
config.yaml → settings.agent（OmegaConf 单例）
    ↓ llm_factory.create_llm() 读取 settings.agent
DashScopeChatModel(model_name=..., api_key=..., ...)
```

**不新增任何环境变量**。Agent 使用的 `MODEL_NAME`、`API_KEY`、`BASE_URL` 与项目其他模块（如 `settings.llm`）共享同一批环境变量，但通过 `settings.agent` 节独立读取，允许 Agent 使用与其他模块不同的模型/参数。

---

## 前置条件：AgentScope PoC 验证

**⚠️ 在正式开工前，必须先完成以下 PoC 验证**，确保技术假设成立：

| 编号 | 验证项 | 验证方式 | 通过标准 |
|------|--------|---------|---------|
| P1 | `agentscope==1.2.8` 可安装且与项目依赖兼容 | `pip install` + `import agentscope` | 无冲突 |
| P2 | `ReActAgent` 支持子类化并注入自定义逻辑 | 写最小子类，覆盖 `__call__` 或 `reply` | 自定义逻辑被执行 |
| P3 | `DashScopeChatModel` 能通过 `settings.agent` 的配置正常调用 LLM API | 用 `llm_factory.create_llm()` 发送简单消息 | 获得合法响应 |
| P4 | `MsgHub` 支持有序的多Agent消息传递 | 3个Agent通过MsgHub按序通信 | 消息按 A→B→C 顺序到达 |
| P5 | `Toolkit.register_tool_function()` 可注册自定义函数 | 注册一个简单函数，Agent调用 | 工具被正确调用 |
| P6 | `ReMeInMemoryMemory` 可与 `ReActAgent` 配合 | 创建带REME记忆的Agent，进行多轮对话 | 记忆压缩正常触发 |

**PoC 脚本位置**：`tests/poc_agentscope.py`

---

## 现有代码问题清单

在迭代过程中需要逐一解决的已知问题：

| 编号 | 问题 | 位置 | 影响 |
|------|------|------|------|
| E1 | `_get_agent_diagnosis()` 返回硬编码数据，不调用LLM | `main_controller.py:301-328` | 会诊流程无实际推理 |
| E2 | `AgentFactory` 引用3个不存在的工具模块 | `agent_factory.py:13-15` | import 即报错 |
| E3 | `MDAgentsService` 用 `ThreadPoolExecutor` 包裹 `asyncio.run()` | `mdagents_service.py:112-115` | 事件循环冲突 |
| E4 | `MainController` 无状态机实现 | `main_controller.py:48` | 流程状态不可追踪 |
| E5 | MDAgents的BaseAgent不调用LLM（`api_wrapper`初始化但未被诊断流程使用） | `mdagents/agents/base_agent.py:37-45` | Agent推理为空壳 |
| E6 | `api/routers/agent.py` 不存在（原方案误写） | 无 | 无需删除 |

---

## 迭代方案

### 第零阶段：PoC 验证与准备（预计 1 天）

**目标**：验证技术选型可行，打 git tag 保留回退点

**任务**：
1. [ ] 打 git tag `pre-agent-refactor` 保留当前代码快照
2. [ ] 运行 PoC 验证脚本（P1-P6，见前置条件）
3. [ ] 确认 `agentscope==1.2.8` 和 `reme-ai==0.3.2` 与现有 `requirements.txt` 无冲突
4. [ ] 确认 `ReActAgent` 的实际扩展方式（`reply()` vs `__call__()` vs 其他）
5. [ ] 确认 `MsgHub` 的实际 API（是否支持有序传递、是否需要主持人角色）
6. [ ] 将验证结果记录到 `docs/迭代文档/PoC验证记录.md`

**验收标准**：
- [ ] 6 项 PoC 全部通过
- [ ] `docs/迭代文档/PoC验证记录.md` 包含每项验证的代码片段和结果截图
- [ ] 记录 `ReActAgent` 实际的扩展 API 名称（后续阶段使用）

**⚠️ 阻塞规则**：任何一项 PoC 不通过，需重新评估技术选型后才能继续。

---

### 第一阶段：LLM层统一与stub消除（预计 2 天）

**目标**：用 `DashScopeChatModel` 替代所有 LLM 调用方式，消除 MainController 中的 stub 代码

**1.1 创建统一 LLM 工厂**

创建 `src/agent/mdagents/core/llm_factory.py`：

```python
from agentscope.model import DashScopeChatModel
from config.settings import settings

def create_llm(**overrides) -> DashScopeChatModel:
    """
    创建 DashScopeChatModel 实例（项目唯一的 LLM 构造入口）。
    
    参数全部从 config.yaml 的 agent 节读取（settings.agent），
    可通过 overrides 覆盖个别参数。
    """
    agent_cfg = settings.agent
    return DashScopeChatModel(
        model_name=overrides.get("model_name", agent_cfg.model_name),
        api_key=overrides.get("api_key", agent_cfg.api_key),
        api_url=overrides.get("base_url", getattr(agent_cfg, "base_url", None)),
        generate_args={
            "max_tokens": overrides.get("max_tokens", agent_cfg.max_tokens),
            "temperature": overrides.get("temperature", agent_cfg.temperature),
        },
    )
```

> **说明**：`settings.agent` 来自 `src/config/config.yaml` 的 `agent` 节，其中 `model_name`、`api_key`、`base_url` 均通过 `${MODEL_NAME}`、`${API_KEY}`、`${BASE_URL}` 环境变量注入。不新增任何环境变量，复用项目已有的 `.env` 配置。

**⚠️ 依赖注入原则：谁调用 `llm_factory`？**

`llm_factory.create_llm()` **只被 `MainController` 调用**，其他所有组件（Agent、REMEManager）通过构造函数参数接收 LLM 实例：

```
MainController.__init__()
    │
    ├── self.llm = create_llm()              ← 唯一调用 llm_factory 的地方
    │
    ├── self.reme_manager = REMEManager(model=self.llm)    ← 注入
    │
    └── 创建 Agent 时:
        ├── SpecialistAgent(model=self.llm, toolkit=..., memory=...) ← 注入
        ├── ModeratorAgent(model=self.llm, memory=...)               ← 注入
        └── IntegratorAgent(model=self.llm, toolkit=..., memory=...) ← 注入
```

**理由**：
- Agent 不应知道 LLM 是如何构造的（关注点分离）
- 同一次会诊的所有 Agent 共享同一个 `DashScopeChatModel` 实例（避免重复创建）
- 便于测试：单测时可传入 mock model，不触发真实 API 调用
- 便于差异化：如需给 specialist 用不同 temperature，由 MainController 用 `create_llm(temperature=0.3)` 创建专用实例

**1.2 消除 `_get_agent_diagnosis()` stub**

当前代码（**必须替换**）：
```python
# main_controller.py:309-324 —— 硬编码 stub
diagnosis_data = {
    "primary_diagnosis": f"{agent.name}的初步诊断",
    "confidence": 0.8,
    ...
}
await asyncio.sleep(0.1)
return diagnosis_data
```

替换为真实 LLM 调用：
```python
async def _get_agent_diagnosis(self, agent, query, patient_info):
    """通过 DashScopeChatModel 获取真实诊断"""
    prompt = agent.build_diagnosis_prompt(query, patient_info)
    messages = [
        {"role": "system", "content": agent.get_system_prompt()},
        {"role": "user", "content": prompt}
    ]
    response = self.llm(messages)
    return agent.parse_diagnosis_response(response.text)
```

**1.3 废弃 `ApiIntegrationManager`**

`src/agent/mdagents/core/api_integration.py` 中的 `ApiIntegrationManager` 和 `ApiModelWrapper` 将被 `DashScopeChatModel` 完全替代。保留 `api_integration.py` 中的**提示词模板**（`_initialize_prompt_templates` 的内容），迁移到独立的 `src/agent/mdagents/prompts/templates.py`。

**任务**：
1. [ ] 创建 `llm_factory.py`，封装 `DashScopeChatModel` 构造（从 `settings.agent` 读配置）
2. [ ] 在 `MainController.__init__()` 中通过 `llm_factory.create_llm()` 创建 LLM 实例（**MainController 是唯一调用 `llm_factory` 的地方**）
3. [ ] LLM 实例通过构造函数注入到各 Agent 和 REMEManager，Agent 不 import `llm_factory`
4. [ ] 替换 `_get_agent_diagnosis()` 的 stub 为真实 LLM 调用（使用 `self.llm`）
5. [ ] 替换 `_generate_final_report()` 的 stub 为真实 LLM 调用（使用 `self.llm`）
6. [ ] 将 `api_integration.py` 中的提示词模板迁移到 `prompts/templates.py`
7. [ ] MDAgents的每个Agent（pcc, specialist, moderator, recruiter, reviewer, integrator）的 `process_query()` 方法改为使用注入的 LLM 实例

**验收标准**：
- [ ] `MainController.process_medical_query()` 对简单问题返回真实 LLM 生成内容（非硬编码）
- [ ] 项目内不再有 `ApiModel`/`ApiIntegrationManager`/`ApiModelWrapper` 的 import
- [ ] `config.yaml` 的 `agent` 节配置正确后（`MODEL_NAME`/`API_KEY`/`BASE_URL` 环境变量），6种Agent均能产生真实诊断输出

---

### 第二阶段：Agent 重构为 AgentScope（预计 3 天）

**目标**：将 MDAgents 的 6 种 Agent 迁移为 AgentScope `ReActAgent` 子类

**2.1 新 Agent 基类设计**

创建 `src/agent/mdagents/agents/agentscope_base.py`：

```python
from agentscope.agent import ReActAgent
from agentscope.model import DashScopeChatModel
from agentscope.message import Msg
from agentscope.tool import Toolkit
from agentscope.memory import ReMeInMemoryMemory

class MedicalBaseAgent(ReActAgent):
    """MDAgents 所有 Agent 的新基类，继承 AgentScope ReActAgent。
    
    LLM 实例通过构造函数注入，Agent 本身不依赖 llm_factory。
    """

    def __init__(
        self,
        name: str,
        model: DashScopeChatModel,    # ← 由调用方（MainController）注入
        sys_prompt: str,
        specialty: str = None,
        toolkit: Toolkit = None,
        memory: ReMeInMemoryMemory = None,
        max_iters: int = 10,
        **kwargs
    ):
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model=model,              # ← 直接使用注入的 LLM，不自行创建
            toolkit=toolkit or Toolkit(),
            memory=memory,
            max_iters=max_iters,
            **kwargs
        )
        self.specialty = specialty
```

> **注意**：`ReActAgent` 的实际扩展方法名需根据第零阶段 PoC 结果确认。上述代码中的 `__init__` 参数名以 AgentScope 1.2.8 文档为准。
>
> **关键设计**：`MedicalBaseAgent` **不 import `llm_factory`**。`model` 参数由 `MainController` 创建并传入，符合依赖注入原则。

**2.2 RAG 封装为 AgentScope Tool**

将 `RAGPipeline.query()` 封装为可注册的工具函数：

```python
# src/agent/mdagents/tools/rag_tool.py

from agentscope.tool import Toolkit
from rag.rag_pipeline import RAGPipeline

_rag_pipeline = None

def get_rag_pipeline() -> RAGPipeline:
    """懒加载 RAGPipeline 单例"""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline

def medical_knowledge_search(query: str, top_k: int = 5) -> str:
    """
    搜索医学知识库，获取与查询相关的医学文献和知识。

    Args:
        query: 医学查询文本，如症状描述、疾病名称、药物名称等
        top_k: 返回的相关文档数量，默认5

    Returns:
        检索到的相关医学知识文本
    """
    pipeline = get_rag_pipeline()
    result = pipeline.query(query, top_k=top_k)
    if isinstance(result, dict):
        return result.get("context", result.get("response", str(result)))
    return str(result)

def create_medical_toolkit() -> Toolkit:
    """创建包含 RAG 检索工具的 Toolkit"""
    toolkit = Toolkit()
    toolkit.register_tool_function(medical_knowledge_search)
    return toolkit
```

**2.3 迁移 6 种 Agent**

每种 Agent 从继承 `mdagents/agents/base_agent.py`（旧BaseAgent）改为继承 `MedicalBaseAgent`（AgentScope）：

| 旧 Agent | 新 Agent | 是否注册 RAG Tool | 说明 |
|----------|---------|------------------|------|
| `PCCAgent` | `PCCAgent(MedicalBaseAgent)` | 否 | 只做问题分发 |
| `SpecialistAgent` | `SpecialistAgent(MedicalBaseAgent)` | **是** | 回答前检索知识 |
| `ModeratorAgent` | `ModeratorAgent(MedicalBaseAgent)` | 否 | 流程控制 |
| `RecruiterAgent` | `RecruiterAgent(MedicalBaseAgent)` | 否 | 专家招募决策 |
| `ReviewerAgent` | `ReviewerAgent(MedicalBaseAgent)` | 否 | 质量审查 |
| `IntegratorAgent` | `IntegratorAgent(MedicalBaseAgent)` | **是**（可选） | 汇总报告，必要时补充检索 |

**2.4 REME 记忆系统集成**

创建 `src/agent/mdagents/memory/reme_manager.py`：

```python
from agentscope.model import DashScopeChatModel
from reme.light import ReMeLight
from agentscope.memory import ReMeInMemoryMemory

class REMEManager:
    """REME 记忆管理器，为每个会话创建独立的记忆上下文。
    
    LLM 实例通过构造函数注入，不依赖 llm_factory。
    """

    def __init__(self, model: DashScopeChatModel, working_dir: str = ".reme"):
        self.reme_light = ReMeLight(working_dir=working_dir, model=model)

    def create_session_memory(
        self,
        max_total_tokens: int = 20000,
        max_tool_message_tokens: int = 2000,
        keep_recent_count: int = 10,
    ) -> ReMeInMemoryMemory:
        """为一次会诊会话创建共享记忆实例"""
        return self.reme_light.get_in_memory_memory(
            working_summary_mode="auto",
            compact_ratio_threshold=0.75,
            max_total_tokens=max_total_tokens,
            max_tool_message_tokens=max_tool_message_tokens,
            keep_recent_count=keep_recent_count,
        )
```

**关键设计：会话级共享记忆**

同一次会诊中的所有 Agent **共享同一个 `ReMeInMemoryMemory` 实例**。这样：
- specialist 的推理结果自动可被 moderator 看到
- moderator 的讨论结果自动可被 integrator 整合
- 记忆压缩统一管理，避免重复存储

```
MainController.__init__():
  self.llm = create_llm()                              ← 唯一调 llm_factory
  self.reme_manager = REMEManager(model=self.llm)       ← 注入 LLM

一次会诊会话（MainController.process_medical_query）:
  session_memory = self.reme_manager.create_session_memory()
      ↓ 共享
  specialist_1 = SpecialistAgent(model=self.llm, memory=session_memory, ...)   ← 注入
  specialist_2 = SpecialistAgent(model=self.llm, memory=session_memory, ...)   ← 注入
  moderator    = ModeratorAgent(model=self.llm, memory=session_memory, ...)    ← 注入
  integrator   = IntegratorAgent(model=self.llm, memory=session_memory, ...)   ← 注入
```

**任务**：
1. [ ] 创建 `agentscope_base.py`，定义 `MedicalBaseAgent`
2. [ ] 创建 `tools/rag_tool.py`，封装 `medical_knowledge_search` 并通过 `Toolkit.register_tool_function()` 注册
3. [ ] 创建 `memory/reme_manager.py`，封装 REME 会话记忆
4. [ ] 迁移 `PCCAgent` → 继承 `MedicalBaseAgent`
5. [ ] 迁移 `SpecialistAgent` → 继承 `MedicalBaseAgent` + 注册 RAG Tool
6. [ ] 迁移 `ModeratorAgent` → 继承 `MedicalBaseAgent`
7. [ ] 迁移 `RecruiterAgent` → 继承 `MedicalBaseAgent`
8. [ ] 迁移 `ReviewerAgent` → 继承 `MedicalBaseAgent`
9. [ ] 迁移 `IntegratorAgent` → 继承 `MedicalBaseAgent` + 可选 RAG Tool
10. [ ] 删除旧的 `src/agent/mdagents/agents/base_agent.py`（被 `agentscope_base.py` 替代）

**验收标准**：
- [ ] 6 种 Agent 均继承自 `MedicalBaseAgent(ReActAgent)`
- [ ] `SpecialistAgent` 调用时自动通过工具检索 RAG 知识
- [ ] 同一会话中各 Agent 的记忆是共享的
- [ ] 旧 `base_agent.py` 已删除，无残留引用

---

### 第三阶段：MsgHub 通信与状态机（预计 2 天）

**目标**：实现 specialist → moderator → integrator 的有序消息传递协议，并在 MainController 中实现真正的有限状态机

**3.1 消息传递协议设计**

Agent 间通信**固定**为以下协议，通过 MsgHub 实现：

```
┌────────────────────────────────────────────────────────────────┐
│                    会诊消息流（MsgHub）                          │
│                                                                │
│  Phase 1: 分发                                                 │
│    pcc ──broadcast──▶ [specialist_1, specialist_2, ...]        │
│                                                                │
│  Phase 2: 专家诊断（RAG Tool 在此阶段被 specialist 自动调用）    │
│    specialist_1 ──msg──▶ MsgHub                                │
│    specialist_2 ──msg──▶ MsgHub                                │
│                                                                │
│  Phase 3: 主持讨论                                              │
│    moderator ◀── 收集所有 specialist 消息                       │
│    moderator ──summary──▶ MsgHub                               │
│                                                                │
│  Phase 4: 整合报告                                              │
│    integrator ◀── 收集 moderator summary                       │
│    integrator ──final_report──▶ MsgHub                         │
│                                                                │
│  Phase 5: 质量审查                                              │
│    reviewer ◀── final_report                                   │
│    reviewer ──review_result──▶ MainController                  │
└────────────────────────────────────────────────────────────────┘
```

**MsgHub 集成代码模式**：

```python
from agentscope.msghub import MsgHub
from agentscope.message import Msg

async def run_consultation(self, query: str, specialists: list, moderator, integrator, reviewer):
    """通过 MsgHub 执行完整会诊流程"""
    participants = specialists + [moderator, integrator, reviewer]

    with MsgHub(participants=participants) as hub:
        # Phase 1: PCC 广播问题
        question_msg = Msg(role="user", content=query, name="pcc")
        hub.broadcast(question_msg)

        # Phase 2: 各 specialist 依次回答（内部自动调用 RAG Tool）
        specialist_responses = []
        for specialist in specialists:
            response = await specialist(question_msg)
            specialist_responses.append(response)

        # Phase 3: moderator 汇总讨论
        moderator_input = Msg(
            role="assistant",
            content=self._format_specialist_responses(specialist_responses),
            name="moderator_input"
        )
        moderator_summary = await moderator(moderator_input)

        # Phase 4: integrator 生成最终报告
        final_report = await integrator(moderator_summary)

        # Phase 5: reviewer 质量审查
        review_result = await reviewer(final_report)

    return final_report, review_result
```

**3.2 有限状态机实现**

在 `MainController` 中实现真正的状态机，替代当前的字符串字段：

```python
from enum import Enum, auto

class ConsultationState(Enum):
    """会诊流程状态"""
    IDLE = auto()
    ANALYZING = auto()      # 复杂度分析中
    ALLOCATING = auto()     # 协作分配中
    RECRUITING = auto()     # 专家招募中
    CONSULTING = auto()     # 专家诊断中（MsgHub Phase 2）
    MODERATING = auto()     # 主持讨论中（MsgHub Phase 3）
    INTEGRATING = auto()    # 整合报告中（MsgHub Phase 4）
    REVIEWING = auto()      # 质量审查中（MsgHub Phase 5）
    COMPLETED = auto()      # 完成
    ERROR = auto()          # 错误

# 合法状态转换表
VALID_TRANSITIONS = {
    ConsultationState.IDLE:        [ConsultationState.ANALYZING],
    ConsultationState.ANALYZING:   [ConsultationState.ALLOCATING, ConsultationState.ERROR],
    ConsultationState.ALLOCATING:  [ConsultationState.RECRUITING, ConsultationState.ERROR],
    ConsultationState.RECRUITING:  [ConsultationState.CONSULTING, ConsultationState.ERROR],
    ConsultationState.CONSULTING:  [ConsultationState.MODERATING, ConsultationState.ERROR],
    ConsultationState.MODERATING:  [ConsultationState.INTEGRATING, ConsultationState.ERROR],
    ConsultationState.INTEGRATING: [ConsultationState.REVIEWING, ConsultationState.ERROR],
    ConsultationState.REVIEWING:   [ConsultationState.COMPLETED, ConsultationState.CONSULTING, ConsultationState.ERROR],
    ConsultationState.COMPLETED:   [ConsultationState.IDLE],
    ConsultationState.ERROR:       [ConsultationState.IDLE],
}

class StateMachine:
    """会诊流程状态机"""

    def __init__(self):
        self._state = ConsultationState.IDLE
        self._history: list[tuple[ConsultationState, ConsultationState, str]] = []

    @property
    def state(self) -> ConsultationState:
        return self._state

    def transition(self, target: ConsultationState, reason: str = ""):
        """执行状态转换，不合法则抛异常"""
        if target not in VALID_TRANSITIONS.get(self._state, []):
            raise StateTransitionError(
                f"非法状态转换: {self._state.name} → {target.name}"
            )
        old = self._state
        self._state = target
        self._history.append((old, target, reason))

    def reset(self):
        self._state = ConsultationState.IDLE

class StateTransitionError(Exception):
    pass
```

**3.3 MainController 重构**

`MainController.process_medical_query()` 重写为状态机驱动：

```python
async def process_medical_query(self, query, patient_info=None):
    session_id = str(uuid.uuid4())
    sm = StateMachine()

    try:
        # IDLE → ANALYZING
        sm.transition(ConsultationState.ANALYZING, "开始复杂度分析")
        complexity = await self._analyze_complexity(query, patient_info)

        # ANALYZING → ALLOCATING
        sm.transition(ConsultationState.ALLOCATING, f"复杂度={complexity['level']}")
        plan = await self._allocate_collaboration(complexity, query)

        # ALLOCATING → RECRUITING
        sm.transition(ConsultationState.RECRUITING, f"协作模式={plan['pattern']}")
        session_memory = self.reme_manager.create_session_memory()
        specialists = self._create_specialists(plan, self.llm, session_memory)  # 注入 LLM

        # RECRUITING → CONSULTING
        sm.transition(ConsultationState.CONSULTING, f"招募{len(specialists)}位专家")

        # CONSULTING → MODERATING → INTEGRATING → REVIEWING（MsgHub 内部流转）
        final_report, review = await self._run_msghub_consultation(
            query, specialists, session_memory, sm
        )

        # REVIEWING → COMPLETED
        sm.transition(ConsultationState.COMPLETED, "会诊完成")

        return {
            "session_id": session_id,
            "status": "success",
            "final_report": final_report,
            "review": review,
            "state_history": [(s.name, t.name, r) for s, t, r in sm._history],
        }

    except Exception as e:
        sm.transition(ConsultationState.ERROR, str(e))
        return {"session_id": session_id, "status": "error", "error": str(e)}
```

**任务**：
1. [ ] 创建 `src/agent/mdagents/core/state_machine.py`，实现 `ConsultationState`、`VALID_TRANSITIONS`、`StateMachine`
2. [ ] 在 `MainController` 中引入 `StateMachine`，驱动 `process_medical_query()` 流程
3. [ ] 实现 `_run_msghub_consultation()` 方法，按 specialist → moderator → integrator → reviewer 顺序通过 MsgHub 通信
4. [ ] 删除旧的 `_execute_collaborative_diagnosis()` 和 `_get_agent_diagnosis()` stub 方法
5. [ ] 删除旧的 `_achieve_consensus()` 方法（共识逻辑融入 moderator Agent 的推理）
6. [ ] `reviewer` 的审查结果若不通过，状态机支持回退到 `CONSULTING` 重新诊断

**验收标准**：
- [ ] 非法状态转换抛出 `StateTransitionError`
- [ ] 会诊结果中包含完整的 `state_history` 记录
- [ ] MsgHub 中消息按 specialist → moderator → integrator → reviewer 顺序传递
- [ ] 日志中可追踪每个状态转换的时间点和原因

---

### 第四阶段：MDAgentsService 异步化与接口清理（预计 2 天）

**目标**：MDAgentsService 统一为 asyncio，删除单Agent体系全部代码

**4.1 MDAgentsService 异步化**

**当前问题**（`mdagents_service.py:112-115`）：
```python
# 反模式：在 async 函数中用 executor 包裹 asyncio.run()
loop = asyncio.get_event_loop()
result = await loop.run_in_executor(
    self.executor,
    lambda: asyncio.run(self.controller.process_medical_query(...))
)
```

**改为原生 asyncio**：
```python
async def process_medical_query(self, request):
    self._ensure_initialized()
    result = await self.controller.process_medical_query(
        request.query, request.patient_context
    )
    return self._build_response(result)
```

**4.2 单例管理改为 asyncio 兼容**

```python
# 废弃 ThreadPoolExecutor 和 threading.Lock
# 改用 asyncio.Lock

_mdagents_service: Optional[MDAgentsService] = None
_service_lock = asyncio.Lock()

async def get_mdagents_service() -> MDAgentsService:
    global _mdagents_service
    if _mdagents_service is None:
        async with _service_lock:
            if _mdagents_service is None:
                _mdagents_service = MDAgentsService()
                await _mdagents_service.initialize()
    return _mdagents_service
```

**4.3 删除模块清单**

| 文件/目录 | 删除原因 |
|----------|---------|
| `src/agent/base_agent.py` | 单Agent体系入口，被AgentScope替代 |
| `src/agent/medical_agent.py` | 单Agent实现，功能融入MDAgents |
| `src/agent/agent_factory.py` | 工厂模式，AgentScope自动管理；且引用3个不存在的工具 |
| `src/agent/agent_manager.py` | 管理器模式，不再需要 |
| `src/agent/multi_agent_system.py` | 4000+行的独立多Agent系统，与MDAgents重复 |
| `src/agent/tools/tool_manager.py` | 被AgentScope `Toolkit` 替代 |
| `src/agent/tools/tool_base.py` | 被AgentScope工具体系替代 |
| `src/agent/tools/search_tool.py` | 迁移为AgentScope Tool后删除 |
| `src/agent/tools/calculator_tool.py` | 迁移为AgentScope Tool后删除 |
| `src/agent/mdagents/core/api_integration.py` | 被 `DashScopeChatModel` 替代（提示词模板已迁移） |
| `src/agent/mdagents/agents/base_agent.py` | 被 `agentscope_base.py` 替代（第二阶段已删除） |

**需保留的文件**：

| 文件 | 保留原因 |
|------|---------|
| `src/agent/mdagents/core/main_controller.py` | 核心协调器，已重构 |
| `src/agent/mdagents/core/complexity_analyzer.py` | 复杂度分析逻辑，不依赖Agent体系 |
| `src/agent/mdagents/core/collaboration_allocator.py` | 协作分配逻辑，不依赖Agent体系 |
| `src/agent/mdagents/core/consensus_mechanism.py` | 共识算法，仍可被moderator内部调用 |
| `src/agent/mdagents/config/` | 配置体系，不依赖Agent体系 |
| `src/rag/rag_pipeline.py` | RAG核心，被封装为Tool后仍需保留 |

**任务**：
1. [ ] 重写 `MDAgentsService`，全部方法改为原生 `async/await`，删除 `ThreadPoolExecutor`
2. [ ] 重写 `get_mdagents_service()` 为 `async` 版本，使用 `asyncio.Lock`
3. [ ] 更新 `src/api/routers/mdagents.py` 中的路由调用方式（配合异步单例）
4. [ ] 删除上述清单中的所有文件
5. [ ] 全局搜索并清理所有对已删除模块的 import 引用
6. [ ] 确认 `.env.example` 中已有的 `MODEL_NAME`、`API_KEY`、`BASE_URL` 环境变量说明覆盖 Agent 使用场景（无需新增变量）

**验收标准**：
- [ ] 项目中不存在以下类/模块的任何引用：`AgentManager`、`AgentFactory`、`BaseAgent`（单Agent的）、`MedicalAgent`、`ToolManager`、`ApiIntegrationManager`、`ApiModelWrapper`
- [ ] `src/agent/` 目录下仅保留 `mdagents/` 子目录和 `__init__.py`
- [ ] `MDAgentsService` 内部无 `ThreadPoolExecutor`、无 `run_in_executor`、无 `asyncio.run()`
- [ ] `python -c "from api.services.mdagents_service import MDAgentsService"` 无 import 错误
- [ ] `POST /api/mdagents/consult` 端点正常响应

---

### 第五阶段：测试与文档（预计 2 天）

**目标**：确保功能完整，编写测试和文档

**5.1 集成测试用例（必须全部通过）**

| 编号 | 测试用例 | 验证方法 | 预期结果 |
|------|---------|---------|---------|
| T1 | 健康检查 | `GET /api/mdagents/health` | `{"status": "healthy"}` |
| T2 | 简单问题会诊 | `POST /api/mdagents/consult` + "感冒了怎么办" | 单specialist参与，状态机经历全部状态，返回真实LLM生成内容 |
| T3 | 复杂问题会诊 | `POST /api/mdagents/consult` + 多症状复杂病例 | ≥2个specialist参与，RAG Tool被调用 |
| T4 | RAG Tool 验证 | 在T3日志中检查 | `medical_knowledge_search` 被specialist调用，返回非空上下文 |
| T5 | MsgHub 消息顺序 | 在T3日志中检查 | 消息按 specialist → moderator → integrator → reviewer 到达 |
| T6 | 状态机完整性 | 检查T2/T3返回的 `state_history` | 包含 IDLE→ANALYZING→...→COMPLETED 全部状态 |
| T7 | 状态机非法转换 | 单元测试：直接调用 `sm.transition(COMPLETED)` 从 IDLE | 抛出 `StateTransitionError` |
| T8 | 记忆压缩 | 连续15轮对话后检查 `.reme/` | 存在压缩后的记录 |
| T9 | 会话共享记忆 | 在T3中检查 | specialist的输出可被moderator引用 |
| T10 | LLM 配置来源 | 检查T2/T3的网络日志 + 断点确认 | 所有LLM参数来自 `settings.agent`，调用发往 `config.yaml agent.base_url` 指定的端点 |
| T11 | 旧模块清理 | `grep -r "AgentFactory\|AgentManager\|ApiIntegrationManager" src/` | 无结果 |
| T12 | 旧路由清理 | 项目中无 `api/routers/agent.py` | 确认不存在（本来就不存在） |

**5.2 文档更新**

1. [ ] 更新 `README.md` 中的 Agent 架构说明
2. [ ] 编写 `docs/迭代文档/AgentScope_REME使用指南.md`
3. [ ] 更新 `.env.example`，在 `MODEL_NAME`/`API_KEY`/`BASE_URL` 旁添加注释说明"Agent 模块通过 `config.yaml agent` 节使用此配置"
4. [ ] 记录本次迭代的所有 Breaking Changes

---

## 架构对比

### 改造前
```
src/agent/
├── base_agent.py              ← 单Agent体系（待删除）
├── medical_agent.py           ← 单Agent体系（待删除）
├── agent_factory.py           ← 引用3个不存在工具（待删除）
├── agent_manager.py           ← 待删除
├── multi_agent_system.py      ← 4000行冗余代码（待删除）
├── tools/
│   ├── tool_manager.py        ← 待删除
│   ├── tool_base.py           ← 待删除
│   ├── search_tool.py         ← 待删除
│   └── calculator_tool.py     ← 待删除
└── mdagents/
    ├── agents/
    │   └── base_agent.py      ← 旧BaseAgent（使用ApiModel）
    ├── core/
    │   ├── main_controller.py ← stub代码，无状态机
    │   └── api_integration.py ← ApiModel封装（待废弃）
    └── ...

LLM调用: ApiModel / ZhipuApiModel / OpenAIApiModel（分散）
异步模型: ThreadPoolExecutor + asyncio.run()（混乱）
Agent通信: asyncio.gather() 并行（无序）
RAG: 未接入MDAgents
```

### 改造后
```
src/agent/
└── mdagents/                          ← 唯一Agent目录
    ├── agents/
    │   ├── agentscope_base.py         ← MedicalBaseAgent(ReActAgent)
    │   ├── pcc_agent.py               ← PCCAgent(MedicalBaseAgent)
    │   ├── specialist_agent.py        ← SpecialistAgent + RAG Tool
    │   ├── moderator_agent.py         ← ModeratorAgent
    │   ├── recruiter_agent.py         ← RecruiterAgent
    │   ├── reviewer_agent.py          ← ReviewerAgent
    │   └── integrator_agent.py        ← IntegratorAgent + RAG Tool
    ├── core/
    │   ├── main_controller.py         ← StateMachine驱动 + MsgHub编排
    │   ├── state_machine.py           ← 有限状态机
    │   ├── llm_factory.py             ← DashScopeChatModel 工厂（仅 MainController 调用）
    │   ├── complexity_analyzer.py     ← 保留
    │   ├── collaboration_allocator.py ← 保留
    │   └── consensus_mechanism.py     ← 保留（moderator内部使用）
    ├── tools/
    │   └── rag_tool.py                ← RAGPipeline → AgentScope Tool
    ├── memory/
    │   └── reme_manager.py            ← REME 会话记忆管理
    ├── prompts/
    │   └── templates.py               ← 提示词模板（从api_integration迁移）
    └── config/                         ← 保留

LLM调用: DashScopeChatModel（llm_factory.py 构造，MainController 注入各 Agent，参数来自 config.yaml agent 节）
异步模型: 原生 asyncio（全链路）
Agent通信: MsgHub（specialist → moderator → integrator → reviewer）
RAG: AgentScope Tool（specialist/integrator 自动调用）
记忆: ReMeInMemoryMemory（会话级共享）
状态机: ConsultationState（IDLE → ANALYZING → ... → COMPLETED）
```

---

## 关键设计决策

1. **DashScopeChatModel 为唯一 LLM 调用方式**：废弃项目自有的 `ApiModel`/`ZhipuApiModel`/`OpenAIApiModel`。**LLM 参数全部从 `config.yaml` 的 `agent` 节读取**（`settings.agent.model_name`、`settings.agent.api_key`、`settings.agent.base_url` 等），不新增环境变量，复用项目已有的 `${MODEL_NAME}`/`${API_KEY}`/`${BASE_URL}`

2. **依赖注入，不直接依赖 `llm_factory`**：只有 `MainController` 调用 `llm_factory.create_llm()` 创建 LLM 实例，然后通过构造函数注入到各 Agent 和 REMEManager。Agent 不 import `llm_factory`，不知道 LLM 是如何构造的。好处：关注点分离、便于测试（可传 mock）、同一会诊共享实例避免浪费

3. **RAG 作为 AgentScope Tool 注册**：通过 `Toolkit.register_tool_function(medical_knowledge_search)` 注册，specialist 在 ReAct 推理循环中自动决定是否调用

4. **固定消息传递协议 specialist → moderator → integrator → reviewer**：通过 MsgHub 保证有序性，不使用 `asyncio.gather()` 做无序并行

5. **会话级共享记忆**：同一次会诊中所有 Agent 共享同一个 `ReMeInMemoryMemory` 实例，确保信息流通

6. **有限状态机驱动**：`MainController` 通过 `StateMachine` 管理流程，每次状态转换都经过合法性校验并记录历史

7. **全链路 asyncio**：从 `mdagents_service.py` 到 `MainController` 到各 Agent，全部使用原生 `async/await`

8. **ReMeLight（非 ReMe）**：选择基于文件的轻量版，避免与项目已有的 Milvus/FAISS 向量库产生依赖冲突

9. **版本锁定**：`agentscope==1.2.8`，`reme-ai==0.3.2`

---

## 风险与对策

| 风险 | 级别 | 对策 |
|------|------|------|
| AgentScope 2.0 API 破坏性变更 | **高** | 锁定 1.2.8；`pre-agent-refactor` tag 可回退 |
| `ReActAgent` 实际扩展 API 与方案假设不符 | **高** | 第零阶段 PoC 验证（P2）；不通过则重新评估 |
| MsgHub 不支持有序传递 | **中** | 第零阶段 PoC 验证（P4）；若不支持则手动编排 `await` 顺序 |
| DashScopeChatModel 与现有 RAGPipeline 的 model 依赖冲突 | **中** | RAGPipeline 仅作为 Tool 被调用，其内部 model 可独立于 AgentScope |
| REME 压缩导致关键诊断信息丢失 | **中** | 设置 `max_total_tokens=20000`，`keep_recent_count=10` |
| 多 specialist 并行时 MsgHub 消息乱序 | **中** | specialist 阶段改为串行 `for` 循环（牺牲速度换正确性） |
| `requirements.txt` 依赖冲突 | **低** | 第零阶段 PoC 验证（P1）；必要时创建独立虚拟环境 |
| 大规模删除代码后遗漏依赖 | **低** | git tag 回退；删除前全局搜索引用 |

---

## 工期估算

| 阶段 | 预计工期 | 前置依赖 |
|------|---------|---------|
| 第零阶段：PoC 验证 | 1 天 | 无 |
| 第一阶段：LLM 层统一 | 2 天 | 第零阶段通过 |
| 第二阶段：Agent 重构 | 3 天 | 第一阶段完成 |
| 第三阶段：MsgHub + 状态机 | 2 天 | 第二阶段完成 |
| 第四阶段：异步化 + 清理 | 2 天 | 第三阶段完成 |
| 第五阶段：测试与文档 | 2 天 | 第四阶段完成 |
| **总计** | **12 天** | |

---

## 参考资料

| 资源 | 链接 |
|------|------|
| AgentScope GitHub | https://github.com/agentscope-ai/agentscope |
| AgentScope 文档 | https://doc.agentscope.io/ |
| REME GitHub | https://github.com/agentscope-ai/ReMe |
| REME 中文 README | https://github.com/agentscope-ai/ReMe/blob/main/README_ZH.md |
| AgentScope Memory 模块 | https://github.com/agentscope-ai/agentscope/tree/main/src/agentscope/memory |
| REME Short-Term Memory 示例 | https://github.com/agentscope-ai/agentscope/tree/main/examples/functionality/short_term_memory/reme |
