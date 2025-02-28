Chatbot 是一个基于FastAPI和Uvicorn的API服务实现，支持流式响应和历史管理。

主要功能特点：

1. **服务端架构**：
- 使用FastAPI构建REST API端点
- 支持多会话隔离（通过session_id）
- 流式响应（text/event-stream）
- 自动内存管理（torch_gc）
- 会话生命周期管理

1. **核心端点**：
- `POST /chat`：流式聊天接口
- `POST /clear`：重置会话历史
- `POST /create_session`：创建新会话

1. **客户端功能**：
- 模拟原始命令行交互体验
- 支持会话管理
- 实时流式输出
- 保留clear/exit命令

使用方式：

1. 启动服务端：
```bash
python server.py
```

2. 运行测试客户端：
```bash
python client.py
```

3. 也可以通过curl测试：
```bash
# 创建会话
SESSION_ID=$(curl -X POST http://localhost:8000/create_session -s | jq -r .session_id)

# 发送请求
curl -X POST -H "Content-Type: application/json" \
-d "{\"session_id\":\"$SESSION_ID\",\"query\":\"什么是脉诊？\"}" \
http://localhost:8000/chat --no-buffer
```

该实现保留了原始代码的以下特性：
- 初始中医角色设定
- 流式文本生成
- 历史记录管理
- 内存清理机制
- 交互式对话体验

同时增加了：
- HTTP API访问能力
- 多会话支持
- 错误处理机制
- 可扩展的会话管理
