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




## lxml, XPath 的使用

 `lxml` 库的作用是解析 HTML 内容，并通过 XPath 提取 HTML 中的特定文本。

---

### 代码：
```python
# 使用lxml解析HTML
html = etree.HTML(content)
# 提取答案
answers = html.xpath("/html/body/div[2]/div/div/b-superframe-body/div/div[2]/div/div/article/section/section/div/div/a/div[2]/text()")[:3]
```

---

### 逐行解释：

#### 1. **`from lxml import etree`**
这行代码没有在你提供的代码中出现，但它是使用 `etree` 的前提，因此我们需要知道它的作用。

- `lxml` 是一个用于处理 XML 和 HTML 文档的强大库，支持高效的解析和提取操作。
- `etree` 是 `lxml` 中的一个模块，专门用于解析和操作树状结构的文档，比如 XML 和 HTML。
- 我们需要先从 `lxml` 中导入 `etree`，才能使用后续的功能。

#### 2. **`html = etree.HTML(content)`**

这行代码的作用是将 HTML 文本内容解析成一个可以操作的树形结构。

- **`etree.HTML(content)`**：
  - `etree.HTML()` 是 `lxml` 提供的一个方法，用于解析 HTML 文本字符串。
  - 它会将 HTML 文本转换为一个树形结构（DOM 树），方便我们通过 XPath 或其他方法提取其中的内容。
  - 参数 `content` 是一个字符串，通常是从网页请求中获取的 HTML 文本。

  例如，假设 `content` 是以下 HTML 文本：
  ```html
  <html>
      <body>
          <div>
              <p>Hello, world!</p>
          </div>
      </body>
  </html>
  ```
  调用 `etree.HTML(content)` 后，它会将 HTML 解析成一个树形结构，供后续操作。

  **简单比喻**：可以把 HTML 文本看成一棵树，`etree.HTML()` 就是把这棵树种出来，方便我们通过“路径”（XPath）找到树上的“果实”（内容）。

---

#### 3. **`answers = html.xpath(...)`**

这行代码的作用是通过 XPath 提取 HTML 中的特定内容。

- **`html.xpath()`**：
  - `xpath()` 是 `lxml` 提供的方法，用来通过 XPath 表达式提取 HTML 文档中的内容。
  - XPath 是一种用于在 XML 或 HTML 文档中定位节点的语言，类似于文件路径。
  - 返回值通常是一个列表，包含所有匹配的节点内容。

- **`"/html/body/div[2]/div/div/b-superframe-body/div/div[2]/div/div/article/section/section/div/div/a/div[2]/text()"`**：
  - 这是一个 XPath 表达式，用于指定我们想要提取的节点路径。
  - 它的含义是：
    1. `/html`：从 HTML 文档的根节点开始。
    2. `/body`：进入 `<body>` 标签。
    3. `/div[2]`：选择 `<body>` 下的第二个 `<div>`。
    4. `/div/div/b-superframe-body/...`：继续按照路径逐层深入，直到目标节点。
    5. `/div[2]/text()`：选择目标节点下的第二个 `<div>` 的文本内容。

  **注意**：这个 XPath 表达式非常具体，通常是根据 HTML 文档的结构手动编写的。如果 HTML 结构发生变化，XPath 可能需要调整。

- **`[:3]`**：
  - 这是 Python 的切片操作，表示只取前 3 个匹配结果。
  - 假设 `html.xpath(...)` 返回一个列表，比如 `['答案1', '答案2', '答案3', '答案4']`，那么 `[:3]` 会截取前 3 个元素：`['答案1', '答案2', '答案3']`。

---

#### 4. **总结代码逻辑**

完整流程如下：
1. 使用 `etree.HTML(content)` 将 HTML 文本解析成树形结构。
2. 使用 `html.xpath(...)` 提取指定路径下的内容。
3. 使用 `[:3]` 截取前 3 个匹配结果。

---

#### 5. **补充知识：XPath 的常用语法**
为了更好地理解 XPath，这里列出一些常用的语法：
- `/`：从根节点开始选择。
- `//`：从文档中的任意位置选择匹配的节点。
- `node[n]`：选择某个节点的第 n 个子节点（从 1 开始计数）。
- `@attr`：选择某个节点的属性值。
- `text()`：选择某个节点的文本内容。

---

#### 6. **假设的应用场景**
这段代码可能用于从网页中提取某些特定的答案或信息，比如爬取网页上的问题答案。实际应用中，`content` 通常是通过网络请求（比如 `requests.get(url).text`）获取的 HTML 内容。

---

如果还有其他问题，欢迎随时提问！ 😊