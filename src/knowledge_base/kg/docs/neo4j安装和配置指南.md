# Neo4j 数据库安装和配置指南

## 1. Neo4j 安装

### macOS (使用 Homebrew)
```bash
# 安装 Neo4j
brew install neo4j

# 启动 Neo4j 服务
neo4j start

# 访问 Neo4j Browser
open http://localhost:7474
```

### Docker (推荐)
```bash
# 拉取 Neo4j 镜像
docker pull neo4j:5.0

# 运行 Neo4j 容器
docker run \\
    --name neo4j \\
    -p7474:7474 -p7687:7687 \\
    -e NEO4J_AUTH=neo4j/password \\
    -e NEO4J_PLUGINS='["apoc"]' \\
    -v $PWD/neo4j/data:/data \\
    -v $PWD/neo4j/logs:/logs \\
    neo4j:5.0
```

### 直接下载
1. 访问 https://neo4j.com/download/
2. 下载适合您操作系统的版本
3. 按照安装向导进行安装

## 2. 环境变量配置

创建 `.env` 文件并添加以下内容：

```bash
# Neo4j 配置
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=medical_kg

# 其他可选配置
PYTHONPATH=/path/to/Chinese-MedQA-Qwen2/src
```

## 3. 验证安装

```bash
# 测试 Neo4j 连接
python -c "
from langchain_community.graphs.neo4j_graph import Neo4jGraph
graph = Neo4jGraph(url='bolt://localhost:7687', username='neo4j', password='password')
print('Neo4j 连接成功!')
"
```

## 4. 运行完整测试

```bash
# 运行核心功能测试
python tests/test_kg_core.py

# 运行端到端测试（需要 Neo4j）
python tests/test_kg_end_to_end.py
```

## 5. 使用知识图谱构建器

```python
from src.config.kg_config import KGConfig
from src.knowledge_base.kg.kg_builder import KGBuilder

# 初始化配置
config = KGConfig()

# 创建知识图谱构建器
kg_builder = KGBuilder(config)

# 从 PDF 目录构建知识图谱
kg_builder.build_kg_from_pdfs("path/to/your/pdf/directory")

# 获取知识图谱摘要
summary = kg_builder.get_kg_summary()
print(summary)

# 关闭连接
kg_builder.close_connection()
```

## 6. 常见问题

### 连接失败
- 检查 Neo4j 服务是否正在运行
- 确认端口 7687 是否可用
- 验证用户名和密码

### 权限问题
- 确保 Neo4j 用户有足够的权限
- 检查数据库访问权限

### 性能优化
- 调整 Neo4j 内存设置
- 创建适当的索引
- 使用批量操作

## 7. 生产环境部署

### 安全配置
1. 修改默认密码
2. 配置 SSL 连接
3. 设置防火墙规则
4. 启用身份验证

### 性能调优
1. 调整 JVM 堆内存
2. 配置页面缓存
3. 优化查询性能
4. 监控资源使用

### 备份策略
1. 定期备份数据库
2. 测试恢复流程
3. 设置监控告警