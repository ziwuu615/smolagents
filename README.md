# smolagents 上手项目 → 代码库/PR 智能问答

基于 HuggingFace [smolagents](https://github.com/huggingface/smolagents) 的可运行项目。

主线一：走通一条完整的 AI Agent 应用工程链路 —— **读源码 → 扩工具 → 持久化记忆 → 向量检索 → 推理增强 → MCP 工具协议 → 服务化**。

主线二：在主线一的能力上，收敛出一个 **「代码库/PR 智能问答」应用** —— AST 级代码切分 + 混合检索 + 真实 GitHub 数据 + 自建评测集，回答可追溯到 `file:line`。

## 目录结构

```
smolagents/
├── .venv/               # 虚拟环境
├── config.py            # 模型配置（DeepSeek/Qwen 自动选择）
├── custom_tools.py      # 自定义工具（calculator / get_current_time / count_words）
├── memory.py            # 持久化记忆（PersistentMemory + build_memory_tools）
├── embedding.py         # TF-IDF 稀疏 + 通义稠密向量（Embedder 接口）
├── retrieval.py         # BM25 / 混合检索(RRF) / LLM重排 / Recall@k + code_tokenize
├── reasoning.py         # Self-Consistency + Reflexion（反思闭环）
├── mcp_server.py        # 自建 MCP server（FastMCP，示例工具）
├── code_index.py        # 【应用】AST 切块 + 代码索引（带 file:line）
├── index_repo.py        # 【应用】CLI：仓库 → code_index.json
├── github_mcp_server.py # 【应用】MCP server：GitHub Issue/PR 只读查询
├── cli.py               # 【应用】端到端入口：ask / ask_pr / ask_agent / ask_harness
├── agents.py            # 【应用】多智能体：Master-Worker + Critic 盲审
├── harness.py           # 自研 MiniAgent：ReAct 循环 + Tool Calling（对照框架）
├── eval_benchmark.py    # 【应用】评测：Recall@k + 端到端准确率
├── benchmark/           # 【应用】评测标注集（code_qa.json + 说明）
├── trace.py             # 【应用】结构化 Trace（step_callbacks → JSONL 可观测）
├── app.py               # FastAPI：/ask /review_pr /trace/summary
├── Dockerfile           # 容器化
├── demo_*.py            # 各能力演示脚本
├── requirements.txt     # 依赖（mcp 需 pin 到 <2）
└── .env.example         # Key 配置模板
```

## 快速开始

```bash
source .venv/Scripts/activate        # 激活虚拟环境
cp .env.example .env                 # 填入 DEEPSEEK_API_KEY（可选 QWEN_API_KEY / GITHUB_TOKEN）
pip install -r requirements.txt      # 首次需安装依赖

# —— 能力演示 ——
python demo.py           # 工具调用
python demo_memory.py    # 跨会话持久化记忆
python demo_retrieval.py # 混合检索 + Rerank + 评估
python demo_reasoning.py # 自一致性 + 反思
python demo_mcp.py       # MCP 工具协议

# —— 代码库/PR 智能问答 ——
python index_repo.py <仓库路径>           # 索引仓库 → code_index.json
python cli.py ask "这个仓库怎么处理限流的？" --repo <仓库路径> [--dense]
python cli.py ask_agent "问题" --index code_index.json [--dense]  # 多智能体问答
python cli.py ask_pr "psf/requests" 6701  # PR 解析（走 GitHub MCP）
python eval_benchmark.py                  # 跑检索 Recall@k
python eval_benchmark.py --dense          # 对比稀疏 vs 稠密混合检索（需 QWEN_API_KEY）
python eval_benchmark.py --dense --rerank # 完整管线：稀疏 → 混合 → 重排

uvicorn app:app --reload                  # 服务化：HTTP 接口
```

> ⚠️ `mcp` 已 pin 到 `<2`：smolagents 依赖的 `mcpadapt 0.1.20` 只支持 mcp 1.x。
> ⚠️ `ask_pr` 走 GitHub API，需在 `.env` 里配 `GITHUB_PROXY`（本机代理），可选 `GITHUB_TOKEN`。

## 代码库/PR 智能问答（应用层）

```
用户提问
   │
   ▼
router（Master，意图路由 + 综合）
   ├─→ code_search（worker）：AST 切块后的混合检索，返回 file:line + 代码
   └─→ GitHub MCP 工具      ：search_issues / get_pull_request / get_pr_diff / get_pr_files
          │
          ▼
reviewer（Critic 盲审）→ 返工（最多 2 轮）
          │
          ▼
   带 file:line 引证的回答
```

- **代码索引**（`code_index.py`）：标准库 `ast` 按「函数/类/方法」粒度切块（自带行号）；
  检索前用 `code_tokenize` 把标识符拆成子词（`getUserById` → `get/user/by/id`），
  复用 `retrieval.HybridRetriever`（BM25 + TF-IDF 向量 + RRF 融合），
  `build_dense()` 接通义稠密向量（批量 embed + `.npy` 缓存，重跑不重复调 API）。
- **真实数据源**（`github_mcp_server.py`）：MCP 暴露 4 个只读工具，走 GitHub REST API，
  最小特权白名单、超长 diff 自动截断。
- **评测**（`eval_benchmark.py`）：自建 `问题 → file:line` 标注集，量化 `Recall@1/@3/@5`
  与端到端答案准确率；`--suggest` 抽样打印 chunk 辅助标注。

## 检索（retrieval.py / embedding.py）

- **BM25**：Okapi BM25 稀疏检索，纯 numpy 实现。
- **混合检索**：BM25 + TF-IDF 向量 + **稠密语义向量**，用 RRF 融合排名
  （稀疏管精确词匹配，稠密管跨语言语义，二者取长补短）。
- **稠密向量**（`embedding.DashScopeEmbedder`）：通义 text-embedding-v3（走 QWEN_API_KEY），
  `embed_many` 批量 + L2 归一化；`CodeIndex.build_dense()` 建向量并缓存到 `code_index_dense.npy`。
- **code_tokenize**：代码感知分词，拆 camelCase/snake_case 标识符。
- **重排序**：`LLMReranker` 对 top-k 候选按相关性重排。
- **评估**：`evaluate()` 输出各方案 Recall@1、@3、@5。

**实测（requests 2.34.2，30 条 `问题→file:line` 标注）：**

| 方案 | Recall@1 | Recall@3 | Recall@5 |
|------|---------|---------|---------|
| 稀疏检索（BM25+TF-IDF） | 33.3% | 73.3% | 83.3% |
| 混合（+稠密向量） | 46.7% | 80.0% | 86.7% |
| 混合（+LLM 重排） | **66.7%** | **93.3%** | **96.7%** |

> 稠密向量解决「中文问英文代码」的词面检索失效：BM25 对「超时」↔ `timeout` 无字面交集给 0 分，
> 稠密向量靠语义对齐把相关代码块捞回来，Recall@1 翻倍。

## 推理增强（reasoning.py）

- **Self-Consistency**：多路采样 + 多数投票，降低单次采样随机误差。
- **Reflexion**：生成 → 自我批判 → 反馈回下一轮，循环改进。

## 后续可做（Roadmap）

1. 稠密向量已接入（通义 text-embedding-v3，解决跨语言检索）；下一步换本地 BGE-M3 免 API 依赖、接 FAISS 面向更大代码库。
2. 用 OpenTelemetry 替代自建 JSONL trace，接入标准可观测平台；加 Redis 缓存。
3. 把多智能体（router + worker）也接入 FastAPI，而非当前的单 Agent 端点。
