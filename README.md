# smolagents 上手项目

基于 HuggingFace [smolagents](https://github.com/huggingface/smolagents) 的可运行项目，
覆盖一条完整的 AI Agent 应用工程主线：**读源码 → 扩工具 → 持久化记忆 → 向量检索 → 推理增强 → MCP 工具协议 → 服务化**。

## 目录结构

```
smolagents/
├── .venv/            # 虚拟环境（已创建）
├── config.py         # 模型配置（DeepSeek/Qwen 自动选择 + OpenAI 客户端）
├── demo.py           # 演示 1：CodeAgent + 自定义工具，多步骤任务
├── demo_memory.py    # 演示 2：持久化记忆 + 向量检索，跨会话 recall
├── demo_retrieval.py # 演示 3：混合检索（BM25+向量）+ LLM Rerank + Recall@k 评估
├── demo_semantic.py  # 演示：稠密语义向量 vs 词面检索（饮料→咖啡）
├── demo_reasoning.py # 演示 4：自一致性（多路采样投票）+ 自我反思（批判-改进）
├── demo_mcp.py       # 演示 5：MCPClient 连接自建 MCP server
├── custom_tools.py   # 自定义工具（calculator / get_current_time / count_words）
├── memory.py         # 持久化记忆模块（PersistentMemory + build_memory_tools）
├── embedding.py      # 向量化：TF-IDF（稀疏）+ DashScope text-embedding（稠密语义）
├── retrieval.py      # BM25 / 混合检索(RRF) / LLM-as-reranker / Recall@k 评估
├── reasoning.py      # Self-Consistency（多路采样投票）+ Reflexion（反思循环）
├── mcp_server.py     # 自建 MCP server（FastMCP，stdio）
├── app.py            # FastAPI 服务层（把 Agent 包成 HTTP 接口）
├── requirements.txt  # 依赖（注意 mcp 需 pin 到 <2）
├── .env.example      # API Key 配置模板
└── README.md
```

## 快速开始

```bash
source .venv/Scripts/activate        # 激活虚拟环境
cp .env.example .env                 # 填入 DEEPSEEK_API_KEY（可选 QWEN_API_KEY）
pip install -r requirements.txt      # 首次需安装依赖

python demo.py          # 演示 1：基础工具调用
python demo_memory.py   # 演示 2：跨会话持久化记忆
python demo_retrieval.py# 演示 3：混合检索 + Rerank + 评估
python demo_reasoning.py# 演示 4：自一致性 + 反思
python demo_mcp.py      # 演示 5：MCP 工具协议
uvicorn app:app --reload# 服务化：HTTP 接口
```

> ⚠️ `mcp` 已 pin 到 `<2`：smolagents 依赖的 `mcpadapt 0.1.20` 只支持 mcp 1.x。

## 核心概念

- **CodeAgent**：让 LLM 直接生成 Python 代码来调用工具（而非 JSON tool-call），执行循环可读可审计。
- **ReAct 循环**：生成代码 → 执行 → 观察结果写回历史 → 直到 `final_answer` 或超 `max_steps`。
- **`@tool` vs MCP**：`@tool` 注册的工具只能在 smolagents 内部用；MCP 是通用协议，任何客户端都能调用。

## 检索（retrieval.py / embedding.py）

- **BM25**：Okapi BM25 稀疏检索，纯 numpy 实现（IDF + 词频饱和 + 长度归一化）。
- **混合检索**：BM25 + TF-IDF 向量 + 可选稠密向量，用 RRF（Reciprocal Rank Fusion）融合排名。
- **稠密语义向量**：`DashScopeEmbedder`（通义 text-embedding），弥补词面检索无法匹配同义/近义的不足。
- **重排序**：`LLMReranker`（LLM-as-reranker）对 top-k 候选按相关性重排。
- **评估**：`evaluate()` 输出 BM25 / 向量 / 混合 的 Recall@1、@3、@5。

## 推理增强（reasoning.py）

- **Self-Consistency**：同一任务多路采样（temperature>0），多数投票 / LLM 聚合，降低单次采样的随机误差。
- **Reflexion**：生成 → 自我批判 → 将批判反馈回下一轮，循环改进。

## 后续可做（写进简历的微调方向）

1. 稠密向量换更强模型（BGE / OpenAI embedding），或接 FAISS/Milvus 做大规模索引。
2. 用 `ManagedAgent` 做「规划者 + 执行者」多智能体。
3. 加检索评估的完整指标（NDCG、MRR），用更大的评测集。
