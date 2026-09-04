"""FastAPI 服务层：把「代码库/PR 智能问答」包成 HTTP 接口。

运行：
    uvicorn app:app --reload

接口：
    GET  /health         健康检查（含索引状态）
    POST /ask            代码库问答（带 file:line 引证）
    POST /review_pr      PR 解析（走 GitHub MCP）
    GET  /trace/summary  汇总 trace 日志（可观测性）

每次请求通过 step_callbacks 把执行情况追加到 trace.jsonl（step/耗时/token/工具调用）。
"""
import os
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI
from mcp import StdioServerParameters
from pydantic import BaseModel
from smolagents import CodeAgent, MCPClient

from agents import make_search_tool
from code_index import CodeIndex
from config import build_model
from trace import TraceLogger, summarize

app = FastAPI(title="代码库/PR 智能问答服务", version="2.0.0")

_model = build_model()
_trace = TraceLogger("trace.jsonl")
INDEX_FILE = os.getenv("CODE_INDEX_FILE", "code_index.json")
_index = CodeIndex.load(INDEX_FILE) if Path(INDEX_FILE).exists() else None


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


class PRRequest(BaseModel):
    repo: str
    number: int


def _new_agent(tools):
    """每个请求建一个新 Agent，注入本请求的 run_id 用于 trace 隔离。"""
    return CodeAgent(
        tools=tools,
        model=_model,
        step_callbacks=_trace.step_callbacks(str(uuid.uuid4())),
    )


@app.get("/health")
def health():
    return {"status": "ok", "index_chunks": len(_index.chunks) if _index else 0}


@app.get("/trace/summary")
def trace_summary():
    return {"summary": summarize("trace.jsonl")}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if _index is None:
        return AskResponse(
            answer=f"未找到索引 {INDEX_FILE}，请先运行 `python index_repo.py <仓库路径>`"
        )
    agent = _new_agent([make_search_tool(_index)])
    answer = agent.run(req.question)
    return AskResponse(answer=str(answer))


@app.post("/review_pr")
def review_pr(req: PRRequest):
    server_params = StdioServerParameters(
        command=sys.executable, args=["github_mcp_server.py"]
    )
    with MCPClient(server_params, structured_output=False) as mcp_tools:
        agent = _new_agent(mcp_tools)
        task = (
            f"分析仓库 {req.repo} 的 PR #{req.number}，请分三点回答：\n"
            "1) 这个 PR 做了什么改动（结合 diff 总结）；\n"
            "2) 改了哪些文件；\n"
            "3) 可能引入的风险或需要关注的测试点。"
        )
        answer = agent.run(task)
    return {"answer": str(answer)}
