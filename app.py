"""FastAPI 服务层：把 CodeAgent 包成 HTTP 接口。

运行：
    uvicorn app:app --reload
调用：
    curl -X POST http://127.0.0.1:8000/agent \
         -H "Content-Type: application/json" \
         -d '{"task": "现在是几点？"}'
"""
from fastapi import FastAPI
from pydantic import BaseModel

from smolagents import CodeAgent

from config import build_model
from custom_tools import calculator, count_words, get_current_time
from memory import PersistentMemory, build_memory_tools

app = FastAPI(title="AI Agent 服务", version="1.0.0")

# 服务启动时初始化：模型 + 自定义工具 + 持久化记忆（记忆跨请求/跨重启保留）
_model = build_model()
_memory = PersistentMemory("memory_store.json")
_tools = [calculator, get_current_time, count_words] + build_memory_tools(_memory)
_agent = CodeAgent(tools=_tools, model=_model)


class AgentRequest(BaseModel):
    task: str


class AgentResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/agent", response_model=AgentResponse)
def run_agent(req: AgentRequest):
    result = _agent.run(req.task)
    return AgentResponse(answer=str(result))
