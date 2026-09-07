"""自研极简 Agent Harness：手写 ReAct 循环 + Tool Calling，不依赖 smolagents 框架。

背景：smolagents 的 CodeAgent 已经封装好了 Agent 运行时，本项目默认用它跑代码问答。
本模块提供一套自研的 MiniAgent，把「裸 LLM → 能调用工具的 Agent」的运行时循环手写出来，
用于对照理解框架到底做了什么。核心四件事：

1. ReAct 循环：推理 → 解析工具调用 → 执行 → 观察 → 再推理，直到产出最终答案
2. Tool Calling：OpenAI function-calling 格式；JSON 参数解析失败 / 工具执行异常都回填给模型而非崩溃
3. 上下文管理：messages 累积，工具结果以 role=tool 回填（带 tool_call_id 关联）
4. 可观测性与防死循环：记录每步 trace；重复调用检测强停，步数用尽则兜底强制作答

用法：
    python harness.py "这个仓库怎么处理超时重试的？" --index code_index.json
"""
import argparse
import json
from dataclasses import dataclass
from typing import Any, Callable

from code_index import CodeIndex
from config import build_openai_client, get_model_id


@dataclass
class Tool:
    """一个可被 Agent 调用的工具：名字 + 描述 + JSON Schema + 实现函数。"""
    name: str
    description: str
    parameters: dict
    fn: Callable[..., str]


class StepLimitError(RuntimeError):
    """超过最大步数或检测到死循环仍未产出最终答案。"""


SYSTEM_PROMPT = (
    "你是代码库问答助手。可以调用工具 search_codebase 检索代码，"
    "基于返回的 file:line 与代码内容回答问题。"
    "整个对话中最多调用 search_codebase 3 次，之后必须直接回答；"
    "即使检索结果不理想，也要基于已知信息给出答案，并引用代码位置（file:line）。"
)


class MiniAgent:
    """自研 ReAct Agent：手写「推理 → 工具 → 观察」循环，无框架依赖。"""

    def __init__(self, tools: list[Tool], max_steps: int = 8,
                 max_repeat: int = 3, temperature: float = 0.0):
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.max_repeat = max_repeat
        self.temperature = temperature
        self.client = build_openai_client()
        self.model_id = get_model_id()
        self.trace: list[dict] = []   # 每步的可观测记录

    # ---- Tool Calling ----
    def _tool_schemas(self) -> list[dict]:
        return [
            {"type": "function", "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }}
            for t in self.tools.values()
        ]

    def _call_llm(self, messages: list[dict]):
        return self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            tools=self._tool_schemas(),
            tool_choice="auto",
            temperature=self.temperature,
        )

    def _execute(self, tool_call) -> str:
        """执行一次工具调用；参数解析/执行失败时回填错误信息，不让循环崩溃。"""
        tool = self.tools.get(tool_call.function.name)
        if tool is None:
            return f"错误：未知工具 {tool_call.function.name}"
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError as e:
            return f"错误：工具参数不是合法 JSON：{e}"
        try:
            return str(tool.fn(**args))
        except Exception as e:  # 工具内部异常兜底
            return f"错误：工具执行失败：{type(e).__name__}: {e}"

    @staticmethod
    def _assistant_message(msg) -> dict:
        """把含 tool_calls 的 assistant 消息转成 OpenAI 消息格式回填。"""
        return {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        }

    def _force_answer(self, messages: list[dict]) -> str:
        """兜底：步数用尽后，追加明确指令强制模型直接作答（不再提供工具）。"""
        final_messages = messages + [{
            "role": "user",
            "content": (
                "请基于以上检索到的代码，直接用自然语言总结回答，"
                "引用相关 file:line，不要再调用工具、不要输出工具调用格式。"
            ),
        }]
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=final_messages,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content or ""

    # ---- 主循环 ----
    def run(self, task: str) -> str:
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        recent: list[str] = []  # 用于重复调用检测

        for step in range(1, self.max_steps + 1):
            msg = self._call_llm(messages).choices[0].message

            # 模型不调工具 → 视为最终答案
            if not msg.tool_calls:
                answer = msg.content or ""
                self.trace.append({"step": step, "action": "answer", "text": answer})
                return answer

            messages.append(self._assistant_message(msg))
            for tc in msg.tool_calls:
                obs = self._execute(tc)
                sig = f"{tc.function.name}({tc.function.arguments})"
                self.trace.append({"step": step, "action": "tool", "call": sig, "obs": obs[:200]})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": obs})

                # 死循环检测：同一调用连续出现 max_repeat 次则强停
                recent.append(sig)
                if recent.count(sig) >= self.max_repeat:
                    raise StepLimitError(
                        f"检测到重复调用 {sig} 达 {self.max_repeat} 次，疑似死循环，已强停"
                    )

        # 步数用尽：兜底强制作答（而非直接失败）
        answer = self._force_answer(messages)
        self.trace.append({"step": self.max_steps + 1, "action": "force_answer", "text": answer})
        return answer


def make_search_tool(index: CodeIndex) -> Tool:
    """把 CodeIndex 检索包装成 Agent 可调用的工具。"""
    def search_codebase(query: str) -> str:
        results = index.search(query, top_k=5)
        if not results:
            return "未检索到相关代码"
        return "\n\n".join(
            f"[{r['file']}:{r['start_line']}] {r['name']}\n{r['code']}" for r in results
        )

    return Tool(
        name="search_codebase",
        description="在代码库中检索与查询最相关的代码块，返回 file:line 及代码内容。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索关键词或问题"},
            },
            "required": ["query"],
        },
        fn=search_codebase,
    )


def main() -> None:
    try:  # Windows 控制台 GBK 兜底
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="自研 MiniAgent 代码库问答（对照框架）")
    ap.add_argument("question", help="关于代码库的问题")
    ap.add_argument("--index", default="code_index.json", help="索引文件")
    ap.add_argument("--max-steps", type=int, default=8, help="最大步数")
    args = ap.parse_args()

    index = CodeIndex.load(args.index)
    print(f"[索引] chunk 数：{len(index.chunks)}")
    agent = MiniAgent(tools=[make_search_tool(index)], max_steps=args.max_steps)

    print(f"\n===== 问题 =====")
    print(args.question)
    print(f"\n===== 运行 trace =====")
    try:
        answer = agent.run(args.question)
    except StepLimitError as e:
        print(f"[强停] {e}")
        answer = "(未产出答案)"

    for t in agent.trace:
        if t["action"] == "tool":
            print(f"  Step{t['step']} [tool] {t['call']}")
            print(f"         -> {t['obs'][:120]}")
        elif t["action"] == "force_answer":
            print(f"  Step{t['step']} [强制作答]")
        else:
            print(f"  Step{t['step']} [最终答案]")

    print(f"\n===== 回答 =====")
    print(answer)


if __name__ == "__main__":
    main()
