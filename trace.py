"""结构化 Trace：用 smolagents 的 step_callbacks 钩子把每次运行落成 JSONL 日志。

不侵入 Agent 内部：通过 step_callbacks 在每个 ActionStep 结束时记录
step、工具调用、耗时、token，供评测统计与「可观测性」展示——每一步的代价都留痕。

用法：
    trace = TraceLogger("trace.jsonl")
    agent = CodeAgent(..., step_callbacks=trace.step_callbacks(run_id="xxx"))
    agent.run(task)
    print(summarize("trace.jsonl"))   # 汇总总步数/耗时/token
"""
import json
from pathlib import Path

from smolagents.memory import ActionStep


class TraceLogger:
    """把 Agent 每一步的执行情况追加写入 JSONL 文件。"""

    def __init__(self, path: str = "trace.jsonl"):
        self.path = Path(path)
        self._run_id = None

    def _on_step(self, step: ActionStep) -> None:
        if self._run_id is None:
            return
        tokens = step.token_usage
        record = {
            "run_id": self._run_id,
            "step": step.step_number,
            "duration_s": round(step.timing.duration, 3) if step.timing.duration else None,
            "input_tokens": tokens.input_tokens if tokens else None,
            "output_tokens": tokens.output_tokens if tokens else None,
            "tool_calls": [tc.name for tc in (step.tool_calls or [])],
            "is_final_answer": step.is_final_answer,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def step_callbacks(self, run_id: str):
        """返回可传给 CodeAgent(step_callbacks=...) 的回调注册表。"""
        self._run_id = run_id
        return {ActionStep: self._on_step}


def summarize(path: str = "trace.jsonl") -> str:
    """汇总 JSONL 日志：运行次数、总步数、总耗时、总 token。"""
    p = Path(path)
    if not p.exists():
        return "（无 trace 日志）"
    steps = [
        json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    runs = len({s["run_id"] for s in steps})
    total_in = sum(s["input_tokens"] or 0 for s in steps)
    total_out = sum(s["output_tokens"] or 0 for s in steps)
    total_dur = sum(s["duration_s"] or 0 for s in steps)
    return (
        f"runs={runs}, steps={len(steps)}, "
        f"duration={total_dur:.1f}s, "
        f"tokens_in={total_in}, tokens_out={total_out}"
    )
