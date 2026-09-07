"""评测：量化代码检索与端到端问答效果。

两个指标：
1. 检索 Recall@1/@3/@5（无 LLM，确定性）：对每条 (query → file:line) 标注，
   检查 top-k 检索结果是否命中目标函数/类的 def 行号。
2. 端到端答案准确率（--e2e，需 LLM）：跑 Agent 后由 LLM-as-judge 判「回答是否正确」。

用法：
    python eval_benchmark.py                 # 用 benchmark/code_qa.json 跑 Recall@k
    python eval_benchmark.py --dense         # 对比稀疏 vs 稠密混合检索（需 QWEN_API_KEY）
    python eval_benchmark.py --dense --rerank# 完整管线：稀疏 → 混合 → 混合+重排
    python eval_benchmark.py --e2e           # 追加端到端答案准确率（较慢、调 API）
    python eval_benchmark.py --suggest       # 从索引抽样打印 chunk，辅助人工标注

标注格式（benchmark/code_qa.json）：[{"query": "...", "file": "xx.py", "line": 410}]
  其中 line 为目标函数/类的 def 起始行号，命中 = 检索结果存在「同文件 + 同起始行」的 chunk。
"""
import argparse
import json
import random
from pathlib import Path

from code_index import CodeIndex
from config import build_model, build_openai_client, get_model_id
from smolagents import CodeAgent, tool


def load_benchmark(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"未找到评测集 {p}。请先按 benchmark/README.md 的说明填写标注。")
    return json.loads(p.read_text(encoding="utf-8"))


def _hit(results: list[dict], gt: dict) -> bool:
    return any(
        r["file"] == gt["file"] and r["start_line"] == gt["line"] for r in results
    )


def recall_at_k(search_fn, bench: list[dict], k: int) -> float:
    hits = sum(1 for item in bench if _hit(search_fn(item["query"], top_k=k), item))
    return hits / len(bench)


def _recall_row(label: str, search_fn, bench: list[dict], ks: list[int]) -> str:
    cells = [f"{recall_at_k(search_fn, bench, k):.3f}" for k in ks]
    return f"{label:<16}" + "".join(f"{c:>12}" for c in cells)


def run_recall_eval(index: CodeIndex, bench: list[dict], rerank: bool = False,
                    dense: bool = False) -> str:
    ks = [1, 3, 5]
    header = f"{'指标':<16}" + "".join(f"{'Recall@' + str(k):>12}" for k in ks)
    lines = [header, _recall_row("稀疏检索", index.search, bench, ks)]
    if dense:
        from embedding import DashScopeEmbedder

        index.build_dense(DashScopeEmbedder())
        lines.append(_recall_row("混合(+稠密)", index.search, bench, ks))
    if rerank:
        label = "混合(+重排)" if dense else "稀疏(+重排)"
        lines.append(
            _recall_row(
                label,
                lambda q, top_k: index.search_reranked(q, top_k=top_k),
                bench,
                ks,
            )
        )
    return "\n".join(lines)


def make_search_tool(index: CodeIndex):
    @tool
    def search_codebase(query: str) -> str:
        """在代码库中检索与查询最相关的代码块，返回 file:line 及代码内容。

        Args:
            query: 要检索的代码问题或关键词。
        """
        results = index.search(query, top_k=5)
        if not results:
            return "未检索到相关代码"
        return "\n\n".join(
            f"[{r['file']}:{r['start_line']}] {r['name']}\n{r['code']}" for r in results
        )

    return search_codebase


def run_e2e(index: CodeIndex, bench: list[dict]) -> str:
    """跑 Agent + LLM-judge，统计端到端答案准确率（慢，需 API）。"""
    agent = CodeAgent(tools=[make_search_tool(index)], model=build_model())
    client = build_openai_client()
    model_id = get_model_id()

    correct = 0
    for item in bench:
        answer = str(agent.run(item["query"]))
        judge = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": (
                f"问题：{item['query']}\n"
                f"参考答案位置：{item['file']}:{item['line']}\n"
                f"模型回答：{answer}\n"
                "请判断该回答是否正确（说对了要点/引用了正确的代码位置）。只回复 YES 或 NO。"
            )}],
            temperature=0,
        )
        if judge.choices[0].message.content.strip().upper().startswith("YES"):
            correct += 1
    return f"端到端答案准确率：{correct}/{len(bench)} = {correct / len(bench):.3f}"


def suggest(index: CodeIndex, n: int = 20, seed: int = 0) -> None:
    """抽样打印 chunk（file:line + 签名 + 代码），辅助快速写标注。"""
    rng = random.Random(seed)
    sample = rng.sample(index.chunks, min(n, len(index.chunks)))
    for c in sample:
        code_preview = c["code"].splitlines()[0] if c["code"] else ""
        print(f"{c['file']}:{c['start_line']}  {c['name']}  |  {code_preview[:60]}")


def main() -> None:
    try:  # Windows 控制台 GBK 会崩中文/重音，强制 UTF-8
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="代码库检索/问答评测")
    ap.add_argument("--index", default="code_index.json", help="索引文件")
    ap.add_argument("--benchmark", default="benchmark/code_qa.json", help="标注文件")
    ap.add_argument("--e2e", action="store_true", help="追加端到端答案准确率（调 API）")
    ap.add_argument("--rerank", action="store_true", help="对比 LLM 重排后的 Recall（调 API）")
    ap.add_argument("--dense", action="store_true", help="加稠密语义向量（需 QWEN_API_KEY），对比混合检索 Recall")
    ap.add_argument("--suggest", action="store_true", help="抽样打印 chunk 辅助标注")
    args = ap.parse_args()

    index = CodeIndex.load(args.index)
    print(f"索引 chunk 数：{len(index.chunks)}")

    if args.suggest:
        suggest(index)
        return

    bench = load_benchmark(args.benchmark)
    print(f"评测样本数：{len(bench)}")
    print("=" * 50)
    print(run_recall_eval(index, bench, rerank=args.rerank, dense=args.dense))
    if args.e2e:
        print(run_e2e(index, bench))


if __name__ == "__main__":
    main()
