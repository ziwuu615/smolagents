"""端到端入口：代码库问答 + PR 解析。

把「代码检索 + GitHub MCP 工具」收拢成两条可用的命令，回答可追溯到具体 file:line。

用法：
    python cli.py ask "这个仓库怎么处理超时重试的？" --repo <仓库路径>
    python cli.py ask_pr "psf/requests" 6701
"""
import argparse
import sys
from pathlib import Path

from mcp import StdioServerParameters
from smolagents import CodeAgent, MCPClient, tool

from code_index import CodeIndex
from config import build_model


def load_or_build_index(repo: str | None, index_file: str) -> CodeIndex:
    """已有索引则复用，否则从仓库构建并落盘。"""
    p = Path(index_file)
    if p.exists():
        print(f"[索引] 复用已有索引：{p}")
        return CodeIndex.load(p)
    if not repo:
        raise SystemExit(
            f"未找到索引文件 {p}。请先运行 `python index_repo.py <仓库路径>`，"
            f"或在本命令里带上 --repo <仓库路径> 让它自动构建。"
        )
    print(f"[索引] 首次索引仓库：{repo} ...")
    index = CodeIndex.from_repo(repo)
    index.save(p)
    print(f"[索引] 已索引 {len(index.chunks)} 个代码块 → {p}")
    return index


def make_search_tool(index: CodeIndex):
    @tool
    def search_codebase(query: str) -> str:
        """在代码库中检索与查询最相关的代码块，返回 file:line 及代码内容。

        Args:
            query: 要检索的代码问题或关键词，例如 "超时重试怎么实现"。
        """
        results = index.search(query, top_k=5)
        if not results:
            return "未检索到相关代码"
        return "\n\n".join(
            f"[{r['file']}:{r['start_line']}] {r['name']}\n{r['code']}" for r in results
        )

    return search_codebase


def cmd_ask(args) -> None:
    index = load_or_build_index(args.repo, args.index)
    agent = CodeAgent(tools=[make_search_tool(index)], model=build_model())
    answer = agent.run(args.question)
    print("\n===== 回答 =====")
    print(answer)


def cmd_ask_pr(args) -> None:
    server_params = StdioServerParameters(
        command=sys.executable, args=["github_mcp_server.py"]
    )
    with MCPClient(server_params, structured_output=False) as mcp_tools:
        agent = CodeAgent(tools=mcp_tools, model=build_model())
        task = (
            f"分析仓库 {args.repo} 的 PR #{args.number}，请分三点回答：\n"
            "1) 这个 PR 做了什么改动（结合 diff 总结）；\n"
            "2) 改了哪些文件；\n"
            "3) 可能引入的风险或需要关注的测试点。"
        )
        answer = agent.run(task)
    print("\n===== 回答 =====")
    print(answer)


def cmd_ask_agent(args) -> None:
    from agents import build_code_qa_system, run_with_critic

    index = load_or_build_index(args.repo, args.index)
    model = build_model()
    router, _, reviewer = build_code_qa_system(index, model)
    answer, rounds, verdict = run_with_critic(router, reviewer, args.question, max_rounds=2)
    print(f"\n===== 回答（经 {rounds} 轮盲审）=====")
    print(answer)
    if rounds > 1:
        print(f"\n[盲审意见] {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description="代码库/PR 智能问答")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ask = sub.add_parser("ask", help="代码库问答（带 file:line 引证）")
    p_ask.add_argument("question", help="关于代码库的问题")
    p_ask.add_argument("--repo", help="仓库路径（首次索引时必需）")
    p_ask.add_argument("--index", default="code_index.json", help="索引文件路径")
    p_ask.set_defaults(func=cmd_ask)

    p_pr = sub.add_parser("ask_pr", help="PR 解析（走 GitHub MCP）")
    p_pr.add_argument("repo", help="仓库全名，如 psf/requests")
    p_pr.add_argument("number", type=int, help="PR 编号")
    p_pr.set_defaults(func=cmd_ask_pr)

    p_agent = sub.add_parser("ask_agent", help="多智能体问答（Master-Worker + 盲审返工）")
    p_agent.add_argument("question", help="关于代码库的问题")
    p_agent.add_argument("--repo", help="仓库路径（首次索引时必需）")
    p_agent.add_argument("--index", default="code_index.json", help="索引文件路径")
    p_agent.set_defaults(func=cmd_ask_agent)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
