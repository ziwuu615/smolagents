"""自建 GitHub MCP server：暴露只读工具，供 Agent 实时查询仓库 Issue/PR 数据。

与 mcp_server.py 里的玩具工具（天气/单位换算）不同，这里的工具走 GitHub REST API，
是真实、可用的外部数据源 —— 对标「通过 MCP 协议接入实时外部接口」。

最小特权设计：
- 只暴露白名单内的只读工具，无任何写操作；
- 未传 token 用匿名额度（60 次/小时，够 demo）；传 GITHUB_TOKEN 升到 5000 次/小时；
- 超长 diff 自动截断，避免把整个 PR 灌进上下文（对应「上下文压缩」）。

运行：python github_mcp_server.py（stdio 模式），或由 cli.py 通过 MCPClient 作为子进程拉起。
"""
import os

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("GitHub 仓库只读查询")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_PROXY = os.getenv("GITHUB_PROXY") or os.getenv("HTTPS_PROXY")
API = "https://api.github.com"
_TIMEOUT = 15
_MAX_DIFF_CHARS = 8000  # 截断超长 diff


def _headers(accept: str | None = None) -> dict:
    h = {"Accept": accept or "application/vnd.github+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def _get(path: str, params: dict | None = None, accept: str | None = None):
    """GET GitHub API，返回 (数据, 错误信息)。错误时数据为 None。"""
    proxies = {"http": GITHUB_PROXY, "https": GITHUB_PROXY} if GITHUB_PROXY else None
    try:
        r = requests.get(
            API + path, headers=_headers(accept), params=params,
            proxies=proxies, timeout=_TIMEOUT,
        )
    except requests.RequestException as e:
        return None, f"GitHub API 请求失败：{e}"
    if r.status_code != 200:
        return None, f"GitHub API 返回 {r.status_code}：{r.text[:200]}"
    if accept == "application/vnd.github.v3.diff":
        return r.text, None
    return r.json(), None


@mcp.tool()
def search_issues(repo: str, query: str) -> str:
    """搜索某个 GitHub 仓库的 issue 和 PR（按关键字过滤）。

    Args:
        repo: 仓库全名，例如 "psf/requests"。
        query: 搜索关键字，例如 "timeout"。
    """
    data, err = _get("/search/issues", params={"q": f"repo:{repo} {query}", "per_page": 5})
    if err:
        return err
    items = data.get("items", [])
    if not items:
        return f"在 {repo} 中未找到与「{query}」相关的 issue/PR"
    lines = []
    for it in items:
        kind = "PR" if "pull_request" in it else "issue"
        lines.append(f"#{it['number']} [{kind}] {it['title']} ({it['state']})")
    return "\n".join(lines)


@mcp.tool()
def get_pull_request(repo: str, number: int) -> str:
    """获取某个 PR 的基本信息（标题/状态/作者/变更规模/描述）。

    Args:
        repo: 仓库全名，例如 "psf/requests"。
        number: PR 编号（整数）。
    """
    data, err = _get(f"/repos/{repo}/pulls/{number}")
    if err:
        return err
    user = (data.get("user") or {}).get("login", "?")
    body = (data.get("body") or "").strip()
    return (
        f"PR #{number}：{data.get('title')}\n"
        f"状态：{data.get('state')}  作者：{user}\n"
        f"变更文件：{data.get('changed_files')} 个  增加 {data.get('additions')} 行 / "
        f"删除 {data.get('deletions')} 行\n"
        f"描述：{body[:500]}"
    )


@mcp.tool()
def get_pr_diff(repo: str, number: int) -> str:
    """获取某个 PR 的 diff（unified diff 格式），超长会自动截断。

    Args:
        repo: 仓库全名，例如 "psf/requests"。
        number: PR 编号（整数）。
    """
    text, err = _get(f"/repos/{repo}/pulls/{number}", accept="application/vnd.github.v3.diff")
    if err:
        return err
    if len(text) > _MAX_DIFF_CHARS:
        text = text[:_MAX_DIFF_CHARS] + "\n...[diff 过长，已截断]"
    return text


@mcp.tool()
def get_pr_files(repo: str, number: int) -> str:
    """获取某个 PR 变更的文件列表（文件名/状态/增删行数）。

    Args:
        repo: 仓库全名，例如 "psf/requests"。
        number: PR 编号（整数）。
    """
    data, err = _get(f"/repos/{repo}/pulls/{number}/files")
    if err:
        return err
    if not data:
        return "该 PR 没有文件变更"
    lines = []
    for f in data[:20]:
        lines.append(f"{f['status']:8} +{f['additions']:>4}/-{f['deletions']:<4}  {f['filename']}")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
