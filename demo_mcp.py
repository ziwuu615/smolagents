"""演示：把自建 MCP server 的工具接进 smolagents Agent。

关键点：MCPClient 会启动 mcp_server.py 作为子进程，通过 stdio 通信，
把 MCP 工具包装成 smolagents 工具，和本地 @tool 工具一样被 Agent 调用。
"""
import sys

from mcp import StdioServerParameters
from smolagents import CodeAgent, MCPClient

from config import build_model


def main():
    # 用当前 venv 的 Python 把 mcp_server.py 作为 MCP server 子进程拉起
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
    )

    with MCPClient(server_params, structured_output=False) as mcp_tools:
        print("从 MCP server 拿到的工具：", [t.name for t in mcp_tools])

        agent = CodeAgent(tools=mcp_tools, model=build_model())
        question = (
            "北京今天天气怎么样？顺便帮我算一下 100 公里等于多少英里，"
            "以及 30 摄氏度等于多少华氏度。"
        )
        print("=" * 60)
        print("提问：", question)
        print("=" * 60)
        result = agent.run(question)
        print("\n最终结果：")
        print(result)


if __name__ == "__main__":
    main()
