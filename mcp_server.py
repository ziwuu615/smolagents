"""自建 MCP Server：用官方 mcp SDK（FastMCP）暴露工具，走 stdio 协议。

为什么值得做：
- 用 @tool 注册的工具只能在 smolagents 内部用；
- 用 MCP 暴露的工具，任何 MCP 客户端都能用（Claude Desktop、Cursor、其他 Agent 框架），
  因为 MCP 是「Agent 调用外部工具」的通用协议，与具体框架解耦。

运行方式：
1. 直接跑：python mcp_server.py （stdio 模式，通过 stdin/stdout 与客户端通信）
2. 被 smolagents 通过 MCPClient 作为子进程拉起（见 demo_mcp.py）
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("我的工具箱")


@mcp.tool()
def get_weather(city: str) -> str:
    """查询某个城市的天气（当前为内置示例数据，替换函数体即可接真实天气 API）。

    Args:
        city: 城市名，例如 "北京"。
    """
    weather = {
        "北京": "晴，25°C，微风",
        "上海": "多云，27°C，东南风 3 级",
        "深圳": "阵雨，29°C，湿度 85%",
    }
    return weather.get(city, f"{city} 暂无数据（示例服务，仅内置三城）")


@mcp.tool()
def convert_unit(value: float, from_unit: str, to_unit: str) -> str:
    """在常用单位之间换算：长度(km/mile)、重量(kg/lb)、温度(c/f)。

    Args:
        value: 要换算的数值。
        from_unit: 原单位。
        to_unit: 目标单位。
    """
    factors = {
        ("km", "mile"): 0.621371,
        ("mile", "km"): 1.60934,
        ("kg", "lb"): 2.20462,
        ("lb", "kg"): 0.453592,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in factors:
        return f"{value} {from_unit} = {value * factors[key]:.4f} {to_unit}"
    if key == ("c", "f"):
        return f"{value}°C = {value * 9 / 5 + 32:.2f}°F"
    if key == ("f", "c"):
        return f"{value}°F = {(value - 32) * 5 / 9:.2f}°C"
    return f"暂不支持 {from_unit} → {to_unit} 的换算"


if __name__ == "__main__":
    mcp.run(transport="stdio")
