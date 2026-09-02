"""自定义工具示例：用 @tool 装饰器把任意 Python 函数变成 Agent 可调用的工具。

这是 smolagents 最核心的扩展点——你写一个普通函数，加一行 @tool 装饰器，
再把它传给 CodeAgent，LLM 就能自主决定何时、如何调用它。
后续你可以在这里继续加"查数据库""调公司内部 API"等真实工具。

注意：smolagents 会解析 docstring 生成工具的参数 schema，
所以每个参数都必须在 docstring 的 Args: 段里写清楚描述。
"""
import re
from datetime import datetime

from smolagents import tool


@tool
def calculator(expression: str) -> str:
    """计算一个纯算术表达式的值。

    Args:
        expression: 要计算的算术表达式，例如 "3 * (2 + 5)"。
    """
    # 只允许数字和基本运算符，禁止任意代码执行（安全收口，面试可讲的点）
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expression):
        return "非法表达式：只允许数字和 + - * / ( ) ."
    try:
        # eval 用空 builtins，避免执行到任何函数/属性
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"计算失败：{e}"


@tool
def get_current_time() -> str:
    """获取当前的日期和时间，格式形如 2026-09-02 14:30:00。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def count_words(text: str) -> str:
    """统计一段文本的英文单词数、中文字数和总字符数。

    Args:
        text: 需要统计的文本内容。
    """
    en_words = len(re.findall(r"[A-Za-z]+", text))
    cn_chars = len(re.findall(r"[一-鿿]", text))
    return f"英文单词数={en_words}，中文字数={cn_chars}，总字符数={len(text)}"
