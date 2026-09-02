"""smolagents 快速上手：一个带自定义工具的 CodeAgent。

运行前准备：
1. 把 .env.example 复制成 .env，填入你的 API Key（推荐 DeepSeek，国内可直连、便宜）
2. 运行： python demo.py
"""
from smolagents import CodeAgent

from config import build_model
from custom_tools import calculator, count_words, get_current_time


def main():
    model = build_model()

    # CodeAgent：smolagents 的核心卖点——让 LLM 直接写 Python 来调用工具，
    # 而不是输出 JSON。整个执行循环可读、可调试。
    agent = CodeAgent(
        tools=[calculator, get_current_time, count_words],
        model=model,
    )

    question = (
        "现在是几点？然后帮我算一下 (2 + 3) * 8 等于多少，"
        "最后统计一下「你好，hello world，很高兴认识你」这句话的中英文字数。"
    )

    print("=" * 60)
    print("提问：", question)
    print("=" * 60)
    result = agent.run(question)
    print("\n最终结果：")
    print(result)


if __name__ == "__main__":
    main()
