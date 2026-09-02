"""演示持久化记忆：两个互相独立的 Agent 会话，共享同一份记忆。

流程：
1. 会话 A：让 Agent 把用户信息记入长期记忆（写入 memory_store.json）
2. 会话 B：一个全新的 Agent 实例，没有任何上下文，
   只能靠 recall 工具从持久化记忆中找回答案。

这能证明「记忆真的跨会话持久化」，而不是模型自己短时记得。
"""
from smolagents import CodeAgent

from config import build_model
from memory import PersistentMemory, build_memory_tools


def main():
    memory = PersistentMemory("memory_store.json")  # 跨会话的持久化存储
    memory.clear()  # 每次演示从干净状态开始，避免"已存在"干扰模型
    memory_tools = build_memory_tools(memory)
    model = build_model()

    # ---- 会话 A：写入记忆 ----
    print("=" * 60)
    print("[会话 A] 任务：把用户信息记入长期记忆")
    print("=" * 60)
    agent_a = CodeAgent(tools=memory_tools, model=model)
    agent_a.run(
        "请把以下信息记入长期记忆：用户叫小明，喜欢喝咖啡，"
        "正在准备 AI Agent 应用工程师的实习面试。"
    )
    print("\n[存储层] memory_store.json 当前内容：")
    print(memory.list_all())

    # ---- 会话 B：全新 Agent，靠 recall 找回 ----
    # 注意：这里 new 了一个全新的 CodeAgent，不带任何历史上下文
    print("\n" + "=" * 60)
    print("[会话 B] 任务：回答关于用户的问题（会话 B 没有任何上下文）")
    print("=" * 60)
    agent_b = CodeAgent(tools=memory_tools, model=model)
    answer = agent_b.run("用户叫什么名字？喜欢喝什么？正在准备什么面试？")
    print("\n[会话 B 的最终回答]：")
    print(answer)


if __name__ == "__main__":
    main()
