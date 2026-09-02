"""演示：自一致性（多路采样投票）与自我反思（批判-改进循环）。"""
from smolagents import CodeAgent

from config import build_model
from reasoning import reflexion, self_consistency


def main():
    # ---- 1. Self-Consistency：采样温度 >0，多次生成取多数/聚合 ----
    print("=== Self-Consistency（自一致性）===")
    sample_agent = CodeAgent(tools=[], model=build_model(temperature=0.7))
    task = "用一句话向一个 8 岁孩子解释什么是机器学习。"
    best, answers = self_consistency(sample_agent, task, n=3)
    print("3 次采样答案：")
    for i, a in enumerate(answers):
        print(f"  [{i}] {a}")
    print("最终（多数投票/LLM 聚合）：", best)

    # ---- 2. Reflexion：生成 → 批判 → 反馈改进 ----
    print("\n=== Reflexion（自我反思）===")
    agent = CodeAgent(tools=[], model=build_model())
    task2 = "为一家咖啡店写一句广告语，要求：押韵、有画面感、不超过 15 个字。"
    answer, rounds, feedback = reflexion(agent, task2, max_rounds=3)
    print(f"最终答案（第 {rounds} 轮）：{answer}")
    if feedback:
        print("触发过的批判反馈：", feedback)


if __name__ == "__main__":
    main()
