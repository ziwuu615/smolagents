"""推理增强：自一致性（Self-Consistency）与自我反思（Reflexion）。

1. Self-Consistency（CoT-SC）：同一任务多路采样，用多数投票降低单次采样的随机误差；
   无多数时用 LLM 聚合，选出最一致的答案。
2. Reflexion：生成 → 自我批判 → 将批判反馈回下一轮，循环改进直到通过或达轮数上限。

两者都是对 Agent 输出的后处理/增强，不侵入 Agent 内部，可叠加在任意 Agent 之上。
"""
from collections import Counter

from config import build_openai_client, get_model_id


def _llm(messages: list[dict], temperature: float = 0.0) -> str:
    client = build_openai_client()
    resp = client.chat.completions.create(
        model=get_model_id(), messages=messages, temperature=temperature
    )
    return resp.choices[0].message.content.strip()


def self_consistency(agent, task: str, n: int = 3):
    """多路采样 + 多数投票；无多数时用 LLM 聚合。返回 (最终答案, 采样列表)。"""
    answers = [str(agent.run(task)) for _ in range(n)]
    counts = Counter(answers)
    top, freq = counts.most_common(1)[0]
    if freq > 1:
        return top, answers
    numbered = "\n".join(f"[{i}] {a}" for i, a in enumerate(answers))
    prompt = (
        f"任务：{task}\n以下是 {n} 次采样得到的答案，请选出最正确/最一致的一个，"
        f"只输出该答案本身：\n{numbered}"
    )
    best = _llm([{"role": "user", "content": prompt}], temperature=0.0)
    return best, answers


def reflexion(agent, task: str, max_rounds: int = 3):
    """生成 → 批判 → 反馈改进。返回 (最终答案, 轮数, 最后一次批判反馈)。"""
    feedback = None
    for r in range(1, max_rounds + 1):
        current_task = (
            task if feedback is None else f"{task}\n\n[上一轮的批判反馈]\n{feedback}"
        )
        answer = str(agent.run(current_task))
        critic = _llm([{"role": "user", "content": (
            f"你是严格的评审。任务：{task}\n候选答案：{answer}\n"
            "判断该答案是否完全达到要求。若已达标，只回复「OK」；若未达标，指出具体问题并给出改进方向。"
        )}], temperature=0.0)
        if critic.strip().upper().startswith("OK"):
            return answer, r, feedback
        feedback = critic
    return answer, max_rounds, feedback
