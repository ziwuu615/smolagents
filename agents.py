"""多智能体编排：Master-Worker（Router + 专精执行者）+ Critic 盲审。

用 smolagents 1.26.0 的 `managed_agents` 机制搭层级化多智能体：
- router（master）：拆解任务，委派给专精 worker，综合成带 file:line 引证的最终答案；
- code_search（worker）：代码库检索，输出带 file:line 的候选；
- reviewer（worker，Critic）：独立盲审，只看到「任务 + 草稿答案」，不接触检索来源。

对标参考简历「Master-Worker + Router 调度」与「Critic 盲审 + 有界返工」。
"""
from smolagents import CodeAgent, tool

from code_index import CodeIndex


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


def build_code_search_agent(index: CodeIndex, model):
    """代码检索专精 worker：只负责检索并输出带 file:line 的候选。"""
    return CodeAgent(
        tools=[make_search_tool(index)],
        model=model,
        name="code_search",
        description="代码库检索专家：给定问题，检索并返回最相关的代码块（含 file:line 与代码片段）。",
    )


def build_reviewer_agent(model):
    """Critic 盲审 worker：无检索工具，只基于任务与草稿答案做严格审查。"""
    return CodeAgent(
        tools=[],
        model=model,
        name="reviewer",
        description="严格盲审专家：判断草稿答案是否准确、是否引用正确的代码位置、是否完整回答；达标只回复 OK，否则指出问题并给出改进方向。",
    )


def build_master_agent(workers: list, model):
    """Router master：持有专精 worker，负责拆解、委派并综合最终答案。"""
    return CodeAgent(
        tools=[],
        model=model,
        managed_agents=workers,
        name="router",
        description="总调度：把用户问题分派给合适的专精 worker（如代码检索），综合成带 file:line 引证的最终答案。",
        instructions=(
            "你是代码库问答的总调度。对于任何关于代码库/源码的问题，必须先委派给 "
            "code_search 检索，再基于检索结果（含 file:line）组织回答，并在回答中"
            "引用具体的 file:line。不要凭模型自身记忆直接回答。"
        ),
    )


def build_code_qa_system(index: CodeIndex, model):
    """构建「router + code_search + reviewer」多智能体系统。

    返回 (router, code_search, reviewer)。router 通过 managed_agents 调度 code_search；
    reviewer 是独立盲审 agent，由 run_with_critic 在外部调用。
    """
    code_search = build_code_search_agent(index, model)
    reviewer = build_reviewer_agent(model)
    router = build_master_agent([code_search], model)
    return router, code_search, reviewer


def run_with_critic(agent, reviewer, task: str, max_rounds: int = 2):
    """生成 → 盲审 → 返工（最多 max_rounds 轮）。

    reviewer 是独立 Agent，只看到「任务 + 草稿答案」、看不到检索来源（盲审）；
    盲审意见回填给生成 agent 返工。返回 (最终答案, 轮数, 最后一次盲审意见)。
    """
    feedback = None
    verdict = ""
    for rnd in range(1, max_rounds + 1):
        current = task if feedback is None else f"{task}\n\n[上一轮盲审意见]\n{feedback}"
        answer = str(agent.run(current))
        verdict = str(reviewer.run(
            f"任务：{task}\n草稿答案：\n{answer}\n"
            "请盲审该答案：是否准确、是否引用正确的代码位置、是否完整回答。达标只回复「OK」，否则指出具体问题。"
        ))
        if verdict.strip().upper().startswith("OK"):
            return answer, rnd, verdict
        feedback = verdict
    return answer, max_rounds, verdict
