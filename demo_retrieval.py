"""演示：混合检索 + Rerank + Recall@k 评估。

1. 用一个小型文档库，对比 BM25 / TF-IDF 向量 / 混合检索 的 Recall@1、@3、@5。
2. 用 LLM-as-reranker 对一次查询的 top-k 候选重排，观察排序是否更合理。
"""
from retrieval import HybridRetriever, LLMReranker, evaluate

corpus = [
    "小明喜欢喝咖啡，尤其是手冲单品豆，对咖啡很挑剔",
    "公司食堂周一供应宫保鸡丁、米饭和例汤",
    "明天下午三点在三楼会议室讨论项目进度",
    "小明最近在学习机器学习，重点在神经网络与反向传播",
    "图书馆新到了一批 Python 编程和数据结构书籍",
    "机器学习中的梯度下降是优化模型参数的常用方法",
    "小明周末计划去爬山",
]

queries_gt = {
    "小明喝什么": [0],
    "机器学习和梯度下降": [3, 5],
    "会议室几点": [2],
    "食堂吃什么": [1],
    "python 书籍": [4],
}


def main():
    print("=== 检索效果对比（Recall@k，越高越好）===")
    print(evaluate(corpus, queries_gt))

    # 演示 LLM 重排
    ret = HybridRetriever()
    ret.fit(corpus)
    query = "小明在学习机器学习"
    top = ret.search(query, top_k=5)
    print("\n=== LLM-as-reranker 演示 ===")
    print("查询：", query)
    print("混合检索 top-5：")
    for i, _ in top:
        print(f"  [{i}] {corpus[i]}")
    reranker = LLMReranker()
    order = reranker.rerank(query, [corpus[i] for i, _ in top], top_k=3)
    print("重排后 top-3：")
    for pos in order:
        doc_idx = top[pos][0]
        print(f"  [{doc_idx}] {corpus[doc_idx]}")


if __name__ == "__main__":
    main()
