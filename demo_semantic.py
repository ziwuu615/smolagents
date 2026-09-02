"""演示：稠密语义向量 vs 词面检索。

核心结论：词面检索（BM25/TF-IDF）对「饮料 → 咖啡」这类同义关系给 0 分（无共同字），
而稠密向量（通义 text-embedding）能捕捉到 ~0.7 的语义相似度。
"""
from dotenv import load_dotenv

load_dotenv()

from embedding import DashScopeEmbedder, TfidfEmbedder
from retrieval import HybridRetriever


def main():
    dense = DashScopeEmbedder()

    # 1. 词面 vs 语义的直接对比
    print("=== 词面 vs 语义：「饮料」与「咖啡」===")
    tfidf = TfidfEmbedder()
    tfidf.fit(["咖啡", "饮料", "会议"])
    sim_lex = float(tfidf.embed("咖啡") @ tfidf.embed("饮料"))
    sim_dense = float(dense.embed("咖啡") @ dense.embed("饮料"))
    sim_dense_unrel = float(dense.embed("咖啡") @ dense.embed("会议"))
    print(f"TF-IDF 词面相似度（咖啡 vs 饮料）：{sim_lex:.4f}  ← 0（无共同字，无法匹配）")
    print(f"稠密向量语义相似度（咖啡 vs 饮料）：{sim_dense:.4f}  ← 高（语义相关）")
    print(f"稠密向量语义相似度（咖啡 vs 会议）：{sim_dense_unrel:.4f}  ← 低（语义无关）")

    # 2. 完整检索对比
    corpus = ["咖啡", "米饭", "会议"]
    print(f"\n=== 检索对比：查询「饮料」，语料 {corpus} ===")

    sparse_ret = HybridRetriever()  # 只有稀疏通道
    sparse_ret.fit(corpus)
    bm25 = sparse_ret.bm25.scores("饮料")
    print("BM25 分数（饮料 → 咖啡/米饭/会议）：", [round(float(x), 4) for x in bm25])
    print("  → 全为 0，词面检索完全失效（无法区分哪个相关）")

    hybrid_ret = HybridRetriever(dense=dense)  # 稀疏 + 稠密
    hybrid_ret.fit(corpus)
    top = hybrid_ret.search("饮料", top_k=3)
    print("混合检索（稀疏+稠密）排序：", [corpus[i] for i, _ in top])
    print("  → 稠密向量把「咖啡」正确排到第一")


if __name__ == "__main__":
    main()
