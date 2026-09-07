"""混合检索 + 重排序 + 评估（RAG 检索标准流水线）。

四个能力：
1. BM25：Okapi BM25 稀疏检索，纯 numpy 实现（IDF + 词频饱和 + 长度归一化），不调库、算法可讲。
2. 混合检索：BM25 与 TF-IDF 向量结果用 RRF（Reciprocal Rank Fusion，k=60）融合。
3. 重排序：LLM-as-reranker，用 LLM 对 top-k 候选按相关性重排。
4. 评估：Recall@k 对比各方案，量化检索效果。
"""
import json
import math
import re
from collections import Counter

import numpy as np

from config import build_openai_client, get_model_id
from embedding import TfidfEmbedder


def tokenize(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text) + re.findall(r"[一-鿿]", text)


def code_tokenize(text: str) -> list[str]:
    """代码感知分词：把 camelCase / snake_case / PascalCase 标识符拆成子词。

    普通 `tokenize` 先转小写再按 [a-z0-9]+ 整段匹配，会把 getUserById 直接压成
    一个 token "getuserbyid"，丢失了大小写边界。这里先保留大小写切分，再拆：
      getUserById  -> get / user / by / id
      snake_case   -> snake / case
      HTTPServer   -> http / server
    这是「代码检索」区别于「文本检索」的关键，检索时能按子词命中。
    """
    tokens: list[str] = []
    for raw in re.findall(r"[A-Za-z0-9]+", text):
        # 先拆下划线 / 连字符分隔（snake_case / kebab-case）
        for part in re.split(r"[_-]+", raw):
            if not part:
                continue
            # 小写/数字 到 大写 的边界：getUserById -> get User By Id
            part = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", part)
            # 连续大写缩写到小写开头的边界：HTTPServer -> HTTP Server
            part = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", part)
            tokens.extend(part.lower().split())
    tokens.extend(re.findall(r"[一-鿿]", text))  # 中文单字，与 tokenize 保持一致
    return tokens


class BM25:
    """Okapi BM25：k1 控制词频饱和，b 控制文档长度归一化。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def fit(self, corpus: list[str]) -> None:
        self.corpus = [tokenize(t) for t in corpus]
        self.n = len(self.corpus)
        self.doc_len = [len(d) for d in self.corpus]
        self.avgdl = sum(self.doc_len) / max(1, self.n)
        df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                df[term] += 1
        # 平滑 IDF：ln((N - df + 0.5) / (df + 0.5) + 1)
        self.idf = {
            term: math.log((self.n - c + 0.5) / (c + 0.5) + 1)
            for term, c in df.items()
        }

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n)
        for term in tokenize(query):
            idf = self.idf.get(term, 0.0)
            if idf == 0.0:
                continue
            for i, doc in enumerate(self.corpus):
                f = doc.count(term)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                out[i] += idf * f * (self.k1 + 1) / denom
        return out


def _rank_from_scores(scores: np.ndarray) -> np.ndarray:
    """分数 -> 排名（从 1 起，降序）。"""
    order = np.argsort(-scores)
    rank = np.empty_like(order)
    rank[order] = np.arange(1, len(order) + 1)
    return rank


class HybridRetriever:
    """稀疏（BM25）+ 稀疏向量（TF-IDF）+ 可选稠密（语义）向量，用 RRF 融合排名。

    混合检索的真正价值在于「稀疏 + 稠密」互补：BM25 擅长精确词匹配，
    稠密向量擅长语义相似，二者取长补短。
    """

    def __init__(self, rrf_k: int = 60, dense=None):
        self.rrf_k = rrf_k
        self.bm25 = BM25()
        self.vector = TfidfEmbedder()
        self.dense = dense

    def fit(self, corpus: list[str]) -> None:
        self.corpus = corpus
        self.bm25.fit(corpus)
        self.vector.fit(corpus)
        self.doc_vecs = [self.vector.embed(t) for t in corpus]
        self.doc_dense = None
        if self.dense is not None:
            if hasattr(self.dense, "embed_many"):
                self.doc_dense = self.dense.embed_many(corpus)
            else:
                self.doc_dense = [self.dense.embed(t) for t in corpus]

    def bm25_only(self, query: str, top_k: int) -> list[int]:
        return np.argsort(-self.bm25.scores(query))[:top_k].tolist()

    def vector_only(self, query: str, top_k: int) -> list[int]:
        qv = self.vector.embed(query)
        s = np.array([float(qv @ v) for v in self.doc_vecs])
        return np.argsort(-s)[:top_k].tolist()

    def dense_only(self, query: str, top_k: int) -> list[int]:
        if self.doc_dense is None:
            raise RuntimeError("未配置稠密 embedder")
        qv = self.dense.embed(query)
        s = np.array([float(qv @ v) for v in self.doc_dense])
        return np.argsort(-s)[:top_k].tolist()

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        ranks = [
            _rank_from_scores(self.bm25.scores(query)),
            _rank_from_scores(
                np.array([float(self.vector.embed(query) @ v) for v in self.doc_vecs])
            ),
        ]
        if self.doc_dense is not None:
            qv = self.dense.embed(query)
            ranks.append(
                _rank_from_scores(np.array([float(qv @ v) for v in self.doc_dense]))
            )
        fused = sum(1.0 / (self.rrf_k + r) for r in ranks)
        idx = np.argsort(-fused)[:top_k]
        return [(int(i), float(fused[i])) for i in idx]


class LLMReranker:
    """LLM-as-reranker：用 LLM 对候选文档按相关性重排（比纯打分更懂语义）。"""

    def __init__(self, client=None, model_id=None):
        self.client = client or build_openai_client()
        self.model_id = model_id or get_model_id()

    def rerank(self, query: str, docs: list[str], top_k: int = 3) -> list[int]:
        numbered = "\n".join(f"[{i}] {docs[i]}" for i in range(len(docs)))
        prompt = (
            "你是检索重排器。给定查询和候选文档，按与查询的相关性从高到低排序。\n"
            f"查询：{query}\n候选文档：\n{numbered}\n"
            "只输出一个 JSON 数组，内容为文档编号、按相关性降序，例如 [2,0,1]。不要输出其他文字。"
        )
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        try:
            order = json.loads(text)
            return [int(i) for i in order[:top_k]]
        except (json.JSONDecodeError, ValueError):
            return list(range(min(top_k, len(docs))))  # 解析失败回退原顺序


def evaluate(corpus: list[str], queries_gt: dict[str, list[int]]):
    """对比 BM25 / TF-IDF 向量 / 混合检索 的 Recall@1、@3、@5。"""
    ret = HybridRetriever()
    ret.fit(corpus)
    methods = {
        "BM25": lambda q, k: ret.bm25_only(q, k),
        "TF-IDF 向量": lambda q, k: ret.vector_only(q, k),
        "混合(BM25+向量)": lambda q, k: [i for i, _ in ret.search(q, top_k=k)],
    }
    ks = [1, 3, 5]
    header = f"{'方法':<16}" + "".join(f"{'Recall@' + str(k):>12}" for k in ks)
    lines = [header]
    for name, fn in methods.items():
        cells = []
        for k in ks:
            recalls = []
            for q, gt in queries_gt.items():
                got = fn(q, k)
                recalls.append(len(set(got) & set(gt)) / len(gt) if gt else 0.0)
            cells.append(f"{np.mean(recalls):.3f}")
        lines.append(f"{name:<16}" + "".join(f"{c:>12}" for c in cells))
    return "\n".join(lines)
