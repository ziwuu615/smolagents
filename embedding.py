"""向量化与相似度检索：把文本变成向量，用余弦相似度排序。

当前实现：TF-IDF 稀疏向量基线，纯 numpy，无外部依赖，可立即运行。
- 分词：英文按词、中文按单字
- 权重：sublinear TF × IDF；向量 L2 归一化后，余弦相似度 = 点积（省一次归一化）

Embedder 抽象成接口，后续替换 embed() 即可，存储层不变：
- 稠密语义向量：BGE / OpenAI / 通义 text-embedding 等预训练模型
- 大规模向量索引：FAISS / Milvus
"""
import math
import os
import re
from collections import Counter

import numpy as np


class TfidfEmbedder:
    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.array([])

    @staticmethod
    def tokenize(text: str) -> list[str]:
        text = text.lower()
        words = re.findall(r"[a-z0-9]+", text)  # 英文按词
        chars = re.findall(r"[一-鿿]", text)     # 中文按单字
        return words + chars

    def fit(self, corpus: list[str]) -> None:
        """用记忆库全集建立词表并计算 IDF（文档频率的倒数权重）。"""
        df: Counter = Counter()
        for text in corpus:
            for term in set(self.tokenize(text)):
                df[term] += 1
        self.vocab = {term: i for i, term in enumerate(df)}
        n = len(corpus)
        # 平滑 IDF：idf(t) = ln((1+n)/(1+df)) + 1
        self.idf = np.array(
            [math.log((1 + n) / (1 + df[term])) + 1.0 for term in self.vocab],
            dtype=float,
        )

    def embed(self, text: str) -> np.ndarray:
        """返回 L2 归一化后的 TF-IDF 向量（此时余弦相似度 = 点积）。"""
        vec = np.zeros(len(self.vocab), dtype=float)
        tf = Counter(self.tokenize(text))
        for term, count in tf.items():
            idx = self.vocab.get(term)
            if idx is not None:
                # sublinear TF：1 + ln(count)，抑制高频词过度主导
                vec[idx] = (1 + math.log(count)) * self.idf[idx]
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec


class DashScopeEmbedder:
    """通义 text-embedding 稠密语义向量（OpenAI 兼容接口）。

    与 TF-IDF 的本质区别：稠密向量来自预训练模型，能捕捉语义（如「饮料」≈「咖啡」），
    而 TF-IDF 只能做字面匹配。需要 QWEN_API_KEY。
    """

    def __init__(self, model_id: str = "text-embedding-v3", api_key: str | None = None):
        from openai import OpenAI

        api_key = api_key or os.getenv("QWEN_API_KEY")
        if not api_key:
            raise RuntimeError("DashScopeEmbedder 需要 QWEN_API_KEY（通义千问）")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model_id = model_id

    def embed(self, text: str) -> np.ndarray:
        r = self.client.embeddings.create(model=self.model_id, input=text)
        return np.array(r.data[0].embedding, dtype=float)
