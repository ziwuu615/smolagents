"""代码索引器：把仓库切分成带 file:line 的代码块，建立可检索索引。

设计要点：
1. AST 切块：用标准库 `ast` 按「函数 / 类 / 方法」粒度切分（自带行号），
   比按固定行数滑窗更贴合代码语义；对解析失败的文件直接跳过。
2. 代码感知分词：索引前用 `code_tokenize` 把标识符拆成子词
   （getUserById → get/user/by/id），弥补普通分词无法命中驼峰/下划线标识符的不足。
3. 复用 `retrieval.HybridRetriever`（BM25 + TF-IDF 向量 + RRF 融合），
   可选接 `DashScopeEmbedder` 做稠密语义检索；Embedder 换掉，存储层不动。
4. 结果带 file:line，回答可追溯到具体代码行（可审计）。

产物：`code_index.json`（chunk 列表），可被 `CodeIndex.load` 直接加载复用。
"""
import ast
import json
import os
from pathlib import Path

import numpy as np

from retrieval import HybridRetriever, LLMReranker, code_tokenize

# 噪音目录：不参与索引
SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", ".tox",
    "build", "dist", "site-packages", ".idea", ".mypy_cache", ".pytest_cache",
}

# 单个 chunk 存进索引的最大行数：超大函数截断，避免撑爆上下文（进模型的仍是完整 file:line）
MAX_CHUNK_LINES = 60


def iter_py_files(repo: Path, skip_tests: bool = False):
    """遍历仓库下的 .py 文件，剪枝噪音目录（含 .venv），可选跳过 tests。"""
    for root, dirs, files in os.walk(repo):
        dirs[:] = [
            d for d in dirs
            if d not in SKIP_DIRS and not (skip_tests and d in {"test", "tests"})
        ]
        for name in sorted(files):
            if name.endswith(".py"):
                yield Path(root) / name


def _make_chunk(rel: str, lines: list[str], node, name: str, code: str) -> dict:
    """由 AST 节点构造一个 chunk，含 file:line 元数据与 code_tokenize 后的检索文本。"""
    start = getattr(node, "lineno", 1)
    end = getattr(node, "end_lineno", start)
    sig = lines[start - 1].strip() if 0 < start <= len(lines) else ""
    doc = ast.get_docstring(node) or ""
    search_text = " ".join(code_tokenize(f"{name} {sig} {doc} {code}"))
    return {
        "file": rel,
        "start_line": start,
        "end_line": end,
        "name": name,
        "signature": sig,
        "code": code,
        "search_text": search_text,
    }


def ast_chunks(file: Path, repo_root: Path) -> list[dict]:
    """解析单个文件，产出函数/类/方法粒度的 chunk 列表。解析失败返回空。"""
    try:
        src = file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    lines = src.splitlines()
    rel = file.relative_to(repo_root).as_posix()
    chunks: list[dict] = []

    # 模块 docstring：作为文件级概览 chunk
    doc = ast.get_docstring(tree)
    if doc:
        chunks.append(_make_chunk(rel, lines, tree, name="<module>", code=doc))

    def _node_code(node) -> str:
        start = getattr(node, "lineno", 1) - 1
        end = getattr(node, "end_lineno", start + 1)
        body = lines[start: min(end, start + MAX_CHUNK_LINES)]
        return "\n".join(body).rstrip()

    def walk(body: list, prefix: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.append(
                    _make_chunk(rel, lines, node, name=f"{prefix}{node.name}",
                                code=_node_code(node))
                )
            elif isinstance(node, ast.ClassDef):
                chunks.append(
                    _make_chunk(rel, lines, node, name=f"{prefix}{node.name}",
                                code=_node_code(node))
                )
                # 类内方法：名字带类前缀（Response.raise_for_status）
                walk(node.body, prefix=f"{prefix}{node.name}.")

    walk(tree.body)
    return chunks


def ingest_repo(repo, skip_tests: bool = False) -> list[dict]:
    """遍历仓库 → AST 切块 → 返回带 id 的 chunk 列表。"""
    repo = Path(repo)
    chunks: list[dict] = []
    for file in iter_py_files(repo, skip_tests=skip_tests):
        chunks.extend(ast_chunks(file, repo))
    for i, c in enumerate(chunks):
        c["id"] = i
    return chunks


class CodeIndex:
    """代码库索引：切块 + 检索。检索时复用 HybridRetriever（BM25+TF-IDF+RRF）。"""

    def __init__(self, chunks: list[dict] | None = None, embedder=None):
        self.chunks = list(chunks or [])
        self.retriever = HybridRetriever(dense=embedder)
        if self.chunks:
            self._fit()

    def _fit(self) -> None:
        corpus = [c["search_text"] for c in self.chunks]
        self.retriever.fit(corpus)

    @classmethod
    def from_repo(cls, repo, skip_tests: bool = False, embedder=None):
        return cls(ingest_repo(repo, skip_tests=skip_tests), embedder=embedder)

    def build_dense(self, embedder, cache_path: str = "code_index_dense.npy",
                    batch_size: int = 10) -> "CodeIndex":
        """给所有 chunk 建稠密语义向量并缓存到 .npy（避免每次重跑都调 API）。

        缓存命中且 chunk 数一致时直接加载；否则批量 embed 后落盘。
        建好后 `search()` 会自动把稠密排名并入 RRF 融合（稀疏+稠密混合检索）。
        """
        corpus = [c["search_text"] for c in self.chunks]
        self.retriever.dense = embedder  # 让 search() 能对 query 做稠密 embedding
        cache = Path(cache_path)
        if cache.exists():
            try:
                cached = np.load(cache_path)
                if len(cached) == len(corpus):
                    self.retriever.doc_dense = [np.asarray(v, dtype=float) for v in cached]
                    print(f"[稠密] 复用缓存向量：{cache_path}（{len(cached)} 条）")
                    return self
            except Exception:
                pass
        print(f"[稠密] 计算 {len(corpus)} 条稠密向量（batch={batch_size}）...")
        if hasattr(embedder, "embed_many"):
            vecs = embedder.embed_many(corpus, batch_size=batch_size)
        else:
            vecs = [embedder.embed(t) for t in corpus]
        mat = np.array(vecs, dtype=float)
        np.save(cache_path, mat)
        self.retriever.doc_dense = [np.asarray(v, dtype=float) for v in mat]
        print(f"[稠密] 已缓存 → {cache_path}")
        return self

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """检索并返回带 file:line + score 的候选（查询同样做 code_tokenize）。"""
        q = " ".join(code_tokenize(query))
        out = []
        for idx, score in self.retriever.search(q, top_k=top_k):
            c = self.chunks[idx]
            out.append({**c, "score": round(float(score), 4)})
        return out

    def search_reranked(self, query: str, top_k: int = 5, candidate_k: int = 8) -> list[dict]:
        """先取 candidate_k 候选，再用 LLM-as-reranker 重排取 top_k。

        词面检索排序不够准（Recall@1 低但 Recall@5 高）；重排让 LLM 逐条读候选代码，
        把真正相关的那条排到前面，主要提升 Recall@1。候选代码截断到前 12 行控制 token。
        """
        q = " ".join(code_tokenize(query))
        candidates = self.retriever.search(q, top_k=candidate_k)
        if len(candidates) <= top_k:
            return [
                {**self.chunks[i], "score": round(float(s), 4)} for i, s in candidates
            ]
        docs = [
            f"{self.chunks[i]['name']} :: {self.chunks[i]['signature']}\n"
            f"{chr(10).join(self.chunks[i]['code'].splitlines()[:12])}"
            for i, _ in candidates
        ]
        order = LLMReranker().rerank(query, docs, top_k=top_k)
        out = []
        for pos in order:
            if 0 <= pos < len(candidates):
                idx, score = candidates[pos]
                out.append({**self.chunks[idx], "score": round(float(score), 4)})
        return out

    def save(self, path: str = "code_index.json") -> None:
        Path(path).write_text(
            json.dumps({"chunks": self.chunks}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str = "code_index.json", embedder=None):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data.get("chunks", []), embedder=embedder)
