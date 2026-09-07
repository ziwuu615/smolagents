"""持久化记忆模块：让 Agent 在多次运行（多个进程/会话）之间记住信息。

设计要点：
1. 存储层用 JSON 文件持久化，跨进程可见，无外部依赖。
2. 每条记忆带 id / 创建时间 / 访问次数；recall 命中会自增访问计数，
   这是后续做 LRU 淘汰、记忆优先级排序的基础。
3. recall 用「向量化 + 余弦相似度」检索（见 embedding.py）：
   文本 → TF-IDF 向量 → L2 归一化 → 点积即余弦 → 排序取 top-k。
   Embedder 抽象成接口，可无缝替换为 BGE/OpenAI 稠密向量或接 FAISS/Milvus。

依赖分离：PersistentMemory 只负责存储与检索，不依赖 smolagents；
build_memory_tools() 负责把它包装成 Agent 可调用的工具。
"""
import json
import time
from pathlib import Path

from smolagents import tool

from embedding import TfidfEmbedder


class PersistentMemory:
    def __init__(self, path: str = "memory_store.json", embedder=None):
        self.path = Path(path)
        self.entries: list[dict] = []
        self.embedder = embedder or TfidfEmbedder()
        self._load()
        self._refit()

    # ---------- 存储 ----------
    def _load(self) -> None:
        if self.path.exists():
            try:
                self.entries = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.entries = []

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(self.entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _refit(self) -> None:
        """用当前全部记忆重建词表/IDF，保证检索与最新数据一致。"""
        corpus = [e["text"] for e in self.entries]
        if corpus:
            self.embedder.fit(corpus)

    # ---------- 对外接口 ----------
    def remember(self, text: str) -> str:
        text = text.strip()
        for e in self.entries:
            if e["text"] == text:  # 去重：内容相同则只更新时间
                e["updated_at"] = time.time()
                self._save()
                return "这条记忆已存在，已更新时间戳"
        self.entries.append(
            {
                "id": len(self.entries) + 1,
                "text": text,
                "created_at": time.time(),
                "access_count": 0,
            }
        )
        self._save()
        self._refit()
        return f"已记住，当前共 {len(self.entries)} 条记忆"

    def recall(self, query: str, top_k: int = 3) -> str:
        if not self.entries:
            return "（暂无任何记忆）"
        self._refit()
        qvec = self.embedder.embed(query)
        scored = [
            (float(qvec @ self.embedder.embed(e["text"])), e) for e in self.entries
        ]
        scored.sort(key=lambda x: -x[0])
        top = [e for s, e in scored[:top_k] if s > 0]
        if not top:  # 完全无重叠时，退回最近记住的几条
            top = [e for s, e in scored[:top_k]]
        lines = []
        for e in top:
            e["access_count"] = e.get("access_count", 0) + 1
            lines.append(f"[{e['id']}] {e['text']}")
        self._save()
        return "\n".join(lines)

    def list_all(self) -> str:
        if not self.entries:
            return "（暂无任何记忆）"
        return "\n".join(f"[{e['id']}] {e['text']}" for e in self.entries)

    def clear(self) -> str:
        n = len(self.entries)
        self.entries = []
        self._save()
        self.embedder = TfidfEmbedder()  # 重置词表，避免残留
        return f"已清空 {n} 条记忆"


def build_memory_tools(memory: PersistentMemory):
    """把 PersistentMemory 包装成 smolagents 工具，供 CodeAgent 调用。"""

    @tool
    def remember(text: str) -> str:
        """把一条需要长期记住的信息写入持久化记忆，供以后的会话使用。

        Args:
            text: 要记住的内容，例如用户的姓名、偏好、目标等。
        """
        return memory.remember(text)

    @tool
    def recall(query: str) -> str:
        """从持久化记忆中检索最相关的历史记忆（向量化 + 余弦相似度排序）。

        Args:
            query: 用于检索记忆的查询，例如 "用户叫什么名字"。
        """
        return memory.recall(query)

    @tool
    def list_memories() -> str:
        """列出当前持久化记忆中的全部内容。"""
        return memory.list_all()

    return [remember, recall, list_memories]
