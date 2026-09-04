"""CLI：把仓库索引成 code_index.json，供问答/评测复用。

用法：
    python index_repo.py <仓库路径>                 # 纯稀疏索引（无外部 API）
    python index_repo.py <仓库路径> --dense          # 加通义稠密向量（需 QWEN_API_KEY）
    python index_repo.py <仓库路径> --skip-tests     # 跳过 test/tests 目录
"""
import argparse

from code_index import CodeIndex


def main():
    ap = argparse.ArgumentParser(description="把 Python 仓库索引成 code_index.json")
    ap.add_argument("repo", help="仓库路径")
    ap.add_argument("-o", "--output", default="code_index.json", help="输出文件（默认 code_index.json）")
    ap.add_argument("--skip-tests", action="store_true", help="跳过 test/tests 目录")
    ap.add_argument("--dense", action="store_true", help="使用通义稠密向量（需 QWEN_API_KEY）")
    args = ap.parse_args()

    embedder = None
    if args.dense:
        from embedding import DashScopeEmbedder
        embedder = DashScopeEmbedder()

    index = CodeIndex.from_repo(args.repo, skip_tests=args.skip_tests, embedder=embedder)
    index.save(args.output)
    print(f"已索引 {len(index.chunks)} 个代码块 → {args.output}")


if __name__ == "__main__":
    main()
