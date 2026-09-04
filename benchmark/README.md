# 评测集说明

`code_qa.json` 是代码检索/问答的标注集，每条标注一个「问题 → 目标代码位置」。

## 格式

```json
[
  { "query": "这个问题问什么", "file": "相对仓库根的文件路径", "line": 410 }
]
```

- `query`：一句自然的代码问题（用户会怎么问，就怎么写）。
- `file`：答案所在文件的**相对仓库根**路径（与 `index_repo.py` 的索引路径一致）。
- `line`：目标**函数/类的 `def`/`class` 起始行号**。
- 命中判定：检索结果的 top-k 里存在「同 file + 同 start_line」的 chunk 即算命中。

## 怎么填（换到你真正要评测的仓库）

1. 索引目标仓库（比如 `requests`）：
   ```bash
   python index_repo.py /path/to/requests -o code_index.json
   ```
2. 抽样看 chunk，拿到准确的 `file:line`：
   ```bash
   python eval_benchmark.py --suggest
   ```
   它会打印类似 `requests/models.py:410  Response.raise_for_status | def raise_for_status...`
3. 针对抽样出来的函数/类，人工写 20~40 条「问题 → file:line」填进 `code_qa.json`。

> 建议问题覆盖真实使用场景（「超时重试怎么实现」「这个函数在什么条件下抛异常」），
> 而不是照抄函数名，这样 Recall 才是有意义的评测。

## 内置示例

当前 `code_qa.json` 里的 3 条标注指向**本项目自身**，用于验证整条链路能跑通：

```bash
python index_repo.py . -o code_index.json   # 索引本项目
python eval_benchmark.py                    # 跑 Recall@1/@3/@5
```

跑通后，把 `code_qa.json` 换成你目标仓库的标注即可。
