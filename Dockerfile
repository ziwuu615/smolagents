FROM python:3.13-slim

WORKDIR /app

# 先装依赖，利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再拷贝源码
COPY . .

EXPOSE 8000

# 运行前需准备：.env（API Key）、code_index.json（索引）。
# 可通过 -v 挂载，或容器内先跑 `python index_repo.py <仓库路径>`。
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
