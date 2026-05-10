FROM python:3.13-slim

LABEL maintainer="Lyon. <lyonzin@users.noreply.github.com>"
LABEL description="Local RAG System for Claude Code — Hybrid search + Cross-encoder Reranking + 12 MCP Tools + 20 Format Parsers"
LABEL org.opencontainers.image.source="https://github.com/lyonzin/knowledge-rag"

WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY mcp_server/ ./mcp_server/
COPY presets/ ./presets/
COPY config.example.yaml LICENSE README.md ./

RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding + reranker models (cached in image)
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')" && \
    python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"

VOLUME ["/app/documents", "/app/data"]

ENTRYPOINT ["python", "-m", "mcp_server.server"]
