FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev

# The SPECTER2 model (~450 MB) downloads on first use into the data volume.
ENV PATH="/app/.venv/bin:$PATH" \
    ARXIV_BOT_CONFIG=/config/config.toml \
    ARXIV_BOT_DB=/data/arxiv-bot.db \
    HF_HOME=/data/huggingface
CMD ["arxiv-bot", "run"]
