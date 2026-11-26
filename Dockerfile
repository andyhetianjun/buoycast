FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY data ./data
COPY artifacts ./artifacts
COPY web ./web

EXPOSE 8000
CMD ["uv", "run", "--no-dev", "uvicorn", "--factory", "buoycast.api:create_app", "--host", "0.0.0.0", "--port", "8000"]
