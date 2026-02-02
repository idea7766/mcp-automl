# Multi-stage build for smaller final image
FROM python:3.11-slim as builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY src/mcp_automl ./src/mcp_automl

# Install dependencies and build
RUN uv pip install --system --no-cache .

# Final stage
FROM python:3.11-slim

# Install system dependencies for LightGBM (includes OpenMP)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create directory for experiments
RUN mkdir -p /root/.mcp-automl/experiments

# Set working directory
WORKDIR /workspace

# Run the MCP server
ENTRYPOINT ["mcp-automl"]
