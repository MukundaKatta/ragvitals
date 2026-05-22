# Cloud Run dashboard for ragvitals. Wraps the published library + a
# Streamlit UI so judges can interact with the drift detector live.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Install the library itself + Streamlit.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . "streamlit>=1.40"

# Copy the dashboard last so tweaks don't bust the deps layer.
COPY app.py ./

EXPOSE 8080

# Cloud Run health probe hits /. Streamlit's default health endpoint is
# /_stcore/health, but the root works fine for liveness too.
CMD streamlit run app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
