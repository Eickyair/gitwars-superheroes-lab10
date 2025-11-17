FROM python:3.11-slim

WORKDIR /app

COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r api/requirements.txt
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1


CMD ["python","api/main.py"]
