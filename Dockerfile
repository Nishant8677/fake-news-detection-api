# Fake News Detection API — 6-class LIAR claim classifier served over FastAPI.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_VERBOSITY=error

WORKDIR /app

# CPU-only torch keeps the image ~2GB instead of ~6GB.
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY inference/ ./inference/
COPY preprocess.py .

# Model weights are not baked into the image (438 MB safetensors).
# Mount at runtime:  docker run -v $(pwd)/model:/app/model -p 8000:8000 <image>
ENV MODEL_DIR=/app/model

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health', timeout=3).status_code==200 else 1)"

CMD ["uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
