# Hugging Face Space image. Mirrors the repository Dockerfile, with three
# differences forced by the platform:
#   - listens on 7860 (the port HF Spaces routes to)
#   - runs as uid 1000, because Spaces containers are not root
#   - pulls weights from the Hub at startup rather than a mounted volume,
#     since a Space has nothing to mount
#
# The serving code is CLONED FROM THE REPOSITORY AT A PINNED COMMIT rather than
# copied into this Space. Copying would create a second copy of inference/app.py
# that could silently drift from the one the benchmarks describe -- the exact
# failure this project exists to avoid. Bump REPO_REF to deploy a new version;
# the running Space then names the commit it is serving.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_VERBOSITY=error \
    LOG_PREDICTIONS=0

# Non-root, per HF Spaces. HF_HOME lands inside the writable home directory so
# the model cache does not try to write to a read-only path.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

USER root
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
USER user

# Pinned commit of the serving code. Single source of truth: this is the same
# inference/app.py that BENCHMARK.md and results/ describe.
ARG REPO_URL=https://github.com/Nishant8677/fake-news-detection-api.git
ARG REPO_REF=main
RUN git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" /tmp/src \
    && cp -r /tmp/src/inference ./inference \
    && (cd /tmp/src && git rev-parse HEAD > $HOME/app/SERVING_COMMIT) \
    && rm -rf /tmp/src

COPY --chown=user app.py .

# Set at build/run time to the Hub model repo, e.g. Nishant8677/fake-news-liar-bert
ENV MODEL_DIR=""

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:7860/health', timeout=3).status_code==200 else 1)"

CMD ["python", "app.py"]
