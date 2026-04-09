FROM python:3.12-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (CPU-only PyTorch to keep image small)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# Download models at build time for faster startup
RUN python3 -c "\
from transformers import AutoModelForCausalLM, AutoTokenizer; \
from peft import PeftModel; \
print('Downloading base model...'); \
AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B', trust_remote_code=True); \
base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-0.8B', trust_remote_code=True); \
print('Downloading LoRA adapter...'); \
PeftModel.from_pretrained(base, 'regolo/brick-complexity-extractor'); \
print('Models cached.'); \
"

COPY server.py .

EXPOSE 8093

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8093/health || exit 1

ENTRYPOINT ["python3", "server.py"]
CMD ["--port", "8093", "--device", "auto"]
