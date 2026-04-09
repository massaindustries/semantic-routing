#!/usr/bin/env python3
"""
Brick Complexity Extractor Server

Loads regolo/brick-complexity-extractor (LoRA on Qwen3.5-0.8B) for query complexity
classification into easy/medium/hard.

Endpoints:
  POST /classify  {"text": "..."} → {"label": "hard", "confidence": 0.9412}
  GET  /health    → {"status": "ok", "model": "regolo/brick-complexity-extractor"}

Usage:
  python server.py --port 8093
  python server.py --port 8093 --device cpu
  python server.py --port 8093 --device cuda
"""

import argparse
import logging
import time

import torch
import torch.nn.functional as F
from aiohttp import web
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_ID = "regolo/brick-complexity-extractor"
BASE_MODEL_ID = "Qwen/Qwen3.5-0.8B"

LABELS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = (
    "You are a query complexity classifier for an LLM routing system. "
    "Classify the following query into exactly one category:\n"
    "- easy: Simple factual recall, 1-2 reasoning steps, basic knowledge\n"
    "- medium: Moderate analysis, 3-5 reasoning steps, domain familiarity needed\n"
    "- hard: Complex multi-step reasoning, expert knowledge, synthesis across domains\n\n"
    "Respond with only the label: easy, medium, or hard."
)


class BrickComplexityServer:
    def __init__(self, device="auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        logger.info(f"Loading base model {BASE_MODEL_ID} on {self.device} ({dtype})...")
        start = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_ID, trust_remote_code=True
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        logger.info(f"Loading LoRA adapter {MODEL_ID}...")
        self.model = PeftModel.from_pretrained(base_model, MODEL_ID)
        self.model.to(self.device)
        self.model.eval()

        # Pre-compute token IDs for label extraction
        self.label_ids = {
            label: self.tokenizer.encode(label, add_special_tokens=False)[0]
            for label in LABELS
        }

        elapsed = time.time() - start
        logger.info(
            f"Model loaded in {elapsed:.1f}s on {self.device} "
            f"(label_ids: {self.label_ids})"
        )

    def classify(self, text: str) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify: {text}"},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1, :]

        label_logits = torch.tensor(
            [logits[self.label_ids[l]].float() for l in LABELS]
        )
        probs = F.softmax(label_logits, dim=0)

        best_idx = probs.argmax().item()
        label = LABELS[best_idx]
        confidence = probs[best_idx].item()

        return {"label": label, "confidence": round(confidence, 4)}


server_instance: BrickComplexityServer | None = None


async def handle_classify(request):
    try:
        data = await request.json()
        text = data.get("text", "")
        if not text:
            return web.json_response({"error": "No text provided"}, status=400)

        start = time.time()
        result = server_instance.classify(text)
        elapsed_ms = (time.time() - start) * 1000

        logger.info(
            f"Classified in {elapsed_ms:.1f}ms: "
            f"label={result['label']}, confidence={result['confidence']}"
        )

        return web.json_response(result)
    except Exception as e:
        logger.error(f"Classification error: {e}", exc_info=True)
        return web.json_response({"error": str(e)}, status=500)


async def handle_health(request):
    return web.json_response(
        {
            "status": "ok",
            "model": MODEL_ID,
            "device": str(server_instance.device) if server_instance else "not loaded",
        }
    )


def main():
    global server_instance

    parser = argparse.ArgumentParser(
        description="Brick Complexity Extractor Server"
    )
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"]
    )
    args = parser.parse_args()

    server_instance = BrickComplexityServer(device=args.device)

    app = web.Application()
    app.router.add_post("/classify", handle_classify)
    app.router.add_get("/health", handle_health)

    logger.info(f"Starting server on port {args.port}")
    web.run_app(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
