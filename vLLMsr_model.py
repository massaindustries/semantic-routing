# Model Database for Regolo API
# IDs modelli senza prefisso ex:

MODELS = {
    "gpt-oss-120b": {
        "id": "gpt-oss-120b",
        "name": "GPT-OSS 120B",
        "type": "text",
        "description": "Modello principale per chat e reasoning",
        "max_tokens": 128000,
        "reasoning": True
    },
    "deepseek-r1-70b": {
        "id": "deepseek-r1-70b",
        "name": "DeepSeek R1 70B",
        "type": "text",
        "description": "Modello di reasoning",
        "max_tokens": 128000,
        "reasoning": True
    },
    "gpt-oss-20b": {
        "id": "gpt-oss-20b",
        "name": "GPT-OSS 20B",
        "type": "text",
        "description": "Modello di chat leggero",
        "max_tokens": 128000,
        "reasoning": False
    },
    "qwen3-coder-next": {
        "id": "qwen3-coder-next",
        "name": "Qwen3 Coder Next",
        "type": "text",
        "description": "Modello di coding",
        "max_tokens": 128000,
        "reasoning": False
    },
    "Llama-3.3-70b-Instruct": {
        "id": "Llama-3.3-70b-Instruct",
        "name": "Llama 3.3 70B Instruct",
        "type": "text",
        "description": "Modello di chat",
        "max_tokens": 128000,
        "reasoning": False
    },
    "mistral-small-3.2": {
        "id": "mistral-small-3.2",
        "name": "Mistral Small 3.2",
        "type": "text",
        "description": "Modello di chat leggero",
        "max_tokens": 128000,
        "reasoning": False
    },
    "gemma-3-27b-it": {
        "id": "gemma-3-27b-it",
        "name": "Gemma 3 27B IT",
        "type": "text",
        "description": "Modello di chat",
        "max_tokens": 128000,
        "reasoning": False
    },
    "Qwen3-8B": {
        "id": "Qwen3-8B",
        "name": "Qwen3 8B",
        "type": "text",
        "description": "Modello di chat",
        "max_tokens": 128000,
        "reasoning": False
    }
}

AUDIO_MODELS = {
    "whisper-large-v3": {
        "id": "whisper-large-v3",
        "name": "Whisper Large V3",
        "type": "stt",
        "description": "Speech-to-Text"
    }
}

VISION_MODELS = {
    "qwen3-vl-32b": {
        "id": "qwen3-vl-32b",
        "name": "Qwen3-VL 32B",
        "type": "vision",
        "description": "Vision model"
    }
}

OCR_MODELS = {
    "deepseek-ocr": {
        "id": "deepseek-ocr",
        "name": "DeepSeek OCR",
        "type": "ocr",
        "description": "OCR model"
    }
}

VLLM_MODELS = {
    **MODELS,
    **AUDIO_MODELS,
    **VISION_MODELS,
    **OCR_MODELS
}

DEFAULT_MODEL = "gpt-oss-120b"
DEFAULT_VLLM_MODEL = "MoM"
