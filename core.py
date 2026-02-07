from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import httpx
import os
import json

from vLLMsr_model import VLLM_MODELS, DEFAULT_MODEL, DEFAULT_VLLM_MODEL, BRICK_MODEL

app = FastAPI()

VLLM_SR_URL = os.getenv("VLLM_SR_URL", "http://vllm-sr:8888/v1/chat/completions")
REGOLO_API_URL = "https://api.regolo.ai/v1/chat/completions"
REGOLO_API_KEY = os.getenv("REGOLO_API_KEY", "")


def normalize_messages_for_vllm_sr(messages: list) -> list:
    """Normalizza i messaggi per vllm-sr convertendo content array in stringa (solo parti text)."""
    normalized = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        texts.append(text)
            normalized.append({
                "role": msg.get("role", "user"),
                "content": " ".join(texts) if texts else ""
            })
        else:
            normalized.append(msg)
    return normalized


def filter_function(messages: list) -> dict:
    """Analyze message structure to detect text/image/audio content."""
    has_image = False
    has_text = False
    has_audio = False
    
    for msg in messages:
        content = msg.get("content", "")
        
        # CASO A: Content is a list (multimodal with type blocks)
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "")
                    
                    if block_type == "text":
                        has_text = True
                    elif block_type == "image_url":
                        has_image = True
                    elif block_type == "audio":
                        has_audio = True
        
        # CASO B: Content is a string (plain text)
        elif isinstance(content, str) and content.strip():
            has_text = True
    
    return {
        "image": has_image,
        "text": has_text,
        "audio": has_audio
    }


async def call_vllm_sr(messages: list) -> dict:
    """Call vLLM Semantic Router - returns final response from Regolo via routing loop."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": DEFAULT_VLLM_MODEL,
        "messages": messages
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            VLLM_SR_URL,
            json=payload,
            headers=headers,
            timeout=300.0  # 5 minutes timeout for routing + Regolo call
        )
        response.raise_for_status()
        result = response.json()
        return result


async def call_regolo_llm(messages: list, model: str) -> dict:
    headers = {
        "Authorization": f"Bearer {REGOLO_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                REGOLO_API_URL,
                json=payload,
                headers=headers,
                timeout=300.0  # 5 minutes timeout for large models like gpt-oss-120b
            )
            response.raise_for_status()
            return response.json()
    except httpx.ReadTimeout:
        return {
            "error": {
                "message": f"Request timeout for model {model}. The model is taking too long to respond.",
                "type": "timeout",
                "code": "timeout"
            }
        }
    except httpx.HTTPStatusError as e:
        return {
            "error": {
                "message": f"HTTP error {e.response.status_code}: {str(e)}",
                "type": "http_error",
                "code": e.response.status_code
            }
        }
    except Exception as e:
        return {
            "error": {
                "message": f"Unexpected error: {str(e)}",
                "type": "unexpected_error",
                "code": "internal_error"
            }
        }


async def call_faster_whisper(audio_content: str) -> dict:
    headers = {
        "Authorization": f"Bearer {REGOLO_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": VLLM_MODELS["whisper-large-v3"]["id"],
        "file": audio_content or ""
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.regolo.ai/v1/audio/transcriptions",
                json=payload,
                headers=headers,
                timeout=120.0
            )
            response.raise_for_status()
            return response.json()
    except httpx.ReadTimeout:
        return {"error": {"message": "Whisper transcription timeout", "type": "timeout", "code": "timeout"}}
    except Exception as e:
        return {"error": {"message": str(e), "type": "unexpected_error", "code": "internal_error"}}


async def call_deepseek_ocr(image_content: str) -> dict:
    headers = {
        "Authorization": f"Bearer {REGOLO_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": VLLM_MODELS["deepseek-ocr"]["id"],
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_content}}
            ]}
        ]
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                REGOLO_API_URL,
                json=payload,
                headers=headers,
                timeout=120.0
            )
            response.raise_for_status()
            return response.json()
    except httpx.ReadTimeout:
        return {"error": {"message": "OCR processing timeout", "type": "timeout", "code": "timeout"}}
    except Exception as e:
        return {"error": {"message": str(e), "type": "unexpected_error", "code": "internal_error"}}


async def call_qwen3_vl(image_content: str) -> dict:
    headers = {
        "Authorization": f"Bearer {REGOLO_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": VLLM_MODELS["qwen3-vl-32b"]["id"],
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_content}}
            ]}
        ]
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                REGOLO_API_URL,
                json=payload,
                headers=headers,
                timeout=120.0
            )
            response.raise_for_status()
            return response.json()
    except httpx.ReadTimeout:
        return {"error": {"message": "Vision model timeout", "type": "timeout", "code": "timeout"}}
    except Exception as e:
        return {"error": {"message": str(e), "type": "unexpected_error", "code": "internal_error"}}


async def process_image_with_fallback(image_content: str) -> dict:
    deepseek_result = await call_deepseek_ocr(image_content)
    deepseek_content = deepseek_result.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    if not deepseek_content or len(deepseek_content.strip()) < 10:
        return await call_qwen3_vl(image_content)
    
    return deepseek_result


def mask_response(response: dict) -> dict:
    """Mask the model name to 'brick' in the response."""
    if isinstance(response, dict):
        response["model"] = "brick"
    return response


def extract_text_from_content(content) -> str:
    """Estrae il testo da content sia se stringa che se array di parts."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    texts.append(text)
        return " ".join(texts)
    return ""


def extract_image_url_from_content(content) -> str:
    """Estrae l'URL dell'immagine da content array (formato OpenAI vision)."""
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                image_url_obj = block.get("image_url", {})
                if isinstance(image_url_obj, dict):
                    return image_url_obj.get("url", "")
                elif isinstance(image_url_obj, str):
                    return image_url_obj
    return ""


def extract_audio_url_from_content(content) -> str:
    """Estrae l'URL audio da content array."""
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type == "audio":
                    return block.get("audio", "") or block.get("url", "")
                elif block_type == "input_audio":
                    audio_url = block.get("audio_url", {})
                    if isinstance(audio_url, dict):
                        return audio_url.get("url", "")
                    elif isinstance(audio_url, str):
                        return audio_url
    return ""


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    requested_model = body.get("model", "")
    
    # Check if this is a routed request from vLLM SR (has x-selected-model header)
    selected_model = request.headers.get("x-selected-model")
    if selected_model:
        # This is a routed request - go directly to Regolo with the selected model
        # Normalize messages to ensure content is string format (not array)
        normalized_messages = normalize_messages_for_vllm_sr(messages)
        llm_result = await call_regolo_llm(normalized_messages, selected_model)
        return JSONResponse(content=mask_response(llm_result))
    
    # Only accept "brick" model for client requests
    if requested_model != "brick":
        return JSONResponse(
            status_code=400,
            content={"error": f"Model '{requested_model}' not supported. Use 'brick'."}
        )
    
    filter_result = filter_function(messages)
    
    # CASO 1: Audio-only
    if filter_result["audio"] and not filter_result["image"] and not filter_result["text"]:
        for msg in messages:
            content = msg.get("content", "")
            audio_url = extract_audio_url_from_content(content)
            if audio_url:
                whisper_result = await call_faster_whisper(audio_url)
                return JSONResponse(content=mask_response(whisper_result))
    
    # CASO 4: Image + Audio (no text)
    if filter_result["image"] and filter_result["audio"] and not filter_result["text"]:
        audio_transcription = ""
        image_content_str = ""
        
        for msg in messages:
            content = msg.get("content", "")
            audio_url = extract_audio_url_from_content(content)
            if audio_url:
                whisper_result = await call_faster_whisper(audio_url)
                audio_transcription = whisper_result.get("choices", [{}])[0].get("message", {}).get("content", "")
            image_url = extract_image_url_from_content(content)
            if image_url:
                image_content_str = image_url
        
        # Process image
        image_result = await process_image_with_fallback(image_content_str)
        
        # Combine and route via vLLM SR
        combined_messages = [
            {"role": "user", "content": f"Audio transcription: {audio_transcription}\n\nImage analysis: {image_result}"}
        ]
        
        llm_result = await call_vllm_sr(combined_messages)
        return JSONResponse(content=mask_response(llm_result))
    
    if filter_result["text"] and filter_result["audio"] and not filter_result["image"]:
        audio_transcription = ""
        text_content = ""
        
        for msg in messages:
            content = msg.get("content", "")
            audio_url = extract_audio_url_from_content(content)
            if audio_url:
                whisper_result = await call_faster_whisper(audio_url)
                audio_transcription = whisper_result.get("choices", [{}])[0].get("message", {}).get("content", "")
            text_from_content = extract_text_from_content(content)
            if text_from_content:
                text_content = text_from_content
        
        combined_messages = [
            {"role": "user", "content": f"Trascrizione audio: {audio_transcription}\n\nTesto originale: {text_content}"}
        ]
        
        llm_result = await call_vllm_sr(combined_messages)
        return JSONResponse(content=mask_response(llm_result))
    
    # CASO 6: Text + Image + Audio
    if filter_result["text"] and filter_result["image"] and filter_result["audio"]:
        audio_transcription = ""
        image_content_str = ""
        text_content = ""
        
        for msg in messages:
            content = msg.get("content", "")
            audio_url = extract_audio_url_from_content(content)
            if audio_url:
                whisper_result = await call_faster_whisper(audio_url)
                audio_transcription = whisper_result.get("choices", [{}])[0].get("message", {}).get("content", "")
            image_url = extract_image_url_from_content(content)
            if image_url:
                image_content_str = image_url
            text_from_content = extract_text_from_content(content)
            if text_from_content:
                text_content = text_from_content
        
        image_result = await process_image_with_fallback(image_content_str)
        
        combined_messages = [
            {"role": "user", "content": f"Trascrizione audio: {audio_transcription}\n\nImmagine result: {image_result}\n\nTesto originale: {text_content}"}
        ]
        
        llm_result = await call_vllm_sr(combined_messages)
        return JSONResponse(content=mask_response(llm_result))
    
    # CASO 7: Text-only
    if filter_result["text"] and not filter_result["image"] and not filter_result["audio"]:
        normalized_messages = normalize_messages_for_vllm_sr(messages)
        llm_result = await call_vllm_sr(normalized_messages)
        return JSONResponse(content=mask_response(llm_result))
    
    # CASO 3: Image + Text
    if filter_result["image"] and filter_result["text"] and not filter_result["audio"]:
        for msg in messages:
            content = msg.get("content", "")
            image_url = extract_image_url_from_content(content)
            if image_url:
                image_result = await call_qwen3_vl(image_url)
                return JSONResponse(content=mask_response(image_result))
    
    # CASO 2: Image-only
    if filter_result["image"] and not filter_result["text"] and not filter_result["audio"]:
        for msg in messages:
            content = msg.get("content", "")
            image_url = extract_image_url_from_content(content)
            if image_url:
                image_result = await process_image_with_fallback(image_url)
                return JSONResponse(content=mask_response(image_result))
    
    # Fallback
    return JSONResponse(content={
        "error": "Unable to process request with current modality combination",
        "filter": filter_result
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
