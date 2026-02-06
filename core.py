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
    """Call vLLM Semantic Router to get routing decision with model info."""
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
            timeout=30.0
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
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            REGOLO_API_URL,
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        return response.json()


async def call_faster_whisper(audio_content: str) -> dict:
    headers = {
        "Authorization": f"Bearer {REGOLO_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": VLLM_MODELS["whisper-large-v3"]["id"],
        "file": audio_content or ""
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.regolo.ai/v1/audio/transcriptions",
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        return response.json()


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
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            REGOLO_API_URL,
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        return response.json()


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
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            REGOLO_API_URL,
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        return response.json()


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


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    requested_model = body.get("model", "")
    
    # Only accept "brick" model
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
            if isinstance(content, str) and "audio" in content.lower():
                whisper_result = await call_faster_whisper(content)
                return JSONResponse(content=mask_response(whisper_result))
    
    # CASO 4: Image + Audio (no text)
    if filter_result["image"] and filter_result["audio"] and not filter_result["text"]:
        audio_transcription = ""
        image_content_str = ""
        
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                content_lower = content.lower()
                if "audio" in content_lower:
                    whisper_result = await call_faster_whisper(content)
                    audio_transcription = whisper_result.get("choices", [{}])[0].get("message", {}).get("content", "")
                if "image" in content_lower:
                    image_content_str = content
        
        # Process image
        image_result = await process_image_with_fallback(image_content_str)
        
        # Combine and route via vLLM SR
        combined_messages = [
            {"role": "user", "content": f"Audio transcription: {audio_transcription}\n\nImage analysis: {image_result}"}
        ]
        
        routing_decision = await call_vllm_sr(combined_messages)
        model = routing_decision.get("routing", {}).get("selected_model", DEFAULT_MODEL)
        llm_result = await call_regolo_llm(combined_messages, model)
        
        return JSONResponse(content=mask_response({
            "routing": routing_decision,
            "model_used": model,
            "response": llm_result
        }))
    
    if filter_result["text"] and filter_result["audio"] and not filter_result["image"]:
        audio_transcription = ""
        text_content = ""
        
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                content_lower = content.lower()
                if "audio" in content_lower:
                    whisper_result = await call_faster_whisper(content)
                    audio_transcription = whisper_result.get("choices", [{}])[0].get("message", {}).get("content", "")
                if "text" in content_lower:
                    text_content = content
        
        combined_messages = [
            {"role": "user", "content": f"Trascrizione audio: {audio_transcription}\n\nTesto originale: {text_content}"}
        ]
        
        routing_decision = await call_vllm_sr(combined_messages)
        model = routing_decision.get("routing", {}).get("selected_model", DEFAULT_MODEL)
        llm_result = await call_regolo_llm(combined_messages, model)
        
        return JSONResponse(content=mask_response({
            "routing": routing_decision,
            "model_used": model,
            "response": llm_result
        }))
    
    # CASO 6: Text + Image + Audio
    if filter_result["text"] and filter_result["image"] and filter_result["audio"]:
        audio_transcription = ""
        image_content_str = ""
        text_content = ""
        
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                content_lower = content.lower()
                if "audio" in content_lower:
                    whisper_result = await call_faster_whisper(content)
                    audio_transcription = whisper_result.get("choices", [{}])[0].get("message", {}).get("content", "")
                if "image" in content_lower:
                    image_content_str = content
                if "text" in content_lower:
                    text_content = content
        
        image_result = await process_image_with_fallback(image_content_str)
        
        combined_messages = [
            {"role": "user", "content": f"Trascrizione audio: {audio_transcription}\n\nImmagine result: {image_result}\n\nTesto originale: {text_content}"}
        ]
        
        routing_decision = await call_vllm_sr(combined_messages)
        model = routing_decision.get("routing", {}).get("selected_model", DEFAULT_MODEL)
        llm_result = await call_regolo_llm(combined_messages, model)
        
        return JSONResponse(content=mask_response({
            "routing": routing_decision,
            "model_used": model,
            "response": llm_result
        }))
    
    # CASO 7: Text-only
    if filter_result["text"] and not filter_result["image"] and not filter_result["audio"]:
        routing_decision = await call_vllm_sr(messages)
        model = routing_decision.get("routing", {}).get("selected_model", DEFAULT_MODEL)
        llm_result = await call_regolo_llm(messages, model)
        
        return JSONResponse(content=mask_response({
            "routing": routing_decision,
            "model_used": model,
            "response": llm_result
        }))
    
    # CASO 3: Image + Text
    if filter_result["image"] and filter_result["text"] and not filter_result["audio"]:
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and "image" in content.lower():
                image_result = await call_qwen3_vl(content)
                return JSONResponse(content=mask_response(image_result))
    
    # CASO 2: Image-only
    if filter_result["image"] and not filter_result["text"] and not filter_result["audio"]:
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and "image" in content.lower():
                image_result = await process_image_with_fallback(content)
                return JSONResponse(content=mask_response(image_result))
    
    # Fallback
    return JSONResponse(content={
        "error": "Unable to process request with current modality combination",
        "filter": filter_result
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
