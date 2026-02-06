from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import httpx
import os
import json

from vLLMsr_model import VLLM_MODELS, DEFAULT_MODEL, DEFAULT_VLLM_MODEL

app = FastAPI()

VLLM_SR_URL = os.getenv("VLLM_SR_URL", "http://vllm-sr:8888/v1/chat/completions")
REGOLO_API_URL = "https://api.regolo.ai/v1/chat/completions"
REGOLO_API_KEY = os.getenv("REGOLO_API_KEY", "")


def filter_function(messages: list) -> dict:
    has_image = False
    has_text = False
    has_audio = False
    
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            content_lower = content.lower()
            if "image" in content_lower:
                has_image = True
            if "text" in content_lower:
                has_text = True
            if "audio" in content_lower:
                has_audio = True
    
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


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    
    filter_result = filter_function(messages)
    
    if filter_result["audio"] and not filter_result["image"] and not filter_result["text"]:
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and "audio" in content.lower():
                whisper_result = await call_faster_whisper(content)
                return JSONResponse(content=whisper_result)
    
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
        
        return JSONResponse(content={
            "routing": routing_decision,
            "model_used": model,
            "response": llm_result
        })
    
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
        
        return JSONResponse(content={
            "routing": routing_decision,
            "model_used": model,
            "response": llm_result
        })
    
    if filter_result["text"] and not filter_result["image"] and not filter_result["audio"]:
        routing_decision = await call_vllm_sr(messages)
        model = routing_decision.get("routing", {}).get("selected_model", DEFAULT_MODEL)
        llm_result = await call_regolo_llm(messages, model)
        
        return JSONResponse(content={
            "routing": routing_decision,
            "model_used": model,
            "response": llm_result
        })
    
    if filter_result["image"] and filter_result["text"] and not filter_result["audio"]:
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and "image" in content.lower():
                image_result = await call_qwen3_vl(content)
                return JSONResponse(content=image_result)
    
    if filter_result["image"] and not filter_result["text"] and not filter_result["audio"]:
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and "image" in content.lower():
                image_result = await process_image_with_fallback(content)
                return JSONResponse(content=image_result)
    
    return JSONResponse(content={
        "filter": filter_result,
        "note": "No matching modality pattern"
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
