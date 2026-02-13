from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import httpx
import os
import json
import logging
import asyncio

from vLLMsr_model import VLLM_MODELS, DEFAULT_MODEL, DEFAULT_VLLM_MODEL, BRICK_MODEL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

vllm_client: httpx.AsyncClient | None = None
regolo_stream_client: httpx.AsyncClient | None = None
regolo_client: httpx.AsyncClient | None = None

app = FastAPI()

@app.on_event("startup")
async def startup():
    global vllm_client, regolo_stream_client, regolo_client
    vllm_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=30.0)
    )
    regolo_stream_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=60.0)
    )
    regolo_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=60.0)
    )

@app.on_event("shutdown")
async def shutdown():
    global vllm_client, regolo_stream_client, regolo_client
    if vllm_client:
        await vllm_client.aclose()
    if regolo_stream_client:
        await regolo_stream_client.aclose()
    if regolo_client:
        await regolo_client.aclose()

VLLM_SR_URL = os.getenv("VLLM_SR_URL", "http://vllm-sr:8888/v1/chat/completions")
REGOLO_API_URL = "https://api.regolo.ai/v1/chat/completions"
REGOLO_API_KEY = os.getenv("REGOLO_API_KEY", "")


def normalize_messages(messages: list) -> list:
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


def detect_modality(messages: list) -> dict:
    has_image = False
    has_text = False
    has_audio = False
    
    for msg in messages:
        content = msg.get("content", "")
        
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
        
        elif isinstance(content, str) and content.strip():
            has_text = True
    
    return {
        "image": has_image,
        "text": has_text,
        "audio": has_audio
    }


async def call_vllm_sr(messages: list) -> dict:
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": DEFAULT_VLLM_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1,
        "stream": False
    }
    
    try:
        logger.info(f"[vLLM-SR] Sending request to {VLLM_SR_URL}")
        response = await vllm_client.post(VLLM_SR_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        selected_model = response.headers.get("x-vsr-selected-model", "")
        decision = response.headers.get("x-vsr-selected-decision", "")
        
        logger.info(f"[vLLM-SR] Decision: {decision}, Selected model: {selected_model}")
        
        return result
    except Exception as e:
        logger.error(f"[vLLM-SR] Error: {str(e)}")
        return {"error": {"message": str(e)}}


async def stream_llm_response(messages: list, model: str):
    headers = {
        "Authorization": f"Bearer {REGOLO_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "stream": True
    }

    try:
        async with regolo_stream_client.stream("POST", REGOLO_API_URL, json=payload, headers=headers) as r:
            r.raise_for_status()
            async for b in r.aiter_bytes():
                if b:
                    text = b.decode('utf-8', errors='replace')
                    for line in text.split('\n'):
                        line = line.strip()
                        if line:
                            if line.startswith("data: "):
                                line = line[6:]
                            yield f"data: {line}\n\n".encode('utf-8')
    except asyncio.CancelledError:
        return
    except Exception as e:
        logger.error(f"[LLM] Streaming error: {str(e)}")
        err = {"error": {"message": str(e)}}
        yield (f"data: {json.dumps(err)}\n\n").encode("utf-8")


async def call_llm(messages: list, model: str) -> dict:
    headers = {
        "Authorization": f"Bearer {REGOLO_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "stream": False
    }

    try:
        r = await regolo_client.post(REGOLO_API_URL, json=payload, headers=headers)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"[LLM] Error: {str(e)}")
        return {"error": {"message": str(e)}}


async def transcribe_audio(audio_content: str) -> dict:
    import base64
    import io
    
    audio_data = audio_content
    if audio_content.startswith("data:"):
        parts = audio_content.split(",")
        if len(parts) > 1:
            audio_data = parts[1]
    
    try:
        audio_bytes = base64.b64decode(audio_data)
    except Exception as e:
        return {"error": {"message": f"Invalid base64: {str(e)}"}}
    
    headers = {"Authorization": f"Bearer {REGOLO_API_KEY}"}
    audio_file = io.BytesIO(audio_bytes)
    files = {"file": ("audio.webm", audio_file, "application/octet-stream")}
    data = {"model": "faster-whisper-large-v3"}
    
    async with httpx.AsyncClient() as client:
        whisper_result = await client.post(
            "https://api.regolo.ai/v1/audio/transcriptions",
            files=files, data=data, headers=headers, timeout=120.0
        )
        response = whisper_result
        response.raise_for_status()
        result = response.json()
        return result


async def process_image(image_content: str) -> dict:
    headers = {
        "Authorization": f"Bearer {REGOLO_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-ocr",
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_content}}]}]
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(REGOLO_API_URL, json=payload, headers=headers, timeout=120.0)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"error": {"message": str(e)}}


def extract_text(content) -> str:
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


def extract_image_url(content) -> str:
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                image_url_obj = block.get("image_url", {})
                if isinstance(image_url_obj, dict):
                    return image_url_obj.get("url", "")
                elif isinstance(image_url_obj, str):
                    return image_url_obj
    return ""


def extract_audio_url(content) -> str:
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


def get_modality_string(filter_result: dict) -> str:
    mods = []
    if filter_result.get("text"):
        mods.append("text")
    if filter_result.get("image"):
        mods.append("image")
    if filter_result.get("audio"):
        mods.append("audio")
    return "+".join(mods) if mods else "unknown"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
        messages = body.get("messages", [])
        requested_model = body.get("model", "")
        is_stream = body.get("stream", False)
        
        logger.info(f"[CHAT] Model: {requested_model}, Stream: {is_stream}")
        
        filter_result = detect_modality(messages)
        modality = get_modality_string(filter_result)
        logger.info(f"[CHAT] Modality: {modality}")
        
        normalized_messages = normalize_messages(messages)
        
        selected_model = request.headers.get("x-selected-model")
        if selected_model:
            logger.info(f"[CHAT] Using selected model: {selected_model}")
            if is_stream:
                return StreamingResponse(
                    stream_llm_response(normalized_messages, selected_model),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                )
            else:
                result = await call_llm(normalized_messages, selected_model)
                return JSONResponse(content=result)
        
        if requested_model != "brick":
            return JSONResponse(status_code=400, content={"error": f"Model '{requested_model}' not supported. Use 'brick'."})
        
        if filter_result["audio"] and not filter_result["image"] and not filter_result["text"]:
            logger.info("[CHAT] Audio-only request")
            for msg in messages:
                content = msg.get("content", "")
                audio_url = extract_audio_url(content)
                if audio_url:
                    whisper_result = await transcribe_audio(audio_url)
                    if whisper_result.get("error"):
                        return JSONResponse(content=whisper_result)
                    
                    transcription = whisper_result.get("text", "")
                    if not transcription:
                        choices = whisper_result.get("choices", [])
                        if choices:
                            transcription = choices[0].get("message", {}).get("content", "")
                    
                    if not transcription:
                        return JSONResponse(content={"error": "Unable to transcribe audio"})
                    
                    vllm_result = await call_vllm_sr([{"role": "user", "content": transcription}])
                    if vllm_result.get("error"):
                        return JSONResponse(content=vllm_result)
                    
                    selected = vllm_result.get("model", "")
                    logger.info(f"[CHAT] Audio routed to: {selected}")
                    
                    return StreamingResponse(
                        stream_llm_response([{"role": "user", "content": transcription}], selected),
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                    )
        
        if filter_result["image"] and filter_result["audio"] and not filter_result["text"]:
            logger.info("[CHAT] Image + Audio request")
            audio_transcription = ""
            image_url = ""
            for msg in messages:
                content = msg.get("content", "")
                audio_result = await transcribe_audio(extract_audio_url(content))
                audio_transcription = audio_result.get("text", "")
                if not audio_transcription:
                    choices = audio_result.get("choices", [])
                    if choices:
                        audio_transcription = choices[0].get("message", {}).get("content", "")
                image_url = extract_image_url(content)
            
            image_result = await process_image(image_url)
            ocr_text = image_result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            combined = f"Audio: {audio_transcription}\n\nImage: {ocr_text}"
            vllm_result = await call_vllm_sr([{"role": "user", "content": combined}])
            return JSONResponse(content=vllm_result)
        
        if filter_result["text"] and filter_result["audio"] and not filter_result["image"]:
            logger.info("[CHAT] Text + Audio request")
            audio_transcription = ""
            text_content = ""
            for msg in messages:
                content = msg.get("content", "")
                audio_result = await transcribe_audio(extract_audio_url(content))
                audio_transcription = audio_result.get("text", "")
                if not audio_transcription:
                    choices = audio_result.get("choices", [])
                    if choices:
                        audio_transcription = choices[0].get("message", {}).get("content", "")
                text_content = extract_text(content)
            
            combined = f"Audio: {audio_transcription}\n\nText: {text_content}"
            vllm_result = await call_vllm_sr([{"role": "user", "content": combined}])
            return JSONResponse(content=vllm_result)
        
        if filter_result["text"] and filter_result["image"] and filter_result["audio"]:
            logger.info("[CHAT] Text + Image + Audio request")
            audio_transcription = ""
            image_url = ""
            text_content = ""
            for msg in messages:
                content = msg.get("content", "")
                audio_result = await transcribe_audio(extract_audio_url(content))
                audio_transcription = audio_result.get("text", "")
                if not audio_transcription:
                    choices = audio_result.get("choices", [])
                    if choices:
                        audio_transcription = choices[0].get("message", {}).get("content", "")
                image_url = extract_image_url(content)
                text_content = extract_text(content)
            
            image_result = await process_image(image_url)
            ocr_text = image_result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            combined = f"Audio: {audio_transcription}\n\nImage: {ocr_text}\n\nText: {text_content}"
            vllm_result = await call_vllm_sr([{"role": "user", "content": combined}])
            return JSONResponse(content=vllm_result)
        
        if filter_result["text"] and not filter_result["image"] and not filter_result["audio"]:
            logger.info("[CHAT] Text-only request")
            vllm_result = await call_vllm_sr(normalized_messages)
            
            if vllm_result.get("error"):
                return JSONResponse(content=vllm_result)
            
            selected_model = vllm_result.get("model", "")
            logger.info(f"[CHAT] Text routed to: {selected_model}")
            
            return StreamingResponse(
                stream_llm_response(normalized_messages, selected_model),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        
        if filter_result["image"] and filter_result["text"] and not filter_result["audio"]:
            logger.info("[CHAT] Image + Text request")
            for msg in messages:
                content = msg.get("content", "")
                image_url = extract_image_url(content)
                if image_url:
                    if is_stream:
                        return StreamingResponse(
                            stream_llm_response([{"role": "user", "content": f"Analyze this image: {image_url}"}], "qwen3-vl-32b"),
                            media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                        )
                    else:
                        result = await process_image(image_url)
                        return JSONResponse(content=result)
        
        if filter_result["image"] and not filter_result["text"] and not filter_result["audio"]:
            logger.info("[CHAT] Image-only request")
            for msg in messages:
                content = msg.get("content", "")
                image_url = extract_image_url(content)
                if image_url:
                    ocr_result = await process_image(image_url)
                    ocr_text = ocr_result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    if ocr_text and len(ocr_text.strip()) > 10:
                        if is_stream:
                            vllm_result = await call_vllm_sr([{"role": "user", "content": ocr_text}])
                            if vllm_result.get("error"):
                                return JSONResponse(content=vllm_result)
                            selected = vllm_result.get("model", "")
                            logger.info(f"[CHAT] OCR routed to: {selected}")
                            return StreamingResponse(
                                stream_llm_response([{"role": "user", "content": ocr_text}], selected),
                                media_type="text/event-stream",
                                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                            )
                        else:
                            vllm_result = await call_vllm_sr([{"role": "user", "content": ocr_text}])
                            return JSONResponse(content=vllm_result)
                    
                    if is_stream:
                        return StreamingResponse(
                            stream_llm_response([{"role": "user", "content": f"Analyze: {image_url}"}], "qwen3-vl-32b"),
                            media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                        )
                    else:
                        result = await process_image(image_url)
                        return JSONResponse(content=result)
        
        return JSONResponse(content={"error": "Unable to process request"})
    
    except Exception as e:
        logger.exception(f"[CHAT] Error")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/v1/models")
async def list_brick_model():
    """
    Restituisce l'elenco dei modelli disponibili.
    In questa implementazione viene restituito **solo** il modello virtuale `brick`,
    nel formato compatibile con l'API OpenAI.
    """
    model_info = BRICK_MODEL["brick"]
    payload = {
        "object": "list",
        "data": [
            {
                "id": model_info["id"],
                "object": "model",
                "created": 0,
                "owned_by": "regolo",
                "type": model_info.get("type", "virtual"),
                "description": model_info.get("description", "")
            }
        ]
    }
    return JSONResponse(content=payload)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
