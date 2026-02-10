from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import httpx
import os
import json
import logging
import asyncio
import time
import uuid

from vLLMsr_model import VLLM_MODELS, DEFAULT_MODEL, DEFAULT_VLLM_MODEL, BRICK_MODEL
from monitor.time_logger import TimeLogger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

monitor_logger = TimeLogger(os.path.join(os.path.dirname(__file__), 'monitor', 'timelog.db'))

# Global clients for connection reuse (initialized in startup event)
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
    logger.info(f"call_vllm_sr: Starting vLLM SR call")
    logger.info(f"call_vllm_sr: Messages count = {len(messages)}")
    logger.info(f"call_vllm_sr: URL = {VLLM_SR_URL}")
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "model": DEFAULT_VLLM_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1,
        "stream": False
    }
    
    logger.info(f"call_vllm_sr: Payload = {json.dumps(payload, indent=2)[:300]}")
    
    try:
        logger.info("call_vllm_sr: Sending request to vLLM SR...")
        response = await vllm_client.post(
            VLLM_SR_URL,
            json=payload,
            headers=headers
        )
        logger.info(f"call_vllm_sr: Response status = {response.status_code}")
        response.raise_for_status()
        result = response.json()
        logger.info(f"call_vllm_sr: Response keys = {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        
        # Check for errors in response
        if result.get("error"):
            logger.error(f"call_vllm_sr: Error in response = {result['error']}")
        elif result.get("choices"):
            logger.info(f"call_vllm_sr: Success - {len(result['choices'])} choices received")
        
        return result
    except Exception as e:
        logger.error(f"call_vllm_sr: Exception = {str(e)}")
        return {"error": {"message": str(e), "type": "vllm_sr_error", "code": "internal_error"}}


async def call_regolo_llm_stream(messages: list, model: str):
    """Stream LLM response from Regolo API using SSE format - pass-through bytes."""
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
                    yield b
    except asyncio.CancelledError:
        return
    except Exception as e:
        err = {"error": {"message": str(e), "type": "streaming_error", "code": "internal_error"}}
        yield (f"data: {json.dumps(err)}\n\n").encode("utf-8")


async def call_regolo_llm(messages: list, model: str) -> dict:
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
    except httpx.ReadTimeout:
        return {"error": {"message": f"Request timeout for model {model}.", "type": "timeout", "code": "timeout"}}
    except httpx.HTTPStatusError as e:
        return {"error": {"message": f"HTTP error {e.response.status_code}: {str(e)}", "type": "http_error", "code": e.response.status_code}}
    except Exception as e:
        return {"error": {"message": str(e), "type": "unexpected_error", "code": "internal_error"}}


async def call_faster_whisper(audio_content: str) -> dict:
    import base64
    import io
    
    logger.info(f"call_faster_whisper: Starting transcription")
    logger.info(f"call_faster_whisper: Audio content length = {len(audio_content) if audio_content else 0}")
    
    # Estrai il contenuto base64 dal data URL
    audio_data = audio_content
    if audio_content.startswith("data:"):
        # Formato: data:audio/webm;base64,...
        parts = audio_content.split(",")
        if len(parts) > 1:
            audio_data = parts[1]
            logger.info(f"call_faster_whisper: Extracted base64 from data URL")
    
    # Decodifica base64 in bytes
    try:
        audio_bytes = base64.b64decode(audio_data)
        logger.info(f"call_faster_whisper: Decoded audio bytes = {len(audio_bytes)}")
    except Exception as e:
        logger.error(f"call_faster_whisper: Failed to decode base64: {str(e)}")
        return {"error": {"message": f"Invalid base64 audio data: {str(e)}", "type": "decode_error", "code": "invalid_audio"}}
    
    # Headers: solo Authorization, Content-Type lo gestisce httpx per multipart
    headers = {
        "Authorization": f"Bearer {REGOLO_API_KEY}"
    }
    
    model_id = VLLM_MODELS["faster-whisper-large-v3"]["id"]
    logger.info(f"call_faster_whisper: Using model = {model_id}")
    
    # Prepara il file per multipart/form-data (come nell'esempio funzionante)
    audio_file = io.BytesIO(audio_bytes)
    files = {
        "file": ("audio.webm", audio_file, "application/octet-stream")
    }
    data = {
        "model": model_id
    }
    
    try:
        async with httpx.AsyncClient() as client:
            logger.info("call_faster_whisper: Sending request to Regolo API (multipart/form-data)...")
            response = await client.post(
                "https://api.regolo.ai/v1/audio/transcriptions",
                files=files,
                data=data,
                headers=headers,
                timeout=120.0
            )
            logger.info(f"call_faster_whisper: Response status = {response.status_code}")
            response.raise_for_status()
            
            # La risposta potrebbe essere text o JSON
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                result = response.json()
                logger.info(f"call_faster_whisper: JSON response, keys = {result.keys()}")
                return result
            else:
                # Risposta in formato text
                transcript_text = response.text
                logger.info(f"call_faster_whisper: Text response = {transcript_text[:100]}")
                # Converte in formato OpenAI-like
                return {
                    "text": transcript_text,
                    "choices": [{
                        "message": {
                            "content": transcript_text
                        }
                    }]
                }
    except httpx.ReadTimeout:
        logger.error("call_faster_whisper: Timeout error")
        return {"error": {"message": "Whisper transcription timeout", "type": "timeout", "code": "timeout"}}
    except Exception as e:
        logger.error(f"call_faster_whisper: Error = {str(e)}")
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


async def call_qwen3_vl_stream(image_content: str):
    """Stream Vision model response from Regolo API using SSE format - pass-through bytes."""
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
        ],
        "stream": True
    }

    try:
        async with regolo_stream_client.stream("POST", REGOLO_API_URL, json=payload, headers=headers) as r:
            r.raise_for_status()
            async for b in r.aiter_bytes():
                if b:
                    yield b
    except asyncio.CancelledError:
        return
    except Exception as e:
        err = {"error": {"message": str(e), "type": "streaming_error", "code": "internal_error"}}
        yield (f"data: {json.dumps(err)}\n\n").encode("utf-8")


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


def extract_transcription_from_whisper(whisper_result: dict) -> str:
    """Estrae la trascrizione dalla risposta di Whisper, gestendo entrambi i formati."""
    # Prima prova il campo 'text' (formato diretto)
    transcription = whisper_result.get("text", "")
    if transcription:
        return transcription
    
    # Fallback al formato choices/message/content (formato OpenAI-like)
    choices = whisper_result.get("choices", [])
    if choices and len(choices) > 0:
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if content:
            return content
    
    return ""


def determine_modality(filter_result: dict) -> str:
    """Converte il risultato del filtro in stringa di modalità."""
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
    request_id = None
    try:
        body = await request.json()
        messages = body.get("messages", [])
        requested_model = body.get("model", "")
        is_stream = body.get("stream", False)
        request_size = len(json.dumps(body))

        request_id = monitor_logger.start_request(
            request_id=f"req_{int(time.time())}_{uuid.uuid4().hex[:6]}",
            modality='unknown',
            model=requested_model,
            stream=is_stream,
            request_size_bytes=request_size
        )

        logger.info(f"=== RICHIESTA RICEVUTA === Request ID: {request_id}")
        logger.info(f"Model: {requested_model}, Stream: {is_stream}")

        monitor_logger.log_phase_start(request_id, 'modality_detection')
        filter_result = filter_function(messages)
        modality = determine_modality(filter_result)
        monitor_logger.update_metadata(request_id, modality=modality)
        monitor_logger.log_phase_end(request_id, 'modality_detection')
        logger.info(f"Filter result: {filter_result}, Modality: {modality}")

        selected_model = request.headers.get("x-selected-model")
        if selected_model:
            logger.warning("Unexpected x-selected-model header in Mode A")
            original_is_stream = request.headers.get("x-original-stream", "false").lower() == "true"
            is_stream = original_is_stream
            normalized_messages = normalize_messages_for_vllm_sr(messages)

            monitor_logger.log_phase_start(request_id, 'regolo_response')
            if is_stream:
                monitor_logger.log_phase_end(request_id, 'regolo_response')
                monitor_logger.end_request(request_id, success=True)
                return StreamingResponse(
                    call_regolo_llm_stream(normalized_messages, selected_model),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
                )
            else:
                llm_result = await call_regolo_llm(normalized_messages, selected_model)
                monitor_logger.log_phase_end(request_id, 'regolo_response')
                monitor_logger.end_request(request_id, success=True, response_size_bytes=len(json.dumps(llm_result)))
                return JSONResponse(content=llm_result)

        if requested_model != "brick":
            monitor_logger.end_request(request_id, success=False, error_type="invalid_model", error_message=f"Model '{requested_model}' not supported")
            return JSONResponse(status_code=400, content={"error": f"Model '{requested_model}' not supported. Use 'brick'."})

        if filter_result["audio"] and not filter_result["image"] and not filter_result["text"]:
            logger.info("=== CASO 1: Audio-only ===")
            for msg in messages:
                content = msg.get("content", "")
                audio_url = extract_audio_url_from_content(content)
                if audio_url:
                    monitor_logger.log_phase_start(request_id, 'audio_transcription')
                    whisper_result = await call_faster_whisper(audio_url)
                    if whisper_result.get("error"):
                        monitor_logger.log_phase_end(request_id, 'audio_transcription', success=False, error_type=whisper_result.get("error", {}).get("type"), error_message=whisper_result.get("error", {}).get("message"))
                        monitor_logger.end_request(request_id, success=False, error_type=whisper_result.get("error", {}).get("type"), error_message=whisper_result.get("error", {}).get("message"))
                        return JSONResponse(content=whisper_result)
                    transcription = extract_transcription_from_whisper(whisper_result)
                    monitor_logger.log_phase_end(request_id, 'audio_transcription')

                    if not transcription:
                        monitor_logger.end_request(request_id, success=False, error_type="transcription_failed", error_message="No transcription found")
                        return JSONResponse(content={"error": "Unable to transcribe audio content", "model": "brick"})

                    transcription_messages = [{"role": "user", "content": transcription}]
                    monitor_logger.log_phase_start(request_id, 'vllm_routing')
                    vllm_result = await call_vllm_sr(transcription_messages)
                    if vllm_result.get("error"):
                        monitor_logger.log_phase_end(request_id, 'vllm_routing', success=False)
                        monitor_logger.end_request(request_id, success=False)
                        return JSONResponse(content=vllm_result)

                    selected = vllm_result.get("model", "")
                    monitor_logger.log_phase_end(request_id, 'vllm_routing')
                    monitor_logger.log_phase_start(request_id, 'regolo_response')
                    monitor_logger.log_phase_end(request_id, 'regolo_response')
                    monitor_logger.end_request(request_id, success=True)
                    return StreamingResponse(
                        call_regolo_llm_stream(transcription_messages, selected),
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
                    )

        if filter_result["image"] and filter_result["audio"] and not filter_result["text"]:
            logger.info("=== CASO 4: Image + Audio ===")
            audio_transcription = ""
            image_content_str = ""
            for msg in messages:
                content = msg.get("content", "")
                audio_url = extract_audio_url_from_content(content)
                if audio_url:
                    monitor_logger.log_phase_start(request_id, 'audio_transcription')
                    whisper_result = await call_faster_whisper(audio_url)
                    audio_transcription = extract_transcription_from_whisper(whisper_result)
                    monitor_logger.log_phase_end(request_id, 'audio_transcription')
                image_url = extract_image_url_from_content(content)
                if image_url:
                    image_content_str = image_url

            monitor_logger.log_phase_start(request_id, 'image_processing')
            image_result = await process_image_with_fallback(image_content_str)
            monitor_logger.log_phase_end(request_id, 'image_processing')

            combined_messages = [{"role": "user", "content": f"Audio transcription: {audio_transcription}\n\nImage analysis: {image_result}"}]
            monitor_logger.log_phase_start(request_id, 'vllm_routing')
            vllm_result = await call_vllm_sr(combined_messages)
            monitor_logger.log_phase_end(request_id, 'vllm_routing')
            monitor_logger.end_request(request_id, success=True, response_size_bytes=len(json.dumps(vllm_result)))
            return JSONResponse(content=vllm_result)

        if filter_result["text"] and filter_result["audio"] and not filter_result["image"]:
            logger.info("=== CASO 5: Text + Audio ===")
            audio_transcription = ""
            text_content = ""
            for msg in messages:
                content = msg.get("content", "")
                audio_url = extract_audio_url_from_content(content)
                if audio_url:
                    monitor_logger.log_phase_start(request_id, 'audio_transcription')
                    whisper_result = await call_faster_whisper(audio_url)
                    audio_transcription = extract_transcription_from_whisper(whisper_result)
                    monitor_logger.log_phase_end(request_id, 'audio_transcription')
                text_from_content = extract_text_from_content(content)
                if text_from_content:
                    text_content = text_from_content

            combined_messages = [{"role": "user", "content": f"Trascrizione audio: {audio_transcription}\n\nTesto originale: {text_content}"}]
            monitor_logger.log_phase_start(request_id, 'vllm_routing')
            vllm_result = await call_vllm_sr(combined_messages)
            monitor_logger.log_phase_end(request_id, 'vllm_routing')
            monitor_logger.end_request(request_id, success=True, response_size_bytes=len(json.dumps(vllm_result)))
            return JSONResponse(content=vllm_result)

        if filter_result["text"] and filter_result["image"] and filter_result["audio"]:
            logger.info("=== CASO 6: Text + Image + Audio ===")
            audio_transcription = ""
            image_content_str = ""
            text_content = ""
            for msg in messages:
                content = msg.get("content", "")
                audio_url = extract_audio_url_from_content(content)
                if audio_url:
                    monitor_logger.log_phase_start(request_id, 'audio_transcription')
                    whisper_result = await call_faster_whisper(audio_url)
                    audio_transcription = extract_transcription_from_whisper(whisper_result)
                    monitor_logger.log_phase_end(request_id, 'audio_transcription')
                image_url = extract_image_url_from_content(content)
                if image_url:
                    image_content_str = image_url
                text_from_content = extract_text_from_content(content)
                if text_from_content:
                    text_content = text_from_content

            monitor_logger.log_phase_start(request_id, 'image_processing')
            image_result = await process_image_with_fallback(image_content_str)
            monitor_logger.log_phase_end(request_id, 'image_processing')

            combined_messages = [{"role": "user", "content": f"Trascrizione audio: {audio_transcription}\n\nImmagine result: {image_result}\n\nTesto originale: {text_content}"}]
            monitor_logger.log_phase_start(request_id, 'vllm_routing')
            vllm_result = await call_vllm_sr(combined_messages)
            monitor_logger.log_phase_end(request_id, 'vllm_routing')
            monitor_logger.end_request(request_id, success=True, response_size_bytes=len(json.dumps(vllm_result)))
            return JSONResponse(content=vllm_result)

        if filter_result["text"] and not filter_result["image"] and not filter_result["audio"]:
            logger.info("=== CASO 7: Text-only ===")
            normalized_messages = normalize_messages_for_vllm_sr(messages)
            monitor_logger.log_phase_start(request_id, 'vllm_routing')
            vllm_result = await call_vllm_sr(normalized_messages)
            if vllm_result.get("error"):
                monitor_logger.log_phase_end(request_id, 'vllm_routing', success=False)
                monitor_logger.end_request(request_id, success=False)
                return JSONResponse(content=vllm_result)

            selected_model = vllm_result.get("model", "")
            monitor_logger.log_phase_end(request_id, 'vllm_routing')
            monitor_logger.log_phase_start(request_id, 'regolo_response')
            monitor_logger.log_phase_end(request_id, 'regolo_response')
            monitor_logger.end_request(request_id, success=True)
            return StreamingResponse(
                call_regolo_llm_stream(normalized_messages, selected_model),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
            )

        if filter_result["image"] and filter_result["text"] and not filter_result["audio"]:
            for msg in messages:
                content = msg.get("content", "")
                image_url = extract_image_url_from_content(content)
                if image_url:
                    monitor_logger.log_phase_start(request_id, 'image_processing')
                    if is_stream:
                        monitor_logger.log_phase_end(request_id, 'image_processing')
                        monitor_logger.end_request(request_id, success=True)
                        return StreamingResponse(
                            call_qwen3_vl_stream(image_url),
                            media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
                        )
                    else:
                        image_result = await call_qwen3_vl(image_url)
                        monitor_logger.log_phase_end(request_id, 'image_processing')
                        monitor_logger.end_request(request_id, success=True, response_size_bytes=len(json.dumps(image_result)))
                        return JSONResponse(content=image_result)

        if filter_result["image"] and not filter_result["text"] and not filter_result["audio"]:
            for msg in messages:
                content = msg.get("content", "")
                image_url = extract_image_url_from_content(content)
                if image_url:
                    monitor_logger.log_phase_start(request_id, 'image_processing')
                    ocr_result = await call_deepseek_ocr(image_url)
                    ocr_text = ocr_result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if ocr_text and len(ocr_text.strip()) > 10:
                        ocr_messages = [{"role": "user", "content": ocr_text}]
                        if is_stream:
                            vllm_result = await call_vllm_sr(ocr_messages)
                            if vllm_result.get("error"):
                                monitor_logger.log_phase_end(request_id, 'image_processing')
                                monitor_logger.end_request(request_id, success=False)
                                return JSONResponse(content=vllm_result)
                            selected = vllm_result.get("model", "")
                            monitor_logger.log_phase_end(request_id, 'image_processing')
                            monitor_logger.log_phase_start(request_id, 'regolo_response')
                            monitor_logger.log_phase_end(request_id, 'regolo_response')
                            monitor_logger.end_request(request_id, success=True)
                            return StreamingResponse(
                                call_regolo_llm_stream(ocr_messages, selected),
                                media_type="text/event-stream",
                                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
                            )
                        else:
                            llm_result = await call_vllm_sr(ocr_messages)
                            monitor_logger.log_phase_end(request_id, 'image_processing')
                            monitor_logger.end_request(request_id, success=True, response_size_bytes=len(json.dumps(llm_result)))
                            return JSONResponse(content=llm_result)

                    if is_stream:
                        monitor_logger.log_phase_end(request_id, 'image_processing')
                        monitor_logger.end_request(request_id, success=True)
                        return StreamingResponse(
                            call_qwen3_vl_stream(image_url),
                            media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
                        )
                    else:
                        image_result = await call_qwen3_vl(image_url)
                        monitor_logger.log_phase_end(request_id, 'image_processing')
                        monitor_logger.end_request(request_id, success=True, response_size_bytes=len(json.dumps(image_result)))
                        return JSONResponse(content=image_result)

        monitor_logger.end_request(request_id, success=False, error_type="unsupported_modality", error_message="Unable to process request")
        return JSONResponse(content={"error": "Unable to process request with current modality combination", "filter": filter_result})

    except Exception as e:
        logger.exception(f"Error processing request {request_id}")
        if request_id:
            monitor_logger.end_request(request_id, success=False, error_type=type(e).__name__, error_message=str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
