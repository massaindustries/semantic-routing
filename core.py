from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import httpx
import os
import json
import logging

from vLLMsr_model import VLLM_MODELS, DEFAULT_MODEL, DEFAULT_VLLM_MODEL, BRICK_MODEL

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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


async def call_vllm_sr(messages: list, is_stream: bool = False) -> dict:
    """Call vLLM Semantic Router - returns final response from Regolo via routing loop."""
    logger.info(f"call_vllm_sr: Starting vLLM SR call")
    logger.info(f"call_vllm_sr: Messages count = {len(messages)}")
    logger.info(f"call_vllm_sr: URL = {VLLM_SR_URL}")
    logger.info(f"call_vllm_sr: is_stream = {is_stream}")
    
    headers = {
        "Content-Type": "application/json",
        "x-original-stream": "true" if is_stream else "false"
    }
    payload = {
        "model": DEFAULT_VLLM_MODEL,
        "messages": messages
    }
    
    logger.info(f"call_vllm_sr: Payload = {json.dumps(payload, indent=2)[:300]}")
    
    try:
        async with httpx.AsyncClient() as client:
            logger.info("call_vllm_sr: Sending request to vLLM SR...")
            response = await client.post(
                VLLM_SR_URL,
                json=payload,
                headers=headers,
                timeout=300.0  # 5 minutes timeout for routing + Regolo call
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


async def call_regolo_llm_stream(messages: list, model: str, is_stream: bool = True):
    """Stream LLM response from Regolo API using SSE format."""
    headers = {
        "Authorization": f"Bearer {REGOLO_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "stream": is_stream
    }
    
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                REGOLO_API_URL,
                json=payload,
                headers=headers,
                timeout=300.0
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    if chunk:
                        # Decode and mask model as 'brick' in each SSE chunk
                        try:
                            text = chunk.decode('utf-8')
                            if text.startswith('data:'):
                                # Parse SSE data line
                                data_content = text[5:].strip()
                                if data_content == '[DONE]':
                                    yield text
                                else:
                                    # Parse JSON and mask model
                                    data = json.loads(data_content)
                                    if 'model' in data:
                                        data['model'] = 'brick'
                                    # Re-encode to SSE format
                                    yield f"data: {json.dumps(data)}\n\n"
                            else:
                                # Non-data lines (empty lines, etc.)
                                yield text
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            # If can't decode/modify, pass through as-is
                            yield chunk.decode('utf-8', errors='replace')
                        
    except Exception as e:
        # Yield error as SSE event
        error_data = {
            "error": {
                "message": str(e),
                "type": "streaming_error",
                "code": "internal_error"
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"


async def call_regolo_llm(messages: list, model: str, is_stream: bool = False) -> dict:
    headers = {
        "Authorization": f"Bearer {REGOLO_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "stream": is_stream
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
    """Stream Vision model response from Regolo API using SSE format."""
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
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                REGOLO_API_URL,
                json=payload,
                headers=headers,
                timeout=120.0
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    if chunk:
                        # Decode and mask model as 'brick' in each SSE chunk
                        try:
                            text = chunk.decode('utf-8')
                            if text.startswith('data:'):
                                # Parse SSE data line
                                data_content = text[5:].strip()
                                if data_content == '[DONE]':
                                    yield text
                                else:
                                    # Parse JSON and mask model
                                    data = json.loads(data_content)
                                    if 'model' in data:
                                        data['model'] = 'brick'
                                    # Re-encode to SSE format
                                    yield f"data: {json.dumps(data)}\n\n"
                            else:
                                # Non-data lines (empty lines, etc.)
                                yield text
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            # If can't decode/modify, pass through as-is
                            yield chunk.decode('utf-8', errors='replace')
                        
    except Exception as e:
        # Yield error as SSE event
        error_data = {
            "error": {
                "message": str(e),
                "type": "streaming_error",
                "code": "internal_error"
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"


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


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    requested_model = body.get("model", "")
    is_stream = body.get("stream", False)

    # Log della richiesta ricevuta
    logger.info(f"=== RICHIESTA RICEVUTA ===")
    logger.info(f"Model: {requested_model}, Stream: {is_stream}")
    logger.info(f"Messages count: {len(messages)}")
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        if isinstance(content, list):
            content_types = [c.get("type", "unknown") for c in content if isinstance(c, dict)]
            logger.info(f"Message {i}: role={msg.get('role')}, content_types={content_types}")
        else:
            logger.info(f"Message {i}: role={msg.get('role')}, content_type=string")

    # Check if this is a routed request from vLLM SR (has x-selected-model header)
    selected_model = request.headers.get("x-selected-model")
    if selected_model:
        # This is a routed request - go directly to Regolo with the selected model
        # Restore original stream parameter from header (vLLM SR may have modified it)
        original_is_stream = request.headers.get("x-original-stream", "false").lower() == "true"
        logger.info(f"Routed request from vLLM SR with model: {selected_model}, original_stream={original_is_stream}, body_stream={is_stream}")
        # Use original stream value, not the one from body (which may have been modified by vLLM SR)
        is_stream = original_is_stream
        # Normalize messages to ensure content is string format (not array)
        normalized_messages = normalize_messages_for_vllm_sr(messages)

        if is_stream:
            # Return streaming response
            return StreamingResponse(
                call_regolo_llm_stream(normalized_messages, selected_model, is_stream),
                media_type="text/event-stream"
            )
        else:
            llm_result = await call_regolo_llm(normalized_messages, selected_model, is_stream)
            return JSONResponse(content=mask_response(llm_result))
    
    # Only accept "brick" model for client requests
    if requested_model != "brick":
        logger.warning(f"Invalid model requested: {requested_model}")
        return JSONResponse(
            status_code=400,
            content={"error": f"Model '{requested_model}' not supported. Use 'brick'."}
        )
    
    filter_result = filter_function(messages)
    logger.info(f"Filter result: {filter_result}")
    
    # CASO 1: Audio-only
    if filter_result["audio"] and not filter_result["image"] and not filter_result["text"]:
        logger.info("=== CASO 1: Audio-only ===")
        for msg in messages:
            content = msg.get("content", "")
            audio_url = extract_audio_url_from_content(content)
            if audio_url:
                # Log audio URL (troncato per privacy)
                audio_url_preview = audio_url[:50] + "..." if len(audio_url) > 50 else audio_url
                logger.info(f"Audio URL extracted: {audio_url_preview}")

                # Trascrizione audio con Whisper (batch - no streaming)
                logger.info("Calling Whisper...")
                whisper_result = await call_faster_whisper(audio_url)
                logger.info(f"Whisper result keys: {whisper_result.keys()}")
                logger.info(f"Whisper result: {json.dumps(whisper_result, indent=2)[:500]}")

                # Verifica errori nella trascrizione
                if whisper_result.get("error"):
                    logger.error(f"Whisper error: {whisper_result.get('error')}")
                    return JSONResponse(content=mask_response(whisper_result))

                # Estrai il testo dalla trascrizione
                transcription = extract_transcription_from_whisper(whisper_result)
                logger.info(f"Transcription extracted: {transcription[:100] if transcription else 'None'}")

                if not transcription:
                    logger.error("No transcription found in Whisper result")
                    return JSONResponse(content={
                        "error": "Unable to transcribe audio content",
                        "model": "brick"
                    })

                if is_stream:
                    # Stream mode: get model from vLLM SR, then stream LLM response
                    logger.info(f"Sending transcription to vLLM SR (for routing): {transcription[:100]}...")
                    transcription_messages = [{"role": "user", "content": transcription}]
                    vllm_result = await call_vllm_sr(transcription_messages, is_stream)

                    if vllm_result.get("error"):
                        logger.error(f"vLLM SR error: {vllm_result.get('error')}")
                        return JSONResponse(content=mask_response(vllm_result))

                    # Extract model from vLLM SR response
                    selected_model = vllm_result.get("model", "")
                    if selected_model:
                        logger.info(f"vLLM SR selected model: {selected_model}")
                        return StreamingResponse(
                            call_regolo_llm_stream(transcription_messages, selected_model, is_stream),
                            media_type="text/event-stream"
                        )
                    else:
                        return JSONResponse(content=mask_response(vllm_result))
                else:
                    # Batch mode: original flow
                    logger.info(f"Sending transcription to vLLM SR: {transcription[:100]}...")
                    transcription_messages = [{"role": "user", "content": transcription}]
                    llm_result = await call_vllm_sr(transcription_messages, is_stream)
                    logger.info(f"vLLM SR result keys: {llm_result.keys() if isinstance(llm_result, dict) else 'Not a dict'}")
                    logger.info(f"Response to frontend: {json.dumps(llm_result, indent=2)[:500]}")
                    return JSONResponse(content=mask_response(llm_result))
    
    # CASO 4: Image + Audio (no text)
    if filter_result["image"] and filter_result["audio"] and not filter_result["text"]:
        logger.info("=== CASO 4: Image + Audio ===")
        audio_transcription = ""
        image_content_str = ""

        for msg in messages:
            content = msg.get("content", "")
            audio_url = extract_audio_url_from_content(content)
            if audio_url:
                logger.info("Transcribing audio...")
                whisper_result = await call_faster_whisper(audio_url)
                audio_transcription = extract_transcription_from_whisper(whisper_result)
                logger.info(f"Audio transcription: {audio_transcription[:100] if audio_transcription else 'None'}")
            image_url = extract_image_url_from_content(content)
            if image_url:
                image_content_str = image_url

        # Process image
        logger.info("Processing image...")
        image_result = await process_image_with_fallback(image_content_str)

        # Combine and route via vLLM SR
        combined_messages = [
            {"role": "user", "content": f"Audio transcription: {audio_transcription}\n\nImage analysis: {image_result}"}
        ]

        if is_stream:
            # Stream mode: get model from vLLM SR, then stream LLM response
            vllm_result = await call_vllm_sr(combined_messages, is_stream)
            if vllm_result.get("error"):
                return JSONResponse(content=mask_response(vllm_result))

            selected_model = vllm_result.get("model", "")
            if selected_model:
                return StreamingResponse(
                    call_regolo_llm_stream(combined_messages, selected_model, is_stream),
                    media_type="text/event-stream"
                )
            else:
                return JSONResponse(content=mask_response(vllm_result))
        else:
            llm_result = await call_vllm_sr(combined_messages, is_stream)
            return JSONResponse(content=mask_response(llm_result))
    
    # CASO 5: Text + Audio
    if filter_result["text"] and filter_result["audio"] and not filter_result["image"]:
        logger.info("=== CASO 5: Text + Audio ===")
        audio_transcription = ""
        text_content = ""

        for msg in messages:
            content = msg.get("content", "")
            audio_url = extract_audio_url_from_content(content)
            if audio_url:
                logger.info("Transcribing audio...")
                whisper_result = await call_faster_whisper(audio_url)
                audio_transcription = extract_transcription_from_whisper(whisper_result)
                logger.info(f"Audio transcription: {audio_transcription[:100] if audio_transcription else 'None'}")
            text_from_content = extract_text_from_content(content)
            if text_from_content:
                text_content = text_from_content

        combined_messages = [
            {"role": "user", "content": f"Trascrizione audio: {audio_transcription}\n\nTesto originale: {text_content}"}
        ]

        if is_stream:
            # Stream mode: get model from vLLM SR, then stream LLM response
            vllm_result = await call_vllm_sr(combined_messages, is_stream)
            if vllm_result.get("error"):
                return JSONResponse(content=mask_response(vllm_result))

            selected_model = vllm_result.get("model", "")
            if selected_model:
                return StreamingResponse(
                    call_regolo_llm_stream(combined_messages, selected_model, is_stream),
                    media_type="text/event-stream"
                )
            else:
                return JSONResponse(content=mask_response(vllm_result))
        else:
            llm_result = await call_vllm_sr(combined_messages, is_stream)
            return JSONResponse(content=mask_response(llm_result))
    
    # CASO 6: Text + Image + Audio
    if filter_result["text"] and filter_result["image"] and filter_result["audio"]:
        logger.info("=== CASO 6: Text + Image + Audio ===")
        audio_transcription = ""
        image_content_str = ""
        text_content = ""

        for msg in messages:
            content = msg.get("content", "")
            audio_url = extract_audio_url_from_content(content)
            if audio_url:
                logger.info("Transcribing audio...")
                whisper_result = await call_faster_whisper(audio_url)
                audio_transcription = extract_transcription_from_whisper(whisper_result)
                logger.info(f"Audio transcription: {audio_transcription[:100] if audio_transcription else 'None'}")
            image_url = extract_image_url_from_content(content)
            if image_url:
                image_content_str = image_url
            text_from_content = extract_text_from_content(content)
            if text_from_content:
                text_content = text_from_content

        # Process image
        logger.info("Processing image...")
        image_result = await process_image_with_fallback(image_content_str)

        combined_messages = [
            {"role": "user", "content": f"Trascrizione audio: {audio_transcription}\n\nImmagine result: {image_result}\n\nTesto originale: {text_content}"}
        ]

        if is_stream:
            # Stream mode: get model from vLLM SR, then stream LLM response
            vllm_result = await call_vllm_sr(combined_messages, is_stream)
            if vllm_result.get("error"):
                return JSONResponse(content=mask_response(vllm_result))

            selected_model = vllm_result.get("model", "")
            if selected_model:
                return StreamingResponse(
                    call_regolo_llm_stream(combined_messages, selected_model, is_stream),
                    media_type="text/event-stream"
                )
            else:
                return JSONResponse(content=mask_response(vllm_result))
        else:
            llm_result = await call_vllm_sr(combined_messages, is_stream)
            return JSONResponse(content=mask_response(llm_result))
    
    # CASO 7: Text-only
    if filter_result["text"] and not filter_result["image"] and not filter_result["audio"]:
        normalized_messages = normalize_messages_for_vllm_sr(messages)

        if is_stream:
            # Stream mode: get model from vLLM SR (routing only), then stream LLM response
            vllm_result = await call_vllm_sr(normalized_messages, is_stream)
            if vllm_result.get("error"):
                return JSONResponse(content=mask_response(vllm_result))

            selected_model = vllm_result.get("model", "")
            if selected_model:
                logger.info(f"Routing selected model: {selected_model}, streaming to Regolo API")
                return StreamingResponse(
                    call_regolo_llm_stream(normalized_messages, selected_model, is_stream),
                    media_type="text/event-stream"
                )
            else:
                logger.error("vLLM SR did not return a model")
                return JSONResponse(content=mask_response(vllm_result))
        else:
            # Batch mode: vLLM SR handles full routing and calls Regolo
            llm_result = await call_vllm_sr(normalized_messages, is_stream)
            return JSONResponse(content=mask_response(llm_result))
    
    # CASO 3: Image + Text
    if filter_result["image"] and filter_result["text"] and not filter_result["audio"]:
        for msg in messages:
            content = msg.get("content", "")
            image_url = extract_image_url_from_content(content)
            if image_url:
                if is_stream:
                    return StreamingResponse(
                        call_qwen3_vl_stream(image_url),
                        media_type="text/event-stream"
                    )
                else:
                    image_result = await call_qwen3_vl(image_url)
                    return JSONResponse(content=mask_response(image_result))
    
    # CASO 2: Image-only
    if filter_result["image"] and not filter_result["text"] and not filter_result["audio"]:
        for msg in messages:
            content = msg.get("content", "")
            image_url = extract_image_url_from_content(content)
            if image_url:
                # Prova OCR con DeepSeek (batch - no streaming)
                ocr_result = await call_deepseek_ocr(image_url)

                # Estrai il testo dall'OCR
                ocr_text = ""
                if ocr_result.get("choices"):
                    ocr_text = ocr_result["choices"][0].get("message", {}).get("content", "")

                # Se OCR ha successo (testo significativo trovato), invia a vLLM SR
                if ocr_text and len(ocr_text.strip()) > 10:
                    ocr_messages = [{"role": "user", "content": ocr_text}]

                    if is_stream:
                        # Stream mode: get model from vLLM SR, then stream LLM response
                        vllm_result = await call_vllm_sr(ocr_messages, is_stream)
                        if vllm_result.get("error"):
                            return JSONResponse(content=mask_response(vllm_result))

                        selected_model = vllm_result.get("model", "")
                        if selected_model:
                            return StreamingResponse(
                                call_regolo_llm_stream(ocr_messages, selected_model, is_stream),
                                media_type="text/event-stream"
                            )
                        else:
                            return JSONResponse(content=mask_response(vllm_result))
                    else:
                        llm_result = await call_vllm_sr(ocr_messages, is_stream)
                        return JSONResponse(content=mask_response(llm_result))

                # Se OCR fallisce, usa Qwen3-VL per analisi visiva
                if is_stream:
                    return StreamingResponse(
                        call_qwen3_vl_stream(image_url),
                        media_type="text/event-stream"
                    )
                else:
                    image_result = await call_qwen3_vl(image_url)
                    return JSONResponse(content=mask_response(image_result))
    
    # Fallback
    return JSONResponse(content={
        "error": "Unable to process request with current modality combination",
        "filter": filter_result
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
