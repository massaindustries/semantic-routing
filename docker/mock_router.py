from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/v1/chat/completions")
async def chat_completions():
    """Mock VSR service returning a static model name.

    Returns a JSON payload compatible with the expected VSR response format.
    The gateway uses the 'model' field from the JSON to determine the selected backend.
    """
    return JSONResponse(content={"model": "mock-model"})
