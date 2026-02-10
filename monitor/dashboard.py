from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import os
import threading

from .time_logger import get_logger, PHASES

monitor_dir = os.path.dirname(os.path.abspath(__file__))
dashboard_dir = os.path.join(monitor_dir, 'dashboard')

app = FastAPI(title="Time Logging Monitor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def serve_dashboard():
    index_path = os.path.join(dashboard_dir, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Dashboard not found")

@app.get("/api/stats")
async def get_stats(start_date: Optional[str] = None, end_date: Optional[str] = None,
                   modality: Optional[str] = None) -> Dict[str, Any]:
    return get_logger().get_statistics(start_date, end_date, modality)

@app.get("/api/requests")
async def get_recent_requests(limit: int = 50) -> List[Dict]:
    return get_logger().get_recent_requests(limit)

@app.get("/api/errors")
async def get_errors(limit: int = 50) -> List[Dict]:
    return get_logger().get_errors(limit)

@app.get("/api/requests/{request_id}/timeline")
async def get_request_timeline(request_id: str) -> List[Dict]:
    return get_logger().get_phase_timeline(request_id)

@app.get("/api/distribution/modality")
async def get_modality_distribution() -> Dict[str, int]:
    return get_logger().get_modality_distribution()

@app.get("/api/distribution/model")
async def get_model_distribution() -> Dict[str, int]:
    return get_logger().get_model_distribution()

@app.get("/api/stats/daily")
async def get_daily_stats(days: int = 7) -> List[Dict]:
    return get_logger().get_daily_stats(days)

@app.get("/api/phases")
async def get_phases() -> List[str]:
    return PHASES

@app.get("/api/requests/{request_id}/classification")
async def get_request_classification(request_id: str) -> Optional[Dict]:
    return get_logger().get_vllm_classification(request_id)

@app.get("/api/requests/{request_id}/messages")
async def get_request_messages_endpoint(request_id: str) -> List[Dict]:
    return get_logger().get_request_messages(request_id)

@app.get("/api/requests/{request_id}/full")
async def get_request_full_details(request_id: str) -> Dict:
    return get_logger().get_request_details(request_id)

@app.delete("/api/logs/clear")
async def clear_old_logs(days: int = 30):
    get_logger().clear_old_logs(days)
    return {"message": f"Cleared logs older than {days} days"}

class LogCustomEventRequest(BaseModel):
    request_id: str
    event_name: str
    data: Optional[Dict[str, Any]] = None

@app.post("/api/events/custom")
async def log_custom_event(request: LogCustomEventRequest):
    get_logger().log_custom_event(request.request_id, request.event_name, request.data or {})
    return {"message": "Event logged"}

class BatchQueryRequest(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    modality: Optional[str] = None
    limit: int = 100

@app.post("/api/batch")
async def batch_query(request: BatchQueryRequest) -> Dict[str, Any]:
    return {
        "statistics": get_logger().get_statistics(request.start_date, request.end_date, request.modality),
        "recent_requests": get_logger().get_recent_requests(request.limit),
        "modality_distribution": get_logger().get_modality_distribution(),
        "model_distribution": get_logger().get_model_distribution(),
        "daily_stats": get_logger().get_daily_stats(7)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
