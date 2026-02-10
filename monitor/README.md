# Monitor - Time Logging System

Sistema di time logging per identificare inefficienze e colli di bottiglia nel processing delle richieste.

## Struttura

```
monitor/
├── __init__.py
├── time_logger.py      # Core logging module (SQLite + timing)
├── core_with_logging.py # core.py con logging integrato (esempio)
├── dashboard.py        # API backend per la dashboard
├── dashboard/
│   └── index.html      # Frontend dashboard web
└── README.md
```

## Installazione

```bash
pip install fastapi uvicorn httpx
```

## Utilizzo

### 1. Time Logger Base

```python
from monitor.time_logger import TimeLogger, timed_phase

# Inizializza il logger
logger = TimeLogger('timelog.db')

# Inizia una richiesta
request_id = logger.start_request(
    modality='text',
    model='gpt-4',
    stream=False
)

# Timing di una fase
logger.log_phase_start(request_id, 'vllm_routing')
# ... operazione ...
logger.log_phase_end(request_id, 'vllm_routing', success=True)

# Alternativa: context manager
with timed_phase(request_id, 'image_processing'):
    # operazione tracciata
    pass

# Eventi custom
logger.log_custom_event(request_id, 'cache_hit', {'key': 'value'})

# Fine richiesta
logger.end_request(request_id, success=True)
```

### 2. Context Manager Decorator

```python
from monitor.time_logger import timed_phase

request_id = "abc123"

with timed_phase(request_id, 'my_operation'):
    # Il tempo viene loggato automaticamente
    do_something()
```

### 3. Query Statistics

```python
stats = logger.get_statistics(
    start_date='2024-01-01',
    end_date='2024-01-31',
    modality='text'
)
# stats = {
#     'phases': [...],
#     'total_requests': 1500,
#     'avg_total_duration_ms': 450.5
# }
```

## Dashboard Web

Avvia il server della dashboard su porta 8001:

```bash
cd monitor
python dashboard.py
```

Accedi a `http://localhost:8001` per vedere:
- Statistiche generali (totali, durata media, tasso successo)
- Tempo medio per fase (bar chart)
- Distribuzione per modality/model
- Timeline richieste recenti
- Errori recenti

## API Endpoints

| Endpoint | Metodo | Descrizione |
|----------|--------|-------------|
| `/api/stats` | GET | Statistiche generali |
| `/api/requests` | GET | Richieste recenti |
| `/api/errors` | GET | Errori recenti |
| `/api/distribution/modality` | GET | Distribuzione modality |
| `/api/distribution/model` | GET | Distribuzione model |
| `/api/stats/daily` | GET | Statistiche giornaliere |
| `/api/requests/{id}/timeline` | GET | Timeline fase request |
| `/api/events/custom` | POST | Log evento custom |
| `/api/logs/clear` | DELETE | Cancella log vecchi |

## Fasi Tracciate

| Fase | Descrizione |
|------|-------------|
| `request_received` | Arrivo richiesta HTTP |
| `modality_detection` | Analisi tipo contenuto |
| `audio_transcription` | Whisper transcription |
| `image_processing` | OCR / Vision processing |
| `vllm_routing` | Chiamata Semantic Router |
| `regolo_response` | Risposta modello finale |
| `total_request` | Tempo totale end-to-end |

## Database Schema

```sql
-- Tabella principale richieste
CREATE TABLE time_logs (
    id INTEGER PRIMARY KEY,
    request_id TEXT UNIQUE,
    timestamp TEXT,
    phase TEXT,
    duration_ms REAL,
    model TEXT,
    modality TEXT,
    stream BOOLEAN,
    success BOOLEAN,
    error_type TEXT,
    error_message TEXT,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    custom_data TEXT
);

-- Timing dettagliato per fase
CREATE TABLE request_phases (
    request_id TEXT,
    phase TEXT,
    start_time REAL,
    end_time REAL,
    duration_ms REAL,
    UNIQUE(request_id, phase)
);

-- Eventi custom
CREATE TABLE custom_events (
    request_id TEXT,
    event_name TEXT,
    timestamp REAL,
    data TEXT
);
```

## Integrazione con core.py Esistente

Per aggiungere logging a un file FastAPI esistente:

```python
from monitor.time_logger import TimeLogger

monitor_logger = TimeLogger()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    request_id = None
    try:
        body = await request.json()
        request_id = monitor_logger.start_request(
            modality='unknown',
            model=body.get('model', ''),
            stream=body.get('stream', False),
            request_size_bytes=len(json.dumps(body))
        )
        
        monitor_logger.log_phase_start(request_id, 'processing')
        # ... tuo codice ...
        monitor_logger.log_phase_end(request_id, 'processing')
        
        monitor_logger.end_request(request_id, success=True)
        return response
        
    except Exception as e:
        if request_id:
            monitor_logger.end_request(request_id, success=False, 
                error_type=type(e).__name__, error_message=str(e))
        raise
```

## Pulizia Automatica

Il logger cancella automaticamente log più vecchi di 30 giorni. Configurabile:

```python
logger.clear_old_logs(days=7)  # Cancella log più vecchi di 7 giorni
```
