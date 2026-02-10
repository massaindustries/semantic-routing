import sqlite3
import json
import time
import uuid
import threading
import queue
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

PHASES = ['modality_detection', 'audio_transcription', 'image_processing', 
          'vllm_routing', 'regolo_response', 'total_request']

class TimeLogger:
    _instance = None
    
    def __init__(self, db_path: str = None):
        if self._initialized:
            return
        self._initialized = True
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), 'timelog.db')
        self.db_path = db_path
        self._local = threading.local()
        self._write_queue = queue.Queue(maxsize=10000)
        self._worker_thread = threading.Thread(target=self._background_writer, daemon=True)
        self._worker_thread.start()
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _background_writer(self):
        while True:
            try:
                data = self._write_queue.get(timeout=1.0)
                self._execute_write(data)
            except queue.Empty:
                continue
            except Exception:
                pass
    
    def _execute_write(self, data: dict):
        conn = self._get_conn()
        try:
            if data['type'] == 'start_request':
                conn.execute('''INSERT OR IGNORE INTO time_logs 
                    (request_id, timestamp, modality, model, stream, request_size_bytes, success)
                    VALUES (?, ?, ?, ?, ?, ?, 1)''',
                    (data['request_id'], data['timestamp'], data['modality'], 
                     data['model'], data['stream'], data['request_size_bytes']))
            elif data['type'] == 'phase_start':
                conn.execute('''INSERT INTO request_phases (request_id, phase, start_time)
                    VALUES (?, ?, ?)''',
                    (data['request_id'], data['phase'], data['start_time']))
            elif data['type'] == 'phase_end':
                conn.execute('''UPDATE request_phases SET end_time = ?, duration_ms = ?
                    WHERE request_id = ? AND phase = ?''',
                    (data['end_time'], data['duration_ms'], data['request_id'], data['phase']))
                conn.execute('''UPDATE time_logs SET phase = ?, duration_ms = ?, success = ?,
                    error_type = ?, error_message = ? WHERE request_id = ?''',
                    (data['phase'], data['duration_ms'], data['success'],
                     data.get('error_type'), data.get('error_message'), data['request_id']))
            elif data['type'] == 'end_request':
                conn.execute('''UPDATE time_logs SET phase = 'total_request', 
                    duration_ms = ?, success = ?, error_type = ?, error_message = ?,
                    response_size_bytes = ? WHERE request_id = ?''',
                    (data['duration_ms'], data['success'], data.get('error_type'),
                     data.get('error_message'), data.get('response_size_bytes'), data['request_id']))
            elif data['type'] == 'update_metadata':
                conn.execute('UPDATE time_logs SET modality = ? WHERE request_id = ?',
                    (data['modality'], data['request_id']))
            conn.commit()
        except Exception:
            pass
    
    def _queue_write(self, data: dict, blocking: bool = False):
        try:
            if blocking:
                self._execute_write(data)
            else:
                self._write_queue.put_nowait(data)
        except queue.Full:
            pass
    
    def start_request(self, request_id: str, modality: str, model: str,
                      stream: bool, request_size_bytes: int):
        self._queue_write({
            'type': 'start_request',
            'request_id': request_id,
            'timestamp': datetime.now().isoformat(),
            'modality': modality,
            'model': model,
            'stream': stream,
            'request_size_bytes': request_size_bytes
        })
        return request_id
    
    def log_phase_start(self, request_id: str, phase: str):
        self._queue_write({
            'type': 'phase_start',
            'request_id': request_id,
            'phase': phase,
            'start_time': time.perf_counter()
        })
    
    def log_phase_end(self, request_id: str, phase: str, success: bool = True,
                      error_type: str = None, error_message: str = None):
        duration_ms = 0.0
        conn = self._get_conn()
        cursor = conn.execute('''SELECT start_time FROM request_phases 
            WHERE request_id = ? AND phase = ?''', (request_id, phase))
        row = cursor.fetchone()
        if row:
            duration_ms = (time.perf_counter() - row['start_time']) * 1000
        
        self._queue_write({
            'type': 'phase_end',
            'request_id': request_id,
            'phase': phase,
            'end_time': time.perf_counter(),
            'duration_ms': duration_ms,
            'success': success,
            'error_type': error_type,
            'error_message': error_message
        })
    
    def update_metadata(self, request_id: str, modality: str = None):
        if modality:
            self._queue_write({'type': 'update_metadata', 'request_id': request_id, 'modality': modality})
    
    def end_request(self, request_id: str, success: bool = True,
                    error_type: str = None, error_message: str = None,
                    response_size_bytes: int = 0):
        conn = self._get_conn()
        cursor = conn.execute('''SELECT SUM(duration_ms) FROM request_phases 
            WHERE request_id = ?''', (request_id,))
        row = cursor.fetchone()
        total_duration = row[0] if row and row[0] else 0.0
        
        self._queue_write({
            'type': 'end_request',
            'request_id': request_id,
            'duration_ms': total_duration,
            'success': success,
            'error_type': error_type,
            'error_message': error_message,
            'response_size_bytes': response_size_bytes
        })
    
    def log_custom_event(self, request_id: str, event_name: str, data: Dict[str, Any] = None):
        conn = self._get_conn()
        conn.execute('''
            INSERT INTO custom_events (request_id, event_name, timestamp, data)
            VALUES (?, ?, ?, ?)
        ''', (request_id, event_name, time.perf_counter(), json.dumps(data or {})))
        conn.commit()
    
    def get_statistics(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                       modality: Optional[str] = None) -> Dict[str, Any]:
        conn = self._get_conn()
        query = '''SELECT phase, COUNT(*) as count, AVG(duration_ms) as avg_duration,
                   MIN(duration_ms) as min_duration, MAX(duration_ms) as max_duration
                   FROM time_logs WHERE phase IS NOT NULL'''
        params = []
        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date)
        if end_date:
            query += ' AND timestamp <= ?'
            params.append(end_date)
        if modality:
            query += ' AND modality = ?'
            params.append(modality)
        query += ' GROUP BY phase'
        
        phases = []
        for row in conn.execute(query, params):
            phases.append({
                'phase': row['phase'],
                'count': row['count'],
                'avg_duration_ms': round(row['avg_duration'], 2) if row['avg_duration'] else 0,
                'min_duration_ms': round(row['min_duration'], 2) if row['min_duration'] else 0,
                'max_duration_ms': round(row['max_duration'], 2) if row['max_duration'] else 0
            })
        
        total = conn.execute('''SELECT COUNT(*) FROM time_logs WHERE phase = ?''', 
                            ('total_request',)).fetchone()
        return {'phases': phases, 'total_requests': total[0] if total else 0}
    
    def get_recent_requests(self, limit: int = 50) -> List[Dict]:
        conn = self._get_conn()
        cursor = conn.execute('''SELECT * FROM time_logs ORDER BY timestamp DESC LIMIT ?''', (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_errors(self, limit: int = 50) -> List[Dict]:
        conn = self._get_conn()
        cursor = conn.execute('''SELECT * FROM time_logs WHERE success = 0 ORDER BY timestamp DESC LIMIT ?''', (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_phase_timeline(self, request_id: str) -> List[Dict]:
        conn = self._get_conn()
        cursor = conn.execute('''SELECT * FROM request_phases WHERE request_id = ? ORDER BY start_time''', (request_id,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_modality_distribution(self) -> Dict[str, int]:
        conn = self._get_conn()
        cursor = conn.execute('''SELECT modality, COUNT(*) as count FROM time_logs GROUP BY modality''')
        return {row['modality']: row['count'] for row in cursor.fetchall()}
    
    def get_model_distribution(self) -> Dict[str, int]:
        conn = self._get_conn()
        cursor = conn.execute('''SELECT model, COUNT(*) as count FROM time_logs GROUP BY model''')
        return {row['model']: row['count'] for row in cursor.fetchall()}
    
    def get_daily_stats(self, days: int = 7) -> List[Dict]:
        conn = self._get_conn()
        cursor = conn.execute('''
            SELECT DATE(timestamp) as date, COUNT(*) as requests, AVG(duration_ms) as avg_duration
            FROM time_logs WHERE phase = 'total_request'
            GROUP BY DATE(timestamp) ORDER BY date DESC LIMIT ?
        ''', (days,))
        return [{'date': row['date'], 'requests': row['requests'],
                 'avg_duration_ms': round(row['avg_duration'], 2) if row['avg_duration'] else 0}
                for row in cursor.fetchall()]
    
    def _init_db(self):
        conn = self._get_conn()
        conn.execute('''CREATE TABLE IF NOT EXISTS time_logs (
            request_id TEXT PRIMARY KEY, timestamp TEXT, phase TEXT, duration_ms REAL,
            model TEXT, modality TEXT, stream BOOLEAN, success BOOLEAN,
            error_type TEXT, error_message TEXT, request_size_bytes INTEGER, response_size_bytes INTEGER
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS request_phases (
            request_id TEXT, phase TEXT, start_time REAL, end_time REAL, duration_ms REAL,
            PRIMARY KEY (request_id, phase)
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS custom_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT, event_name TEXT, timestamp REAL, data TEXT
        )''')
        conn.execute('''CREATE INDEX IF NOT EXISTS idx_timestamp ON time_logs(timestamp)''')
        conn.execute('''CREATE INDEX IF NOT EXISTS idx_phase ON time_logs(phase)''')
        conn.execute('''CREATE INDEX IF NOT EXISTS idx_modality ON time_logs(modality)''')
        conn.commit()
    
    def clear_old_logs(self, days: int = 30):
        conn = self._get_conn()
        conn.execute('DELETE FROM time_logs WHERE timestamp < datetime("now", ?)', (f'-{days} days',))
        conn.commit()

_instance = None

def get_logger(db_path: str = None) -> 'TimeLogger':
    global _instance
    if _instance is None:
        db = db_path or os.path.join(os.path.dirname(__file__), 'timelog.db')
        _instance = TimeLogger(db)
    return _instance

def init_logger(db_path: str = None):
    global _instance
    db = db_path or os.path.join(os.path.dirname(__file__), 'timelog.db')
    _instance = TimeLogger(db)
    return _instance
