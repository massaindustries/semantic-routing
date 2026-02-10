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

_time_logger_instance = None
_time_logger_lock = threading.Lock()

def get_logger(db_path: str = None) -> 'TimeLogger':
    global _time_logger_instance
    if _time_logger_instance is None:
        with _time_logger_lock:
            if _time_logger_instance is None:
                _time_logger_instance = TimeLogger(db_path)
    return _time_logger_instance

def init_logger(db_path: str = None) -> 'TimeLogger':
    global _time_logger_instance
    _time_logger_instance = TimeLogger(db_path)
    return _time_logger_instance

class TimeLogger:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(os.path.dirname(__file__), 'timelog.db')
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
                if data.get('modality'):
                    conn.execute('UPDATE time_logs SET modality = ? WHERE request_id = ?',
                        (data['modality'], data['request_id']))
            conn.commit()
        except Exception:
            pass
    
    def _queue_write(self, data: dict):
        try:
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
        
        conditions = []
        params = []
        
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)
        else:
            conditions.append("timestamp >= '1970-01-01'")
        
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)
        else:
            conditions.append("timestamp <= '2099-12-31'")
        
        if modality:
            conditions.append("modality = ?")
            params.append(modality)
        
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        
        total_query = f"SELECT COUNT(*) as total, SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count FROM time_logs{where_clause}"
        
        total_row = conn.execute(total_query, params).fetchone()
        total = total_row['total'] if total_row else 0
        success_count = total_row['success_count'] if total_row else 0
        success_rate = (success_count / total * 100) if total > 0 else 0
        
        phase_stats = {}
        
        date_params = []
        if start_date:
            date_params.append(start_date)
        else:
            date_params.append('1970-01-01')
        if end_date:
            date_params.append(end_date)
        else:
            date_params.append('2099-12-31')
        
        for row in conn.execute('''
            SELECT phase, COUNT(*) as count, AVG(duration_ms) as avg_duration,
                   MIN(duration_ms) as min_duration, MAX(duration_ms) as max_duration
            FROM request_phases
            WHERE request_id IN (SELECT request_id FROM time_logs WHERE timestamp >= ? AND timestamp <= ?)
            GROUP BY phase
        ''', date_params):
            phase_stats[row['phase']] = {
                'phase': row['phase'],
                'count': row['count'],
                'avg_duration_ms': round(row['avg_duration'], 2) if row['avg_duration'] else 0,
                'min_duration_ms': round(row['min_duration'], 2) if row['min_duration'] else 0,
                'max_duration_ms': round(row['max_duration'], 2) if row['max_duration'] else 0
            }
        
        if total > 0:
            phase_stats['total_request'] = {
                'phase': 'total_request',
                'count': total,
                'avg_duration_ms': 0,
                'min_duration_ms': 0,
                'max_duration_ms': 0,
                'success_rate': round(success_rate, 1)
            }
        
        return {'phases': list(phase_stats.values()), 'total_requests': total}
    
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
        cursor = conn.execute('''SELECT modality, COUNT(*) as count FROM time_logs WHERE modality IS NOT NULL AND modality != '' GROUP BY modality''')
        return {row['modality']: row['count'] for row in cursor.fetchall() if row['modality']}
    
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
        conn.execute('''CREATE TABLE IF NOT EXISTS vllm_classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT,
            original_model TEXT,
            selected_model TEXT,
            decision TEXT,
            category TEXT,
            confidence REAL,
            matched_rules TEXT,
            matched_keywords TEXT,
            reasoning TEXT,
            timestamp TEXT,
            FOREIGN KEY(request_id) REFERENCES time_logs(request_id)
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS request_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT,
            role TEXT,
            content TEXT,
            content_type TEXT DEFAULT 'text',
            sequence_order INTEGER,
            FOREIGN KEY(request_id) REFERENCES time_logs(request_id)
        )''')
        conn.execute('''CREATE INDEX IF NOT EXISTS idx_timestamp ON time_logs(timestamp)''')
        conn.execute('''CREATE INDEX IF NOT EXISTS idx_phase ON time_logs(phase)''')
        conn.execute('''CREATE INDEX IF NOT EXISTS idx_modality ON time_logs(modality)''')
        conn.execute('''CREATE INDEX IF NOT EXISTS idx_vllm_request ON vllm_classifications(request_id)''')
        conn.execute('''CREATE INDEX IF NOT EXISTS idx_msg_request ON request_messages(request_id)''')
        conn.commit()
    
    def clear_old_logs(self, days: int = 30):
        conn = self._get_conn()
        conn.execute('DELETE FROM time_logs WHERE timestamp < datetime("now", ?)', (f'-{days} days',))
        conn.commit()
    
    def log_vllm_classification(self, request_id: str, classification_data: Dict[str, Any]):
        """Log vLLM SR classification data extracted from response headers."""
        conn = self._get_conn()
        try:
            conn.execute('''
                INSERT INTO vllm_classifications 
                (request_id, original_model, selected_model, decision, category, 
                 confidence, matched_rules, matched_keywords, reasoning, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                request_id,
                classification_data.get('original_model'),
                classification_data.get('selected_model'),
                classification_data.get('decision'),
                classification_data.get('category'),
                classification_data.get('confidence'),
                json.dumps(classification_data.get('matched_rules', [])),
                json.dumps(classification_data.get('matched_keywords', [])),
                classification_data.get('reasoning'),
                datetime.now().isoformat()
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error logging vLLM classification: {e}")
    
    def log_request_messages(self, request_id: str, messages: List[Dict[str, Any]]):
        """Log request messages for analysis."""
        conn = self._get_conn()
        try:
            for idx, msg in enumerate(messages):
                content = msg.get('content', '')
                content_type = 'text'
                
                # Detect content type from message structure
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get('type') == 'image_url':
                                content_type = 'image'
                            elif block.get('type') == 'audio':
                                content_type = 'audio'
                
                # Normalize content to string for storage
                if isinstance(content, list):
                    texts = []
                    for block in content:
                        if isinstance(block, dict) and block.get('type') == 'text':
                            texts.append(block.get('text', ''))
                    content_str = ' '.join(texts) if texts else json.dumps(content)
                else:
                    content_str = str(content)
                
                conn.execute('''
                    INSERT INTO request_messages 
                    (request_id, role, content, content_type, sequence_order)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    request_id,
                    msg.get('role', 'user'),
                    content_str[:4000],  # Limit to avoid overflow
                    content_type,
                    idx
                ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error logging request messages: {e}")
    
    def get_vllm_classification(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get vLLM classification data for a request."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                'SELECT * FROM vllm_classifications WHERE request_id = ? ORDER BY id DESC LIMIT 1',
                (request_id,)
            )
            row = cursor.fetchone()
            if row:
                result = dict(row)
                # Parse JSON fields
                result['matched_rules'] = json.loads(result.get('matched_rules', '[]'))
                result['matched_keywords'] = json.loads(result.get('matched_keywords', '[]'))
                return result
            return None
        except Exception as e:
            logger.error(f"Error getting vLLM classification: {e}")
            return None
    
    def get_request_messages(self, request_id: str) -> List[Dict[str, Any]]:
        """Get request messages for a request."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                'SELECT * FROM request_messages WHERE request_id = ? ORDER BY sequence_order',
                (request_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting request messages: {e}")
            return []
    
    def get_request_details(self, request_id: str) -> Dict[str, Any]:
        """Get complete request details including classification and messages."""
        conn = self._get_conn()
        try:
            # Get base request info
            cursor = conn.execute('SELECT * FROM time_logs WHERE request_id = ?', (request_id,))
            request_row = cursor.fetchone()
            request_info = dict(request_row) if request_row else {}
            
            # Get classification
            classification = self.get_vllm_classification(request_id)
            
            # Get messages
            messages = self.get_request_messages(request_id)
            
            # Get timeline
            timeline = self.get_phase_timeline(request_id)
            
            return {
                'request': request_info,
                'classification': classification,
                'messages': messages,
                'timeline': timeline
            }
        except Exception as e:
            logger.error(f"Error getting request details: {e}")
            return {'request': {}, 'classification': None, 'messages': [], 'timeline': []}
