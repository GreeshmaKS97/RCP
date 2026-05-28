import contextvars
import uuid
import time
import os
import json
import sqlite3
from datetime import datetime
from functools import wraps
import traceback

# Thread-local / Async-local Context Variable
_pipeline_ctx = contextvars.ContextVar("pipeline_ctx", default=None)

class PipelineContext:
    """
    Context-local container to track telemetry variables for a single pipeline run.
    Uses contextvars to remain thread-safe and asynchronous-safe.
    """
    def __init__(self, request_id: str = None, request_type: str = "query"):
        self._data = {
            "log_request_id": request_id or str(uuid.uuid4()),
            "log_pipeline_trace": {},
            "log_error": None,
            "request_type": request_type, # 'upload' or 'query'
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
        
    def set(self, key: str, value):
        self._data[key] = value
        
    def get(self, key: str, default=None):
        return self._data.get(key, default)
        
    def to_dict(self):
        return self._data.copy()

    def __getattr__(self, name):
        # Allow dotted attribute reading: ctx.doc_filename
        if name.startswith('_'):
            return super().__getattribute__(name)
        return self._data.get(name)

    def __setattr__(self, name, value):
        # Allow dotted attribute writing: ctx.doc_filename = "test.pdf"
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    @classmethod
    def get_current(cls) -> "PipelineContext":
        ctx = _pipeline_ctx.get()
        if ctx is None:
            ctx = cls()
            _pipeline_ctx.set(ctx)
        return ctx

    @classmethod
    def initialize(cls, request_id: str = None, request_type: str = "query") -> "PipelineContext":
        ctx = cls(request_id, request_type)
        _pipeline_ctx.set(ctx)
        return ctx


def trace_stage(stage_name: str):
    """
    Decorator to wrap sync or async RAG pipeline steps.
    Measures duration, records timestamps, detects status, and captures errors.
    """
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            ctx = PipelineContext.get_current()
            start_time = time.perf_counter()
            start_ts = datetime.utcnow().isoformat() + "Z"
            
            trace = ctx.log_pipeline_trace
            trace[stage_name] = {
                "start_time": start_ts,
                "status": "PROCESSING",
                "latency_ms": 0.0
            }
            
            try:
                result = func(*args, **kwargs)
                latency = (time.perf_counter() - start_time) * 1000.0
                trace[stage_name].update({
                    "end_time": datetime.utcnow().isoformat() + "Z",
                    "status": "SUCCESS",
                    "latency_ms": round(latency, 2)
                })
                if stage_name == "generator" or stage_name == "llm":
                    ctx.llm_latency_ms = round(latency, 2)
                return result
            except Exception as e:
                latency = (time.perf_counter() - start_time) * 1000.0
                trace[stage_name].update({
                    "end_time": datetime.utcnow().isoformat() + "Z",
                    "status": "FAILED",
                    "latency_ms": round(latency, 2)
                })
                # Format a highly readable traceback
                error_msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                ctx.log_error = error_msg.strip()
                if stage_name == "generator" or stage_name == "llm":
                    ctx.llm_latency_ms = round(latency, 2)
                raise e
                
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            ctx = PipelineContext.get_current()
            start_time = time.perf_counter()
            start_ts = datetime.utcnow().isoformat() + "Z"
            
            trace = ctx.log_pipeline_trace
            trace[stage_name] = {
                "start_time": start_ts,
                "status": "PROCESSING",
                "latency_ms": 0.0
            }
            
            try:
                result = await func(*args, **kwargs)
                latency = (time.perf_counter() - start_time) * 1000.0
                trace[stage_name].update({
                    "end_time": datetime.utcnow().isoformat() + "Z",
                    "status": "SUCCESS",
                    "latency_ms": round(latency, 2)
                })
                if stage_name == "generator" or stage_name == "llm":
                    ctx.llm_latency_ms = round(latency, 2)
                return result
            except Exception as e:
                latency = (time.perf_counter() - start_time) * 1000.0
                trace[stage_name].update({
                    "end_time": datetime.utcnow().isoformat() + "Z",
                    "status": "FAILED",
                    "latency_ms": round(latency, 2)
                })
                error_msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                ctx.log_error = error_msg.strip()
                if stage_name == "generator" or stage_name == "llm":
                    ctx.llm_latency_ms = round(latency, 2)
                raise e
                
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


class AuditStore:
    """
    Manages structured storage for log audit records.
    Saves to a local JSONL log file and a SQLite database for querying.
    """
    def __init__(self, base_dir: str = "logs"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.base_dir, "audit_logs.jsonl")
        self.db_path = os.path.join(self.base_dir, "audit_store.db")
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                request_id TEXT PRIMARY KEY,
                request_type TEXT,
                timestamp TEXT,
                user_id TEXT,
                filename TEXT,
                mime_type TEXT,
                file_size_bytes INTEGER,
                parse_strategy TEXT,
                query_text TEXT,
                llm_latency_ms REAL,
                llm_tokens_used INTEGER,
                llm_answer TEXT,
                status TEXT,
                error TEXT,
                full_trace_json TEXT,
                audit_entry_json TEXT
            )
        """)
        conn.commit()
        conn.close()

    def write_audit_entry(self, ctx: PipelineContext) -> dict:
        """
        Builds the final log_audit_entry dictionary and saves it.
        """
        raw_data = ctx.to_dict()
        
        # Structure the formal log_audit_entry according to requirements
        audit_entry = {
            "log_request_id": raw_data.get("log_request_id"),
            "request_type": raw_data.get("request_type", "query"),
            "timestamp": raw_data.get("created_at"),
            
            # Auth
            "auth_user_id": raw_data.get("auth_user_id"),
            "auth_session": raw_data.get("auth_session"),
            
            # File parameters
            "doc_filename": raw_data.get("doc_filename"),
            "doc_mime_type": raw_data.get("doc_mime_type"),
            "doc_upload_id": raw_data.get("doc_upload_id"),
            "doc_upload_ts": raw_data.get("doc_upload_ts"),
            
            # Validation & Parse parameters
            "val_is_valid": raw_data.get("val_is_valid"),
            "val_errors": raw_data.get("val_errors"),
            "val_file_size_bytes": raw_data.get("val_file_size_bytes"),
            "parse_strategy": raw_data.get("parse_strategy"),
            "parse_pages_count": len(raw_data.get("parse_pages", [])) if raw_data.get("parse_pages") else None,
            
            # Structure & Classify
            "clf_label": raw_data.get("clf_label"),
            "clf_confidence": raw_data.get("clf_confidence"),
            
            # Chunking & Embeddings
            "chunk_count": raw_data.get("chunk_count"),
            "emb_model_name": raw_data.get("emb_model_name"),
            
            # Query details
            "query_text": raw_data.get("query_text"),
            "query_top_k": raw_data.get("query_top_k"),
            
            # Rerank details
            "rerank_model_name": raw_data.get("rerank_model_name"),
            "rerank_top_n": raw_data.get("rerank_top_n"),
            "rerank_chunks": raw_data.get("rerank_chunks"),
            
            # Generator
            "llm_model_name": raw_data.get("llm_model_name"),
            "llm_latency_ms": raw_data.get("llm_latency_ms"),
            "llm_tokens_used": raw_data.get("llm_tokens_used"),
            "llm_answer": raw_data.get("llm_answer"),
            
            # Logging & Timing Diagnostics
            "log_pipeline_trace": raw_data.get("log_pipeline_trace"),
            "log_error": raw_data.get("log_error")
        }
        
        ctx.set("log_audit_entry", audit_entry)
        
        # Determine overall request status
        status = "SUCCESS"
        if audit_entry["log_error"]:
            status = "FAILED"
        else:
            # Check individual stage failures in trace
            for stage, info in audit_entry["log_pipeline_trace"].items():
                if info.get("status") == "FAILED":
                    status = "FAILED"
                    break
        
        # 1. Write to JSON-Lines
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit_entry) + "\n")
            
        # 2. Write to SQLite database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO audit_logs (
                request_id, request_type, timestamp, user_id, 
                filename, mime_type, file_size_bytes, parse_strategy, 
                query_text, llm_latency_ms, llm_tokens_used, llm_answer, 
                status, error, full_trace_json, audit_entry_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            audit_entry["log_request_id"],
            audit_entry["request_type"],
            audit_entry["timestamp"],
            audit_entry["auth_user_id"],
            audit_entry["doc_filename"],
            audit_entry["doc_mime_type"],
            audit_entry["val_file_size_bytes"],
            audit_entry["parse_strategy"],
            audit_entry["query_text"],
            audit_entry["llm_latency_ms"],
            audit_entry["llm_tokens_used"],
            audit_entry["llm_answer"],
            status,
            audit_entry["log_error"],
            json.dumps(audit_entry["log_pipeline_trace"]),
            json.dumps(audit_entry)
        ))
        conn.commit()
        conn.close()
        
        return audit_entry

    def get_summary_stats(self) -> dict:
        """
        Calculates aggregate statistics over all logs stored in SQLite.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        stats = {
            "total_requests": 0,
            "total_uploads": 0,
            "total_queries": 0,
            "success_rate": 100.0,
            "error_count": 0,
            "average_latency_ms": 0.0,
            "total_tokens_used": 0,
            "average_tokens_per_query": 0,
            "strategy_breakdown": {},
            "latency_by_stage": {}
        }
        
        try:
            # Basic counts
            cursor.execute("SELECT COUNT(*) as cnt FROM audit_logs")
            stats["total_requests"] = cursor.fetchone()["cnt"]
            
            if stats["total_requests"] > 0:
                cursor.execute("SELECT COUNT(*) as cnt FROM audit_logs WHERE request_type='upload'")
                stats["total_uploads"] = cursor.fetchone()["cnt"]
                
                cursor.execute("SELECT COUNT(*) as cnt FROM audit_logs WHERE request_type='query'")
                stats["total_queries"] = cursor.fetchone()["cnt"]
                
                cursor.execute("SELECT COUNT(*) as cnt FROM audit_logs WHERE status='FAILED'")
                stats["error_count"] = cursor.fetchone()["cnt"]
                
                stats["success_rate"] = round(((stats["total_requests"] - stats["error_count"]) / stats["total_requests"]) * 100, 2)
                
                # Average latency for LLM
                cursor.execute("SELECT AVG(llm_latency_ms) as avg_lat FROM audit_logs WHERE request_type='query' AND llm_latency_ms IS NOT NULL")
                avg_lat = cursor.fetchone()["avg_lat"]
                stats["average_latency_ms"] = round(avg_lat, 2) if avg_lat else 0.0
                
                # Total Tokens
                cursor.execute("SELECT SUM(llm_tokens_used) as total_tok, AVG(llm_tokens_used) as avg_tok FROM audit_logs WHERE llm_tokens_used IS NOT NULL")
                row = cursor.fetchone()
                stats["total_tokens_used"] = row["total_tok"] or 0
                stats["average_tokens_per_query"] = round(row["avg_tok"], 1) if row["avg_tok"] else 0
                
                # Strategy breakdown
                cursor.execute("SELECT parse_strategy, COUNT(*) as cnt FROM audit_logs WHERE parse_strategy IS NOT NULL GROUP BY parse_strategy")
                stats["strategy_breakdown"] = {row["parse_strategy"]: row["cnt"] for row in cursor.fetchall()}
                
                # Latency by stage parsing
                cursor.execute("SELECT full_trace_json FROM audit_logs")
                stage_latencies = {}
                stage_counts = {}
                for row in cursor.fetchall():
                    try:
                        trace = json.loads(row["full_trace_json"])
                        for stage, trace_info in trace.items():
                            if "latency_ms" in trace_info and trace_info["status"] == "SUCCESS":
                                stage_latencies[stage] = stage_latencies.get(stage, 0.0) + trace_info["latency_ms"]
                                stage_counts[stage] = stage_counts.get(stage, 0) + 1
                    except Exception:
                        continue
                
                stats["latency_by_stage"] = {
                    stage: round(stage_latencies[stage] / stage_counts[stage], 2)
                    for stage in stage_latencies
                }
        finally:
            conn.close()
            
        return stats

    def get_recent_traces(self, limit: int = 10, offset: int = 0, search: str = None) -> tuple[list, int]:
        """
        Retrieves the most recent audit entries from the database with pagination and optional search filtering.
        Returns a tuple of (traces_list, total_matching_count).
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        traces = []
        total = 0
        try:
            query = "SELECT audit_entry_json, status FROM audit_logs"
            count_query = "SELECT COUNT(*) FROM audit_logs"
            where_clause = ""
            params = []
            
            if search and search.strip():
                search_val = f"%{search.strip()}%"
                where_clause = " WHERE filename LIKE ? OR query_text LIKE ? OR request_id LIKE ? OR user_id LIKE ?"
                params = [search_val, search_val, search_val, search_val]
            
            # Get total count matching search
            cursor.execute(count_query + where_clause, params)
            total = cursor.fetchone()[0]
            
            # Get paginated matching traces
            paginated_query = query + where_clause + " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            query_params = params + [limit, offset]
            
            cursor.execute(paginated_query, query_params)
            for row in cursor.fetchall():
                try:
                    entry = json.loads(row["audit_entry_json"])
                    entry["status"] = row["status"]
                    traces.append(entry)
                except Exception:
                    continue
        finally:
            conn.close()
            
        return traces, total

# Global Singleton Instance for convenience
audit_store = AuditStore()

