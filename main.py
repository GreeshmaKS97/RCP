from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json

from logger import audit_store
from pipeline import execute_upload_pipeline, execute_query_pipeline

app = FastAPI(title="Customizable Chatbot Telemetry Sandbox", version="1.0.0")

# Enable CORS for standard developer environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session tracking for convenience
class QueryRequest(BaseModel):
    query_text: str
    user_id: str = "usr_active_42"

# Ensure directories exist
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Try mounting static directory if it exists and has content (mounted later if needed)
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    pass

@app.get("/")
async def root():
    # Redirect root to dashboard for instant developer access
    return RedirectResponse(url="/dashboard")


# -------------------------------------------------------------
# CORE API ENDPOINTS
# -------------------------------------------------------------

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...), user_id: str = "usr_active_42"):
    """
    Simulates document upload and triggers the complete preprocessing, chunking,
    embedding, and database indexing pipeline under a unified telemetry scope.
    """
    try:
        file_bytes = await file.read()
        audit_entry = execute_upload_pipeline(
            filename=file.filename,
            file_bytes=file_bytes,
            user_id=user_id
        )
        
        # If parsing or validation failed, return appropriate error status
        if audit_entry.get("log_error"):
            return {
                "success": False,
                "request_id": audit_entry["log_request_id"],
                "error": audit_entry["log_error"],
                "trace": audit_entry["log_pipeline_trace"]
            }
            
        return {
            "success": True,
            "request_id": audit_entry["log_request_id"],
            "filename": audit_entry["doc_filename"],
            "chunks_created": audit_entry["chunk_count"],
            "parse_strategy": audit_entry["parse_strategy"],
            "classification": audit_entry["clf_label"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
async def query_chatbot(payload: QueryRequest):
    """
    Simulates a query in the chatbot system. Runs Retrieval, Reranking, and LLM 
    answer generation while tracking latency and status parameters.
    """
    try:
        audit_entry = execute_query_pipeline(
            query_text=payload.query_text,
            user_id=payload.user_id
        )
        
        if audit_entry.get("log_error"):
            return {
                "success": False,
                "request_id": audit_entry["log_request_id"],
                "error": audit_entry["log_error"],
                "trace": audit_entry["log_pipeline_trace"]
            }
            
        return {
            "success": True,
            "request_id": audit_entry["log_request_id"],
            "answer": audit_entry["llm_answer"],
            "tokens_used": audit_entry["llm_tokens_used"],
            "latency_ms": audit_entry["llm_latency_ms"],
            "sources": [
                {
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "section": chunk["section"],
                    "score": chunk["rerank_score"]
                }
                for chunk in audit_entry.get("rerank_chunks", [])
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------
# ANALYTICS & MONITORING API
# -------------------------------------------------------------

@app.get("/api/analytics/summary")
async def get_summary():
    """
    Returns aggregated telemetry statistics (total requests, token sizes,
    average LLM latency, step timings breakdown).
    """
    try:
        stats = audit_store.get_summary_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/traces")
async def get_traces(page: int = 1, limit: int = 10, search: str = None):
    """
    Retrieves individual execution traces in reverse chronological order with pagination and optional search.
    """
    try:
        offset = (page - 1) * limit
        traces, total = audit_store.get_recent_traces(limit=limit, offset=offset, search=search)
        total_pages = (total + limit - 1) // limit if limit > 0 else 0
        
        return {
            "traces": traces,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": total_pages
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/demo/seed")
async def seed_demo_data():
    """
    Populates the database with simulated runs (uploads, queries, errors)
    to instantly demonstrate the dashboard's visualization capabilities.
    """
    try:
        # 1. Successful Uploads
        execute_upload_pipeline(
            filename="system_release_notes.pdf",
            file_bytes=b"Release notes text structure and metadata info for version 1.4.",
            user_id="usr_admin_01"
        )
        execute_upload_pipeline(
            filename="chatbot_architecture.pdf",
            file_bytes=b"This document describes validation, parsing, classification, chunks, and reranking layers.",
            user_id="usr_admin_01"
        )
        execute_upload_pipeline(
            filename="quarterly_invoices.csv",
            file_bytes=b"InvoiceId,Amount,Date,Customer\nINV-001,1500,2026-05-15,AcmeCorp\nINV-002,2800,2026-05-20,BetaIndustries",
            user_id="usr_billing"
        )
        execute_upload_pipeline(
            filename="user_avatar_scan.png",
            file_bytes=b"OCR scan data and character validation text segments.",
            user_id="usr_dev_99"
        )
        
        # 2. Validation / Parser Failures
        execute_upload_pipeline(
            filename="dangerous_patch.exe",
            file_bytes=b"Simulated payload containing malicious executable bytes.",
            user_id="usr_hacker_x"
        )
        execute_upload_pipeline(
            filename="corrupt.pdf",
            file_bytes=b"Malformed header and truncated byte string.",
            user_id="usr_dev_99"
        )
        
        # 3. Successful Queries
        execute_query_pipeline(
            query_text="How does the document parsing strategy work?",
            user_id="usr_dev_99"
        )
        execute_query_pipeline(
            query_text="What models are used in the reranking layer?",
            user_id="usr_dev_99"
        )
        execute_query_pipeline(
            query_text="Explain customizable chatbot telemetry variables",
            user_id="usr_admin_01"
        )
        execute_query_pipeline(
            query_text="Tell me about Antigravity multi-agent systems",
            user_id="usr_active_42"
        )
        
        # 4. Query Failures
        execute_query_pipeline(
            query_text="Trigger a simulate-failure action in Llama generator",
            user_id="usr_active_42"
        )
        
        return {"success": True, "message": "Successfully seeded 11 diverse telemetry records."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------
# DASHBOARD UI ROUTER
# -------------------------------------------------------------

@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """
    Renders the beautiful dark-themed monitoring interface.
    The HTML is loaded dynamically from the templates folder.
    """
    try:
        with open("templates/dashboard.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        # Fallback in case templates folder hasn't loaded yet
        return HTMLResponse(
            content="<h3>Dashboard HTML loading... Please refresh in a moment.</h3>",
            status_code=503
        )
