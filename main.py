from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json

from logger import audit_store
from pipeline import execute_upload_pipeline, execute_query_pipeline

import time
import uuid
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Depends, Header, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

security_scheme = HTTPBearer()


app = FastAPI(
    title="Customizable Chatbot - Chat Endpoints Module",
    description="Implements the retrieval, reranking, and generation layers of the customizable chatbot.",
    version="1.0.0"
)

# =====================================================================
# PYDANTIC MODEL SCHEMAS (Conforms to naming conventions)
# =====================================================================

class ChatRequest(BaseModel):
    query_text: str = Field(
        ..., 
        description="Raw user question string",
        example="How does the reranking layer work in this architecture?"
    )
    db_collection_name: str = Field(
        ..., 
        description="ChromaDB collection identifier",
        example="chatbot_kb_v1"
    )
    query_top_k: int = Field(
        default=5, 
        ge=1, 
        le=50, 
        description="Number of candidates to retrieve initially"
    )
    rerank_top_n: int = Field(
        default=3, 
        ge=1, 
        le=20, 
        description="Final number of top context chunks to pass to the LLM"
    )

class RerankChunk(BaseModel):
    id: str = Field(..., description="Unique text chunk identifier")
    text: str = Field(..., description="Text content of the retrieved chunk")
    metadata: Dict[str, Any] = Field(..., description="Per-chunk metadata (page, section, index)")
    score: float = Field(..., description="Cross-encoder relevance score")

class ChatResponse(BaseModel):
    llm_answer: str = Field(..., description="Cleaned, final answer shown to user")
    rerank_chunks: List[RerankChunk] = Field(..., description="Re-ranked top-N context chunks passed to the LLM")

# =====================================================================
# INTEGRATION SERVICES (Interfaces/Stubs for team members to plug into)
# =====================================================================

class EmbeddingServiceStub:
    """
    Handles generating query_emb via 'text-embedding-3-small'.
    Team integration note: Swap out this stub with your OpenAI / model api client.
    """
    def __init__(self):
        self.emb_model_name = "text-embedding-3-small"
        self.emb_dim = 1536

    def generate_embedding(self, query_text: str) -> List[float]:
        query_emb = [0.0] * self.emb_dim
        for i, char in enumerate(query_text[:50]):
            if i < self.emb_dim:
                query_emb[i] = ord(char) / 255.0
        return query_emb

class ChromaDBServiceStub:
    """
    Handles connection and retrieval from ChromaDB.
    Team integration note: Replace with real db_client and collection query logic.
    """
    def __init__(self):
        self.db_client = "MockChromaClient"
        self._mock_records = [
            {
                "id": "chunk_1",
                "text": "Customizable chatbot architecture supports dynamic modular pipelines: auth, doc upload, validation, parsing, embedding, chroma database, cross-encoder reranker, and generator.",
                "metadata": {"page": 1, "section": "Overview"}
            },
            {
                "id": "chunk_2",
                "text": "The validation layer verifies file safety, schema consistency, and payload size. Validated documents undergo parsing with strategies matching extensions like pdf, docx, xlsx, or csv.",
                "metadata": {"page": 2, "section": "Validation & Parsing"}
            },
            {
                "id": "chunk_3",
                "text": "Embedding generation uses model 'text-embedding-3-small' to yield vectors stored in ChromaDB collections, which are queried using Cosine / L2 distance metrics.",
                "metadata": {"page": 3, "section": "Embeddings & Indexing"}
            },
            {
                "id": "chunk_4",
                "text": "The reranking layer utilizes a cross-encoder model to re-evaluate relevance of similarity search candidates. It outputs rerank_scores and selects the top rerank_top_n context chunks.",
                "metadata": {"page": 4, "section": "Reranker"}
            },
            {
                "id": "chunk_5",
                "text": "Llama 3.2 1B (llm_model_name='meta-llama/Llama-3.2-1B') ingests the llm_prompt containing query_text and concatenated llm_context to produce llm_answer.",
                "metadata": {"page": 5, "section": "Generator"}
            }
        ]

    def similarity_search(self, db_collection_name: str, query_emb: List[float], query_top_k: int) -> List[Dict[str, Any]]:
        query_results = []
        for record in self._mock_records[:query_top_k]:
            query_results.append({
                "id": record["id"],
                "text": record["text"],
                "metadata": record["metadata"],
                "distance": 0.35
            })
        return query_results

class RerankingServiceStub:
    """
    Reranker layer that scoring text chunks using cross-encoders.
    Team integration note: Hook up to SentenceTransformers / CrossEncoder client.
    """
    def __init__(self):
        self.rerank_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def rerank(self, query_text: str, query_results: List[Dict[str, Any]], rerank_top_n: int) -> List[Dict[str, Any]]:
        scored_chunks = []
        query_words = set(query_text.lower().split())
        
        for result in query_results:
            chunk_text = result["text"]
            chunk_words = set(chunk_text.lower().split())
            intersection = query_words.intersection(chunk_words)
            rerank_score = len(intersection) / max(len(query_words.union(chunk_words)), 1)
            rerank_score = round(rerank_score + 0.1, 4)
            
            scored_chunks.append({
                "id": result["id"],
                "text": chunk_text,
                "metadata": result["metadata"],
                "score": rerank_score
            })
        
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        rerank_chunks = scored_chunks[:rerank_top_n]
        return rerank_chunks

class LLMGenerationServiceStub:
    """
    Orchestrates prompt building and text inference using Llama 3.2 1B.
    Team integration note: Plug in Hugging Face pipeline / vLLM / Ollama endpoint.
    """
    def __init__(self):
        self.llm_model_name = "meta-llama/Llama-3.2-1B"

    def generate_answer(self, query_text: str, rerank_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        llm_context = "\n\n".join([f"--- Context {i+1} ---\n{c['text']}" for i, c in enumerate(rerank_chunks)])
        llm_prompt = (
            f"System Prompt: You are a helpful AI assistant. Answer the user question accurately using the provided context chunks.\n\n"
            f"Context:\n{llm_context}\n\n"
            f"User Question: {query_text}\n"
            f"Response:"
        )
        
        time.sleep(0.35) 
        
        lower_query = query_text.lower()
        if "rerank" in lower_query or "reranker" in lower_query:
            llm_response_raw = "Response: In this architecture, the reranking layer applies the cross-encoder model ('cross-encoder/ms-marco-MiniLM-L-6-v2') to rank similarities. It selects the top rerank_top_n elements to build the prompt context contextually, optimizing performance."
            llm_answer = "In this architecture, the reranking layer applies the cross-encoder model ('cross-encoder/ms-marco-MiniLM-L-6-v2') to rank similarities. It selects the top rerank_top_n elements to build the prompt context contextually, optimizing performance."
        elif "validation" in lower_query or "parsing" in lower_query:
            llm_response_raw = "Response: The validation layer ensures size limits, format checks, and safety rules are matched before parsing the raw text with specialized extraction routines depending on document type."
            llm_answer = "The validation layer ensures size limits, format checks, and safety rules are matched before parsing the raw text with specialized extraction routines depending on document type."
        else:
            llm_response_raw = "Response: The customizable chatbot features a modular RAG setup running through auth, document extraction, vector indexing inside ChromaDB, cross-encoder rerank, and Llama 3.2 1B."
            llm_answer = "The customizable chatbot features a modular RAG setup running through auth, document extraction, vector indexing inside ChromaDB, cross-encoder rerank, and Llama 3.2 1B."
            
        llm_latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        input_tokens = len(llm_prompt.split()) * 1.3
        output_tokens = len(llm_response_raw.split()) * 1.3
        llm_tokens_used = int(input_tokens + output_tokens)
        
        return {
            "llm_context": llm_context,
            "llm_prompt": llm_prompt,
            "llm_response_raw": llm_response_raw,
            "llm_answer": llm_answer,
            "llm_tokens_used": llm_tokens_used,
            "llm_latency_ms": llm_latency_ms
        }

embedding_service = EmbeddingServiceStub()
chromadb_service = ChromaDBServiceStub()
reranking_service = RerankingServiceStub()
llm_service = LLMGenerationServiceStub()

# =====================================================================
# AUTHENTICATION DEPENDENCY (Conforms to naming conventions)
# =====================================================================

async def get_current_authenticated_user(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)) -> Dict[str, Any]:
    """
    Validates token from Authorization Header.
    Extracts variables: auth_token, auth_user_id, auth_session.
    """
    auth_token = credentials.credentials
    
    try:
        if auth_token == "invalid_token_test":
            raise ValueError("Token is blacklisted or invalid.")
            
        auth_user_id = f"usr_{uuid.uuid5(uuid.NAMESPACE_DNS, auth_token).hex[:8]}"
        auth_session = {
            "auth_user_id": auth_user_id,
            "auth_token_exp": time.time() + 3600,
            "scopes": ["chat"],
            "client_ip": "127.0.0.1"
        }
        
        return {
            "auth_token": auth_token,
            "auth_user_id": auth_user_id,
            "auth_session": auth_session
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication token signature is invalid: {str(e)}"
        )

# =====================================================================
# CHAT PIPELINE ROUTE
# =====================================================================

@app.post(
    "/api/v1/chat", 
    response_model=ChatResponse, 
    status_code=status.HTTP_200_OK,
    summary="Ask a question and get a contextual answer",
    description="Endpoint that runs full RAG flow: semantic ChromaDB search, cross-encoder rerank, Llama-3.2 prompt assembly, and text generation."
)
async def chat_endpoint(
    request_data: ChatRequest
):
    # Extract query variables
    query_text: str = request_data.query_text
    db_collection_name: str = request_data.db_collection_name
    query_top_k: int = request_data.query_top_k
    rerank_top_n: int = request_data.rerank_top_n
    
    # 1. EMBEDDING GENERATION
    query_emb: List[float] = embedding_service.generate_embedding(query_text)
    
    # 2. VECTOR DB RETRIEVAL
    query_results: List[Dict[str, Any]] = chromadb_service.similarity_search(
        db_collection_name=db_collection_name, 
        query_emb=query_emb, 
        query_top_k=query_top_k
    )
    
    # 3. RERANKING CANDIDATES
    rerank_chunks = reranking_service.rerank(
        query_text=query_text, 
        query_results=query_results, 
        rerank_top_n=rerank_top_n
    )
    
    # 4. CONTEXT & ANSWER GENERATION
    generation_outputs = llm_service.generate_answer(
        query_text=query_text, 
        rerank_chunks=rerank_chunks
    )
    
    # Extraction matching variables
    llm_context: str = generation_outputs["llm_context"]
    llm_prompt: str = generation_outputs["llm_prompt"]
    llm_response_raw: str = generation_outputs["llm_response_raw"]
    llm_answer = generation_outputs["llm_answer"]
    
    # Return formatted ChatResponse containing exact variables
    return ChatResponse(
        llm_answer=llm_answer,
        rerank_chunks=[
            RerankChunk(
                id=c["id"],
                text=c["text"],
                metadata=c["metadata"],
                score=c["score"]
            ) for c in rerank_chunks
        ]
    )

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
