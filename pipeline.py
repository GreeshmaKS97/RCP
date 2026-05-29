import time
import uuid
import random
import os
from datetime import datetime
from logger import PipelineContext, trace_stage, audit_store

# -------------------------------------------------------------
# MOCK DATABASE / CHUNKS FOR RETRIEVAL SIMULATION
# -------------------------------------------------------------
MOCK_DOCUMENTS = [
    {
        "text": "Antigravity Chatbot Framework version 1.4 has been released. It supports multi-agent scheduling and features built-in telemetry capturing step timings.",
        "metadata": {"page": 1, "section": "Introduction", "source": "release_notes.pdf"}
    },
    {
        "text": "The validation layer is responsible for validating file sizes, mime types, and file extensions. Files exceeding 15MB are rejected with errors.",
        "metadata": {"page": 2, "section": "Validation Layer", "source": "architecture.pdf"}
    },
    {
        "text": "Document Parsing supports PDF, DOCX, XLSX, and CSV. For scanned documents, an OCR strategy is engaged using Tesseract or cloud Vision APIs.",
        "metadata": {"page": 4, "section": "Document Parsing", "source": "architecture.pdf"}
    },
    {
        "text": "Embeddings are generated using OpenAI's 'text-embedding-3-small' model, producing vectors with 1536 dimensions. Chunks are batched to reduce API calls.",
        "metadata": {"page": 7, "section": "Embeddings", "source": "architecture.pdf"}
    },
    {
        "text": "Cross-Encoder models from sentence-transformers are used in the reranking layer to score the top-K retrieved candidate chunks and pick the top-N.",
        "metadata": {"page": 9, "section": "Reranking Layer", "source": "architecture.pdf"}
    },
    {
        "text": "The LLM Generator is powered by meta-llama/Llama-3.2-1B. It has a context length of 128k tokens, but context window feeds are capped at 4k tokens.",
        "metadata": {"page": 12, "section": "Generator", "source": "architecture.pdf"}
    }
]


# -------------------------------------------------------------
# PIPELINE STAGES (UPLOAD CHAIN)
# -------------------------------------------------------------

@trace_stage("auth")
def run_auth(user_id: str, token: str = "mock-jwt-token-xyz"):
    time.sleep(random.uniform(0.02, 0.05)) # Simulating DB/OAuth latency
    ctx = PipelineContext.get_current()
    
    ctx.auth_user_id = user_id
    ctx.auth_token = token
    ctx.auth_session = {
        "user_id": user_id,
        "roles": ["developer", "user"],
        "client_ip": "127.0.0.1",
        "device": "Web Browser (Chrome)"
    }
    return True


@trace_stage("upload")
def run_upload(filename: str, file_bytes: bytes):
    time.sleep(random.uniform(0.04, 0.08)) # Simulating network upload latency
    ctx = PipelineContext.get_current()
    
    # Set upload parameters
    ctx.doc_raw_file = file_bytes
    ctx.doc_filename = filename
    
    # Determine MIME type based on extension
    ext = filename.split(".")[-1].lower()
    mime_map = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "csv": "text/csv",
        "png": "image/png",
        "jpg": "image/jpeg"
    }
    ctx.doc_mime_type = mime_map.get(ext, "application/octet-stream")
    ctx.doc_upload_id = "doc_" + str(uuid.uuid4())[:8]
    ctx.doc_upload_ts = datetime.utcnow().isoformat() + "Z"
    return ctx.doc_upload_id


@trace_stage("validation")
def run_validation():
    time.sleep(random.uniform(0.01, 0.03))
    ctx = PipelineContext.get_current()
    
    file_bytes = ctx.doc_raw_file
    filename = ctx.doc_filename
    
    val_file_size_bytes = len(file_bytes) if file_bytes else 0
    ctx.val_file_size_bytes = val_file_size_bytes
    
    val_errors = []
    
    # Simulated validation checks
    if val_file_size_bytes > 15 * 1024 * 1024:
        val_errors.append("File size exceeds maximum limit of 15MB.")
    
    ext = filename.split(".")[-1].lower()
    if ext not in ["pdf", "docx", "xlsx", "csv", "png", "jpg"]:
        val_errors.append(f"Unsupported file extension: .{ext}")
        
    val_is_valid = len(val_errors) == 0
    ctx.val_is_valid = val_is_valid
    ctx.val_errors = val_errors
    
    if not val_is_valid:
        raise ValueError(f"File validation failed: {', '.join(val_errors)}")
        
    return val_is_valid


@trace_stage("parser")
def run_parser():
    time.sleep(random.uniform(0.15, 0.35)) # Simulating parsing CPU/IO latency
    ctx = PipelineContext.get_current()
    
    filename = ctx.doc_filename
    ext = filename.split(".")[-1].lower()
    
    # Pick parser strategy
    if ext in ["png", "jpg"]:
        parse_strategy = "ocr"
    elif ext in ["xlsx", "xls"]:
        parse_strategy = "xlsx"
    elif ext in ["csv"]:
        parse_strategy = "csv"
    elif ext in ["docx"]:
        parse_strategy = "docx"
    else:
        parse_strategy = "pdf"
        
    ctx.parse_strategy = parse_strategy
    
    # Simulate a parsing failure for demonstration
    if filename.lower() == "corrupt.pdf":
        raise IOError("PDF Structure Corrupt: Unable to locate cross-reference (xref) table.")
        
    # Generate mock parsed outputs
    ctx.parse_raw_text = f"Sample text parsed from file {filename}. Contain chatbot architecture data."
    ctx.parse_pages = [
        {"page_number": 1, "text": f"Page 1 content of {filename}. Framework details.", "char_count": 250},
        {"page_number": 2, "text": f"Page 2 content of {filename}. Validation layer details.", "char_count": 310}
    ]
    ctx.parse_tables = [
        {"table_name": "Prefixes Table", "headers": ["Module", "Prefix"], "rows": [["Auth", "auth_"], ["Upload", "doc_"]]}
    ]
    
    if parse_strategy == "ocr":
        ctx.parse_images = [b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR...mock_image_bytes"]
        ctx.parse_ocr_text = "TEXT RECOVERED FROM OCR PIPELINE: Customizable Chatbot telemetry module."
    else:
        ctx.parse_images = []
        ctx.parse_ocr_text = ""
        
    return parse_strategy


@trace_stage("structurer")
def run_structurer():
    time.sleep(random.uniform(0.05, 0.12)) # Simulating JSON structuring
    ctx = PipelineContext.get_current()
    
    ctx.struct_json = {
        "document_type": "technical_specification",
        "author": "Engineering Team",
        "created_date": "2026-05-27",
        "sections_count": 2
    }
    ctx.struct_metadata = {
        "title": ctx.doc_filename.replace(".pdf", "").replace("_", " ").title(),
        "author": "Anonymous",
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "page_count": len(ctx.parse_pages) if ctx.parse_pages else 0
    }
    ctx.struct_sections = [
        {"heading": "1. Introduction", "level": 1, "start_page": 1},
        {"heading": "2. Core System Architecture", "level": 1, "start_page": 2}
    ]
    return True


@trace_stage("classifier")
def run_classifier():
    time.sleep(random.uniform(0.04, 0.08)) # Simulating zero-shot class labeling
    ctx = PipelineContext.get_current()
    
    # Classify document based on filename keywords
    fn = ctx.doc_filename.lower()
    if "invoice" in fn or "billing" in fn:
        label = "financial_invoice"
        confidence = 0.94
    elif "cv" in fn or "resume" in fn:
        label = "hr_resume"
        confidence = 0.98
    else:
        label = "technical_documentation"
        confidence = 0.88
        
    ctx.clf_label = label
    ctx.clf_confidence = confidence
    ctx.clf_labels_all = [
        (label, confidence),
        ("general_memo", 0.08),
        ("legal_contract", 0.04)
    ]
    return label


@trace_stage("chunker")
def run_chunker():
    time.sleep(random.uniform(0.03, 0.06)) # Chunker tokenization
    ctx = PipelineContext.get_current()
    
    chunk_size = 512
    chunk_overlap = 64
    
    ctx.chunk_size = chunk_size
    ctx.chunk_overlap = chunk_overlap
    
    # Generate mock chunks
    ctx.chunk_list = [
        f"Chunk 1 text from {ctx.doc_filename}. Core configuration variables: auth_user_id, doc_filename.",
        f"Chunk 2 text from {ctx.doc_filename}. Custom logger telemetry: log_pipeline_trace, log_audit_entry."
    ]
    ctx.chunk_meta_list = [
        {"page": 1, "section": "1. Introduction", "index": 0},
        {"page": 2, "section": "2. Core System Architecture", "index": 1}
    ]
    ctx.chunk_count = len(ctx.chunk_list)
    return ctx.chunk_count


@trace_stage("embedder")
def run_embedder():
    time.sleep(random.uniform(0.08, 0.18)) # Simulating API latency to OpenAI
    ctx = PipelineContext.get_current()
    
    ctx.emb_model_name = "text-embedding-3-small"
    ctx.emb_dim = 1536
    ctx.emb_batch_size = 8
    
    # Mock embedding vectors (random float arrays of 1536 floats)
    mock_vector = [round(random.uniform(-0.05, 0.05), 6) for _ in range(20)] # Truncated for print size
    ctx.emb_vectors = [mock_vector for _ in range(ctx.chunk_count)]
    return len(ctx.emb_vectors)


@trace_stage("chromadb")
def run_chromadb():
    time.sleep(random.uniform(0.03, 0.06))
    ctx = PipelineContext.get_current()
    
    ctx.db_collection_name = "chatbot_docs_collection"
    ctx.db_client = "<ChromaDB Client Object>"
    ctx.db_collection = "<ChromaDB Collection Object>"
    
    doc_ids = [f"chunk_{ctx.doc_upload_id}_{i}" for i in range(ctx.chunk_count)]
    ctx.db_doc_ids = doc_ids
    return doc_ids


# -------------------------------------------------------------
# PIPELINE STAGES (QUERY CHAIN)
# -------------------------------------------------------------

@trace_stage("retriever")
def run_retriever(query_text: str):
    time.sleep(random.uniform(0.05, 0.12)) # Simulating similarity vector query
    ctx = PipelineContext.get_current()
    
    ctx.query_text = query_text
    # Mock embedding for the query text
    ctx.query_emb = [round(random.uniform(-0.05, 0.05), 6) for _ in range(20)]
    ctx.query_top_k = 4
    
    # Match query words to mock documents to simulate retrieval
    results = []
    query_lower = query_text.lower()
    
    scored_docs = []
    for doc in MOCK_DOCUMENTS:
        score = 0.35 + random.uniform(0.0, 0.25)
        # Boost score if keywords match
        for word in query_lower.split():
            if len(word) > 3 and word[:4] in doc["text"].lower():
                score += 0.20
        scored_docs.append((doc, min(score, 0.98)))
        
    # Sort by score descending and take top_k
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    for i, (doc, score) in enumerate(scored_docs[:ctx.query_top_k]):
        results.append({
            "id": f"chunk_sys_{i}",
            "text": doc["text"],
            "metadata": doc["metadata"],
            "score": round(score, 4)
        })
        
    ctx.query_results = results
    return results


@trace_stage("reranker")
def run_reranker():
    time.sleep(random.uniform(0.03, 0.08)) # Cross-encoder scoring latency
    ctx = PipelineContext.get_current()
    
    ctx.rerank_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ctx.rerank_top_n = 3
    
    query_results = ctx.query_results or []
    
    # Simulate reranking (shuffle score slightly or prioritize higher-quality documents)
    reranked = []
    for chunk in query_results:
        # Cross encoder yields a score between -10 and 10 usually
        score = float(chunk["score"]) * 8.0 - random.uniform(-1.0, 1.0)
        reranked.append((chunk, round(score, 3)))
        
    # Sort by rerank score
    reranked.sort(key=lambda x: x[1], reverse=True)
    
    rerank_scores = []
    rerank_chunks = []
    
    for chunk, r_score in reranked[:ctx.rerank_top_n]:
        rerank_scores.append(r_score)
        # Structure final context chunk
        rerank_chunks.append({
            "chunk_id": chunk["id"],
            "text": chunk["text"],
            "page": chunk["metadata"]["page"],
            "section": chunk["metadata"]["section"],
            "source": chunk["metadata"]["source"],
            "rerank_score": r_score
        })
        
    ctx.rerank_scores = rerank_scores
    ctx.rerank_chunks = rerank_chunks
    return rerank_chunks


@trace_stage("generator")
def run_generator():
    time.sleep(random.uniform(0.40, 0.85)) # Simulating local Llama 3.2 1B generation
    ctx = PipelineContext.get_current()
    
    ctx.llm_model_name = "meta-llama/Llama-3.2-1B"
    
    # Formulate llm_context
    chunks = ctx.rerank_chunks or []
    llm_context = "\n\n".join([f"Source: {c['source']} (Page {c['page']}):\n{c['text']}" for c in chunks])
    ctx.llm_context = llm_context
    
    # Formulate llm_prompt
    llm_prompt = f"System: Use the context below to answer the user's question.\nContext:\n{llm_context}\n\nQuestion: {ctx.query_text}\nAnswer:"
    ctx.llm_prompt = llm_prompt
    
    # Simulate a pipeline failure for demo
    if "simulate-failure" in ctx.query_text.lower():
        raise TimeoutError("Llama-3.2 Inference Error: Connection timed out after 850ms.")
        
    # Dynamic Mock Responses based on query keywords
    query_lower = ctx.query_text.lower()
    
    if "parsing" in query_lower:
        llm_answer = "According to the system architecture specification, document parsing supports multiple file formats including PDF, DOCX, XLSX, and CSV. When scanned images or PDFs are uploaded, the pipeline dynamically activates an OCR parsing strategy to extract characters."
    elif "rerank" in query_lower or "cross-encoder" in query_lower:
        llm_answer = "The system utilizes Cross-Encoder models (specifically cross-encoder/ms-marco-MiniLM-L-6-v2) in the reranking layer. This layer takes the top candidate chunks retrieved by similarity search and computes exact relevance scores to supply only the top-N most accurate context pieces to the Llama model."
    elif "telemetry" in query_lower or "variable" in query_lower or "logging" in query_lower:
        llm_answer = "The Customizable Chatbot incorporates unified tracking. Major stages like Authentication, Validation, Embedding, Retrieval, and Reranking trace their execution status, timing, and errors, appending structured audit logs to standard outputs and SQLite databases."
    else:
        llm_answer = "Based on the retrieved architecture notes, the system is designed as a modular pipeline starting from Authentication and Document Upload all the way to validation, chunking, embedding, vector database indexing, semantic retriever search, reranking, and Llama 3.2 1B text generation."
        
    ctx.llm_response_raw = f"<thinking>Formulating answer based on 3 context snippets</thinking>\n{llm_answer}"
    ctx.llm_answer = llm_answer
    
    # Tokens used (rough estimate)
    prompt_tokens = len(llm_prompt.split()) * 1.3
    answer_tokens = len(llm_answer.split()) * 1.3
    ctx.llm_tokens_used = int(prompt_tokens + answer_tokens)
    
    return llm_answer


# -------------------------------------------------------------
# ORCHESTRATION PIPELINES (MAIN INTERFACE FOR CONTROLLERS)
# -------------------------------------------------------------

def execute_upload_pipeline(filename: str, file_bytes: bytes, user_id: str = "usr_active_42") -> dict:
    """
    Executes the entire upload-validation-parse-chunk-embed-store pipeline.
    Catches errors at the top level and guarantees that an audit log is saved.
    """
    ctx = PipelineContext.initialize(request_type="upload")
    try:
        run_auth(user_id=user_id)
        run_upload(filename=filename, file_bytes=file_bytes)
        run_validation()
        run_parser()
        run_structurer()
        run_classifier()
        run_chunker()
        run_embedder()
        run_chromadb()
    except Exception:
        # The exceptions are already tracked by @trace_stage, we just let the runner catch it
        pass
    finally:
        # Write audit record to JSONL and SQLite
        audit_entry = audit_store.write_audit_entry(ctx)
        return audit_entry


def execute_query_pipeline(query_text: str, user_id: str = "usr_active_42") -> dict:
    """
    Executes the retrieve-rerank-generate query pipeline.
    Guarantees an audit log is saved even if errors occur.
    """
    ctx = PipelineContext.initialize(request_type="query")
    try:
        run_auth(user_id=user_id)
        run_retriever(query_text=query_text)
        run_reranker()
        run_generator()
    except Exception:
        pass
    finally:
        audit_entry = audit_store.write_audit_entry(ctx)
        return audit_entry
