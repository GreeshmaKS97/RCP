import time
import logging
from typing import List, Dict, Any
import httpx

logger = logging.getLogger(__name__)


def generate_llm_answer(
    query_text: str,
    rerank_chunks: List[Dict[str, Any]],
    llm_model_name: str = "llama3.2:1b",
    ollama_url: str = "http://localhost:11434/api/generate",
    system_prompt: str = "You are a helpful AI assistant. Answer the user question accurately using the provided context chunks.",
) -> Dict[str, Any]:
    """
    Assembles a prompt from the query and re-ranked context chunks,
    sends it to a local Ollama instance running Llama 3.2 1B (synchronously),
    and returns LLM generation outputs conforming to the naming conventions.
    """
    # 1. Assemble llm_context
    context_parts = []
    for i, chunk in enumerate(rerank_chunks):
        text = chunk.get("text") or chunk.get("content") or ""
        context_parts.append(f"--- Context {i+1} ---\n{text}")
    llm_context = "\n\n".join(context_parts)

    # 2. Assemble llm_prompt
    llm_prompt = (
        f"System Prompt: {system_prompt}\n\n"
        f"Context:\n{llm_context}\n\n"
        f"User Question: {query_text}\n"
        f"Response:"
    )

    start_time = time.perf_counter()

    payload = {
        "model": llm_model_name,
        "prompt": llm_prompt,
        "stream": False,
    }

    try:
        logger.info(f"Sending prompt to Ollama model '{llm_model_name}' at {ollama_url}...")
        response = httpx.post(ollama_url, json=payload, timeout=60.0)
        response.raise_for_status()
        result = response.json()

        llm_response_raw = result.get("response", "")
        llm_answer = llm_response_raw.strip()

        # Extract tokens and latency from Ollama response, falling back to estimations if missing
        prompt_tokens = result.get("prompt_eval_count", 0)
        completion_tokens = result.get("eval_count", 0)
        llm_tokens_used = prompt_tokens + completion_tokens

        total_duration_ns = result.get("total_duration")
        if total_duration_ns:
            llm_latency_ms = total_duration_ns / 1_000_000.0
        else:
            llm_latency_ms = (time.perf_counter() - start_time) * 1000.0

    except Exception as e:
        logger.error(f"Error communicating with Ollama: {e}")
        raise RuntimeError(f"Failed to generate answer from Ollama: {e}") from e

    return {
        "llm_model_name": llm_model_name,
        "llm_context": llm_context,
        "llm_prompt": llm_prompt,
        "llm_response_raw": llm_response_raw,
        "llm_answer": llm_answer,
        "llm_tokens_used": llm_tokens_used,
        "llm_latency_ms": llm_latency_ms,
    }


async def generate_llm_answer_async(
    query_text: str,
    rerank_chunks: List[Dict[str, Any]],
    llm_model_name: str = "llama3.2:1b",
    ollama_url: str = "http://localhost:11434/api/generate",
    system_prompt: str = "You are a helpful AI assistant. Answer the user question accurately using the provided context chunks.",
) -> Dict[str, Any]:
    """
    Assembles a prompt from the query and re-ranked context chunks,
    sends it to a local Ollama instance running Llama 3.2 1B (asynchronously),
    and returns LLM generation outputs conforming to the naming conventions.
    """
    # 1. Assemble llm_context
    context_parts = []
    for i, chunk in enumerate(rerank_chunks):
        text = chunk.get("text") or chunk.get("content") or ""
        context_parts.append(f"--- Context {i+1} ---\n{text}")
    llm_context = "\n\n".join(context_parts)

    # 2. Assemble llm_prompt
    llm_prompt = (
        f"System Prompt: {system_prompt}\n\n"
        f"Context:\n{llm_context}\n\n"
        f"User Question: {query_text}\n"
        f"Response:"
    )

    start_time = time.perf_counter()

    payload = {
        "model": llm_model_name,
        "prompt": llm_prompt,
        "stream": False,
    }

    try:
        logger.info(f"Sending prompt asynchronously to Ollama model '{llm_model_name}' at {ollama_url}...")
        async with httpx.AsyncClient() as client:
            response = await client.post(ollama_url, json=payload, timeout=60.0)
        response.raise_for_status()
        result = response.json()

        llm_response_raw = result.get("response", "")
        llm_answer = llm_response_raw.strip()

        # Extract tokens and latency from Ollama response, falling back to estimations if missing
        prompt_tokens = result.get("prompt_eval_count", 0)
        completion_tokens = result.get("eval_count", 0)
        llm_tokens_used = prompt_tokens + completion_tokens

        total_duration_ns = result.get("total_duration")
        if total_duration_ns:
            llm_latency_ms = total_duration_ns / 1_000_000.0
        else:
            llm_latency_ms = (time.perf_counter() - start_time) * 1000.0

    except Exception as e:
        logger.error(f"Error communicating asynchronously with Ollama: {e}")
        raise RuntimeError(f"Failed to generate answer asynchronously from Ollama: {e}") from e

    return {
        "llm_model_name": llm_model_name,
        "llm_context": llm_context,
        "llm_prompt": llm_prompt,
        "llm_response_raw": llm_response_raw,
        "llm_answer": llm_answer,
        "llm_tokens_used": llm_tokens_used,
        "llm_latency_ms": llm_latency_ms,
    }
