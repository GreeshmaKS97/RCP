#import os
#os.environ["TRANSFORMERS_OFFLINE"] = "1"
import time
import logging
from typing import Dict, List, Tuple, Any
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

_MODEL_CACHE = {}


def get_model_and_tokenizer(model_name: str) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Loads and caches the cross-encoder model and tokenizer on CPU."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if model_name not in _MODEL_CACHE:
        logger.info(f"Loading reranking model '{model_name}' on CPU...")
        start_time = time.time()

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()

        _MODEL_CACHE[model_name] = (model, tokenizer)
        logger.info(f"Loaded '{model_name}' in {time.time() - start_time:.2f}s.")

    return _MODEL_CACHE[model_name]


def clear_model_cache() -> None:
    """Clears cached models and tokenizers from memory."""
    _MODEL_CACHE.clear()
    logger.info("Reranker model cache cleared.")


def rerank_query_results(
    query_text: str,
    query_results: List[Dict[str, Any]],
    rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    rerank_top_n: int = 5,
) -> Tuple[List[float], List[Dict[str, Any]]]:
    """
    Reranks retrieved chunks using a Cross-Encoder model.

    Args:
        query_text: The user's query string.
        query_results: List of retrieved chunk dicts. Each must have a 'text'
                       (primary) or 'content' (fallback) string key.
        rerank_model_name: HuggingFace model identifier for the cross-encoder.
        rerank_top_n: Number of top chunks to return.

    Returns:
        rerank_scores: Relevance scores for the top-N chunks.
        rerank_chunks: Top-N chunks sorted by descending relevance score.
    """
    if not query_results:
        logger.warning("Empty query_results provided to reranker.")
        return [], []

    import torch

    model, tokenizer = get_model_and_tokenizer(rerank_model_name)

    pairs = []
    valid_chunks = []

    for chunk in query_results:
        chunk_text = chunk.get("text") or chunk.get("content")
        if isinstance(chunk_text, str):
            pairs.append([query_text, chunk_text])
            valid_chunks.append(chunk)
        else:
            logger.warning("Chunk missing valid 'text' or 'content' key. Skipping.")

    if not pairs:
        logger.warning("No valid text chunks found in query_results for reranking.")
        return [], []

    inputs = tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")

    with torch.no_grad():
        scores = model(**inputs).logits.squeeze(-1)
        scores = [scores.item()] if scores.dim() == 0 else scores.tolist()

    scored_chunks = sorted(
        [(float(s), c) for s, c in zip(scores, valid_chunks)],
        key=lambda x: x[0],
        reverse=True,
    )

    rerank_scores = [item[0] for item in scored_chunks[:rerank_top_n]]
    rerank_chunks = [item[1] for item in scored_chunks[:rerank_top_n]]

    return rerank_scores, rerank_chunks