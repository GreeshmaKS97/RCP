import logging
from typing import List, Dict, Tuple, Callable, Any
from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)

DEFAULT_QUERY_TOP_K = 5

def query_similarity_search(
    db_collection: Collection,
    query_text: str,
    embedder_fn: Callable[[List[str]], Tuple[List[List[float]], int]],
    query_top_k: int = DEFAULT_QUERY_TOP_K
) -> Tuple[List[float], List[Dict[str, Any]]]:
    """
    Performs similarity search on ChromaDB for a given query text.
    
    Adheres to naming conventions:
    - query_text (str)
    - query_emb (list[float])
    - query_top_k (int)
    - query_results (list[dict])
    
    Returns:
        tuple: (query_emb, query_results)
    """
    logger.info(f"Received similarity search query: '{query_text}' with top_k={query_top_k}")
    
    # 1. Generate query embedding
    emb_vectors, emb_dim = embedder_fn([query_text])
    if not emb_vectors:
        raise ValueError("Failed to generate embedding for the search query.")
        
    query_emb: List[float] = emb_vectors[0]
    
    # 2. Perform ChromaDB query
    raw_response = db_collection.query(
        query_embeddings=[query_emb],
        n_results=query_top_k
    )
    
    # 3. Format results into query_results list[dict]
    query_results: List[Dict[str, Any]] = []
    
    # Process ChromaDB structure
    # Chroma returns lists corresponding to the batched query embeddings. 
    # Since we query with exactly one vector, we look at index 0.
    ids = raw_response.get("ids", [[]])[0]
    documents = raw_response.get("documents", [[]])[0]
    metadatas = raw_response.get("metadatas", [[]])[0]
    distances = raw_response.get("distances", [[]])[0]
    
    # Ensure lists exist and are of matching lengths
    for idx in range(len(ids)):
        result_item = {
            "id": ids[idx],
            "document": documents[idx] if idx < len(documents) else "",
            "metadata": metadatas[idx] if metadatas and idx < len(metadatas) else {},
            "distance": distances[idx] if distances and idx < len(distances) else 0.0
        }
        query_results.append(result_item)
        
    logger.info(f"Retrieved {len(query_results)} matches from ChromaDB.")
    return query_emb, query_results
