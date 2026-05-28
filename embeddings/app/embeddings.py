import os
import logging
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Default variables as per requirements
DEFAULT_EMB_MODEL = "text-embedding-3-small"
DEFAULT_EMB_BATCH_SIZE = 32

def emb_generate_embeddings(
    chunk_list: List[str],
    emb_model_name: str = DEFAULT_EMB_MODEL,
    emb_batch_size: int = DEFAULT_EMB_BATCH_SIZE,
    api_key: str = None
) -> Tuple[List[List[float]], int]:
    """
    Generates embedding vectors for a list of text chunks.
    
    Adheres to naming conventions:
    - emb_model_name (str)
    - emb_vectors (list[list[float]])
    - emb_dim (int)
    - emb_batch_size (int)
    
    Returns:
        tuple: (emb_vectors, emb_dim)
    """
    if not chunk_list:
        return [], 0

    emb_vectors: List[List[float]] = []
    emb_dim: int = 0
    
    openai_key = api_key or os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY is not set. A valid OpenAI API key is required to use 'text-embedding-3-small'.")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        logger.info(f"Generating embeddings using OpenAI model '{emb_model_name}'...")
        
        # Batch the chunk_list
        for i in range(0, len(chunk_list), emb_batch_size):
            batch = chunk_list[i : i + emb_batch_size]
            response = client.embeddings.create(
                input=batch,
                model=emb_model_name
            )
            for data in response.data:
                emb_vectors.append(data.embedding)
        
        if emb_vectors:
            emb_dim = len(emb_vectors[0])
            logger.info(f"Successfully generated {len(emb_vectors)} embeddings with dimension {emb_dim} via OpenAI.")
            return emb_vectors, emb_dim
        else:
            raise ValueError("No embeddings returned from OpenAI API.")
    except Exception as e:
        logger.error(f"Failed to generate embeddings using OpenAI API model '{emb_model_name}': {e}")
        raise

