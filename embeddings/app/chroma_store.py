import uuid
import logging
from typing import List, Dict, Tuple, Optional, Any
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)

DEFAULT_DB_COLLECTION_NAME = "chatbot_docs"
DEFAULT_DB_PATH = "chroma_db"

def db_init_client(
    db_collection_name: str = DEFAULT_DB_COLLECTION_NAME,
    db_path: str = DEFAULT_DB_PATH,
    ephemeral: bool = False
) -> Tuple[ClientAPI, Collection]:
    """
    Initializes the ChromaDB client and gets or creates the active collection.
    
    Adheres to naming conventions:
    - db_collection_name (str)
    - db_client (ClientAPI/ChromaClient)
    - db_collection (Collection)
    
    Returns:
        tuple: (db_client, db_collection)
    """
    logger.info(f"Initializing ChromaDB client (ephemeral={ephemeral})...")
    
    if ephemeral:
        db_client = chromadb.EphemeralClient()
    else:
        db_client = chromadb.PersistentClient(path=db_path)
        
    db_collection = db_client.get_or_create_collection(name=db_collection_name)
    logger.info(f"Connected to ChromaDB collection: '{db_collection_name}' (Total documents: {db_collection.count()})")
    
    return db_client, db_collection

def db_add_documents(
    db_collection: Collection,
    chunk_list: List[str],
    emb_vectors: List[List[float]],
    chunk_meta_list: Optional[List[Dict[str, Any]]] = None,
    db_doc_ids: Optional[List[str]] = None
) -> List[str]:
    """
    Stores document chunks and their embeddings into ChromaDB collection.
    
    Adheres to naming conventions:
    - db_collection (Collection)
    - chunk_list (list[str])
    - emb_vectors (list[list[float]])
    - chunk_meta_list (list[dict])
    - db_doc_ids (list[str])
    
    Returns:
        list[str]: db_doc_ids assigned to stored chunks
    """
    if not chunk_list:
        logger.warning("No chunks to add to ChromaDB.")
        return []
        
    if len(chunk_list) != len(emb_vectors):
        raise ValueError(f"Mismatch between number of chunks ({len(chunk_list)}) and embeddings ({len(emb_vectors)})")
        
    # Generate unique IDs if not provided
    if db_doc_ids is None:
        db_doc_ids = [f"chunk_{uuid.uuid4().hex}" for _ in range(len(chunk_list))]
    elif len(db_doc_ids) != len(chunk_list):
        raise ValueError(f"Mismatch between chunk count ({len(chunk_list)}) and provided db_doc_ids ({len(db_doc_ids)})")
        
    # Standardize and clean metadata for Chroma (must be simple types: str, int, float, bool)
    cleaned_metadatas = []
    for idx, chunk in enumerate(chunk_list):
        meta = {}
        if chunk_meta_list and idx < len(chunk_meta_list):
            provided_meta = chunk_meta_list[idx]
            for k, v in provided_meta.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v) # Serialize complex types
        # Ensure at least an index and text snippet or key exists
        meta["index"] = meta.get("index", idx)
        cleaned_metadatas.append(meta)
        
    logger.info(f"Adding {len(chunk_list)} chunks to ChromaDB collection...")
    db_collection.add(
        ids=db_doc_ids,
        embeddings=emb_vectors,
        documents=chunk_list,
        metadatas=cleaned_metadatas
    )
    
    logger.info(f"Successfully added {len(chunk_list)} chunks. Collection count is now {db_collection.count()}.")
    return db_doc_ids
