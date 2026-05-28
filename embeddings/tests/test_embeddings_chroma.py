import os
import pytest
import logging
from unittest.mock import MagicMock, patch
from app.embeddings import emb_generate_embeddings
from app.chroma_store import db_init_client, db_add_documents
from app.retriever import query_similarity_search

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock function to generate deterministic, semantically-aware embeddings for testing
def get_mock_openai_response(input_chunks, model_name):
    # Vocabulary of key terms to simulate semantic similarity in tests
    vocab = ["google", "antigravity", "chatbot", "chromadb", "vector", "database", "similarity", "search"]
    
    data_list = []
    for text in input_chunks:
        text_lower = text.lower()
        vector = [0.0] * 1536
        
        # Set values for vocab terms if they appear in the text
        for idx, word in enumerate(vocab):
            if word in text_lower:
                vector[idx] = 1.0
                
        # Fill remaining dimensions with tiny deterministic noise based on hash
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        for j in range(len(vocab), 1536):
            byte_idx = (j * 2) % len(h)
            val = (h[byte_idx] + h[(byte_idx + 1) % len(h)] * 256) / 65535.0
            vector[j] = (val - 0.5) * 0.05
            
        # Normalize vector to unit length
        import math
        sq_sum = sum(x*x for x in vector)
        magnitude = math.sqrt(sq_sum) if sq_sum > 0 else 1.0
        normalized_vector = [x / magnitude for x in vector]
        
        mock_item = MagicMock()
        mock_item.embedding = normalized_vector
        data_list.append(mock_item)
        
    mock_response = MagicMock()
    mock_response.data = data_list
    return mock_response

@patch("openai.OpenAI")
def test_embeddings_and_search_pipeline(mock_openai_class):
    logger.info("Starting integration test for embedding and search pipeline...")
    
    # Set dummy API key for testing
    os.environ["OPENAI_API_KEY"] = "test-dummy-key"
    
    # Configure mock OpenAI client
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.embeddings.create.side_effect = lambda input, model: get_mock_openai_response(input, model)
    
    # 1. Define sample text chunks representing document parts
    chunk_list = [
        "Google Antigravity is a coding assistant designed by the Google DeepMind team.",
        "Customizable chatbots allow users to upload documents, parse them, and perform semantic queries.",
        "ChromaDB is an open-source vector database used for storing embeddings and performing fast similarity search."
    ]
    
    chunk_meta_list = [
        {"source": "antigravity_docs", "page": 1, "topic": "AI"},
        {"source": "chatbot_manual", "page": 3, "topic": "Architecture"},
        {"source": "chroma_ref", "page": 12, "topic": "Database"}
    ]
    
    # 2. Generate embeddings using mocked text-embedding-3-small
    emb_vectors, emb_dim = emb_generate_embeddings(
        chunk_list=chunk_list,
        emb_model_name="text-embedding-3-small",
        emb_batch_size=2
    )
    
    assert mock_client.embeddings.create.called
    assert isinstance(emb_vectors, list), "Embeddings must be a list"
    assert len(emb_vectors) == len(chunk_list), "Number of embeddings must match number of chunks"
    assert len(emb_vectors[0]) == emb_dim, "Vector dimension must match emb_dim"
    assert emb_dim == 1536, "text-embedding-3-small dimension must be 1536"
    
    logger.info(f"Embeddings validation passed. Dimension: {emb_dim}")
    
    # 3. Initialize ChromaDB (use ephemeral client for clean testing)
    db_client, db_collection = db_init_client(
        db_collection_name="test_collection",
        ephemeral=True
    )
    
    # Check count is zero initially
    assert db_collection.count() == 0, "Ephemeral database must be empty at start"
    
    # 4. Add documents to ChromaDB
    db_doc_ids = db_add_documents(
        db_collection=db_collection,
        chunk_list=chunk_list,
        emb_vectors=emb_vectors,
        chunk_meta_list=chunk_meta_list
    )
    
    assert isinstance(db_doc_ids, list), "Stored doc IDs must be a list"
    assert len(db_doc_ids) == len(chunk_list), "Must return one doc ID per chunk"
    assert db_collection.count() == len(chunk_list), "Collection count should match inserted documents"
    
    logger.info("ChromaDB storage validation passed.")
    
    # 5. Query and Similarity Search
    query_text = "What is ChromaDB and how is it used?"
    query_top_k = 2
    
    # Reset mock call count to track query embedding call
    mock_client.embeddings.create.reset_mock()
    
    # Execute search using our embedder (which is also mocked)
    query_emb, query_results = query_similarity_search(
        db_collection=db_collection,
        query_text=query_text,
        embedder_fn=lambda q: emb_generate_embeddings(q, emb_model_name="text-embedding-3-small"),
        query_top_k=query_top_k
    )
    
    assert mock_client.embeddings.create.called
    
    # Assert query results
    assert isinstance(query_emb, list), "Query embedding must be a list"
    assert len(query_emb) == emb_dim, "Query embedding dimension must match emb_dim"
    assert isinstance(query_results, list), "Query results must be a list"
    assert len(query_results) == query_top_k, f"Query results count should equal top_k ({query_top_k})"
    
    # Assert fields are present in each result
    for res in query_results:
        assert "id" in res, "Result must contain 'id'"
        assert "document" in res, "Result must contain 'document'"
        assert "metadata" in res, "Result must contain 'metadata'"
        assert "distance" in res, "Result must contain 'distance'"
        
    # The most relevant document should be about ChromaDB (third chunk)
    first_doc = query_results[0]["document"]
    logger.info(f"Top search result: {first_doc}")
    assert "ChromaDB" in first_doc, "Search result should match the ChromaDB chunk"
    assert query_results[0]["metadata"]["topic"] == "Database", "Metadata should match corresponding chunk metadata"
    
    logger.info("Similarity search validation passed successfully!")

if __name__ == "__main__":
    test_embeddings_and_search_pipeline()

