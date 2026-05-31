import sys
import os

# Add current directory to python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.llm import generate_llm_answer

def run_demo():
    print("=" * 60)
    print("         OLLAMA / LLAMA 3.2 DEMO CHAT TEST")
    print("=" * 60)
    
    query = "What is the primary vector database mentioned in the context?"
    chunks = [
        {"text": "We store document embeddings in ChromaDB to perform fast vector search queries."},
        {"text": "The reranker uses a cross-encoder model to sort initial retrieve results from ChromaDB."}
    ]
    
    print(f"\n[Test Query]: {query}")
    print("\n[Provided Context Chunks]:")
    for i, chunk in enumerate(chunks):
        print(f"  - Chunk {i+1}: {chunk['text']}")
        
    print("\nSending prompt to local Ollama running llama3.2:1b...")
    
    try:
        result = generate_llm_answer(
            query_text=query,
            rerank_chunks=chunks,
            llm_model_name="llama3.2:1b"
        )
        
        print("\n" + "=" * 60)
        print("                 GENERATED ANSWER")
        print("=" * 60)
        print(result["llm_answer"])
        print("=" * 60)
        print(f"Model Used:      {result['llm_model_name']}")
        print(f"Tokens Used:     {result['llm_tokens_used']}")
        print(f"Latency (ms):    {result['llm_latency_ms']:.2f} ms")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] Failed to communicate with Ollama: {e}")
        print("Please check that Ollama is running and that 'llama3.2:1b' is pulled.")

if __name__ == "__main__":
    run_demo()
