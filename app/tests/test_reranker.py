import unittest
from app.reranker import rerank_query_results, clear_model_cache

class TestReranker(unittest.TestCase):
    def setUp(self):
        self.query_text = "What is the capital of France?"
        # Simple sample documents
        self.query_results = [
            {"id": "doc1", "text": "The capital of France is Paris. It is a major European city and a global center for art, fashion, gastronomy and culture."},
            {"id": "doc2", "text": "London is the capital and largest city of England and the United Kingdom."},
            {"id": "doc3", "text": "Berlin is the capital and largest city of Germany by both area and population."},
            {"id": "doc4", "text": "Rome is the capital city of Italy. It is also the country's most populated commune."}
        ]

    def test_basic_reranking(self):
        """Test that the reranker returns the correct types, scores, and sorted order."""
        model_name = "cross-encoder/ms-marco-tinybert-L-2-v2"
        top_n = 2
        
        rerank_scores, rerank_chunks = rerank_query_results(
            query_text=self.query_text,
            query_results=self.query_results,
            rerank_model_name=model_name,
            rerank_top_n=top_n
        )
        
        # Check types
        self.assertIsInstance(rerank_scores, list)
        self.assertIsInstance(rerank_chunks, list)
        
        # Check lengths match top_n
        self.assertEqual(len(rerank_scores), top_n)
        self.assertEqual(len(rerank_chunks), top_n)
        
        # Check list elements types
        for score in rerank_scores:
            self.assertIsInstance(score, float)
        for chunk in rerank_chunks:
            self.assertIsInstance(chunk, dict)
            # Verify no score pollution in chunk dict
            self.assertNotIn("_rerank_score", chunk)

        # Check sorting order: highest score first
        for i in range(len(rerank_scores) - 1):
            self.assertGreaterEqual(rerank_scores[i], rerank_scores[i+1])

        # For this specific query, Paris (doc1) should be the top-ranked item
        self.assertEqual(rerank_chunks[0]["id"], "doc1")

    def test_single_chunk(self):
        """Test that a single chunk doesn't cause a 0-dim tensor crash."""
        rerank_scores, rerank_chunks = rerank_query_results(
            query_text=self.query_text,
            query_results=[{"id": "doc1", "text": "The capital of France is Paris."}],
            rerank_model_name="cross-encoder/ms-marco-tinybert-L-2-v2",
            rerank_top_n=1
        )
        self.assertEqual(len(rerank_scores), 1)
        self.assertEqual(len(rerank_chunks), 1)
        self.assertIsInstance(rerank_scores[0], float)
        self.assertIsInstance(rerank_chunks[0], dict)
    def test_empty_query_results(self):
        """Test that passing an empty list returns empty lists."""
        rerank_scores, rerank_chunks = rerank_query_results(
            query_text=self.query_text,
            query_results=[],
            rerank_top_n=3
        )
        self.assertEqual(rerank_scores, [])
        self.assertEqual(rerank_chunks, [])

    def test_different_text_keys(self):
        """Test that the reranker handles 'text' (primary) and 'content' (fallback) keys."""
        mixed_results = [
            {"id": "doc1", "text": "The capital of France is Paris."},
            {"id": "doc2", "content": "London is the capital and largest city of England."},
            {"id": "doc3", "invalid_key": "Berlin is the capital of Germany."}
        ]
        
        model_name = "cross-encoder/ms-marco-tinybert-L-2-v2"
        
        rerank_scores, rerank_chunks = rerank_query_results(
            query_text=self.query_text,
            query_results=mixed_results,
            rerank_model_name=model_name,
            rerank_top_n=2
        )
        
        # Should only parse doc1 and doc2, doc3 is skipped.
        self.assertEqual(len(rerank_chunks), 2)
        self.assertEqual(rerank_chunks[0]["id"], "doc1")
        self.assertEqual(rerank_chunks[1]["id"], "doc2")

    def test_no_valid_text_fields(self):
        """Test that chunks without any recognizable text keys are skipped."""
        bad_results = [
            {"id": "doc1", "unknown_field": "This cannot be processed."},
            {"id": "doc2", "another_unknown_field": "Nor this one."}
        ]
        
        rerank_scores, rerank_chunks = rerank_query_results(
            query_text=self.query_text,
            query_results=bad_results,
            rerank_top_n=2
        )
        
        self.assertEqual(rerank_scores, [])
        self.assertEqual(rerank_chunks, [])

    def test_clear_model_cache(self):
        """Test that clear_model_cache empties the model cache."""
        from app.reranker import _MODEL_CACHE
        model_name = "cross-encoder/ms-marco-tinybert-L-2-v2"
        
        # Load a model to populate the cache
        _, _ = rerank_query_results("test", [{"text": "dummy"}], rerank_model_name=model_name)
        self.assertIn(model_name, _MODEL_CACHE)
        
        # Clear the cache
        clear_model_cache()
        self.assertEqual(len(_MODEL_CACHE), 0)

if __name__ == '__main__':
    unittest.main()
