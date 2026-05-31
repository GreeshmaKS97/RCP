import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import httpx
from app.llm import generate_llm_answer, generate_llm_answer_async


class TestLLMGeneration(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.query_text = "What is the primary embedding model used?"
        self.rerank_chunks = [
            {
                "id": "doc1",
                "text": "Embedding generation uses model 'text-embedding-3-small' to yield vectors stored in ChromaDB.",
            },
            {"id": "doc2", "content": "Vector database collections are queried using distance metrics."},
        ]
        self.mock_ollama_response = {
            "model": "llama3.2:1b",
            "response": "According to the context, the primary embedding model used is 'text-embedding-3-small'.",
            "prompt_eval_count": 40,
            "eval_count": 15,
            "total_duration": 250000000,  # 250 ms in nanoseconds
            "done": True,
        }

    @patch("httpx.post")
    def test_generate_llm_answer_sync_success(self, mock_post):
        # Configure mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_ollama_response
        mock_post.return_value = mock_response

        # Execute
        result = generate_llm_answer(
            query_text=self.query_text, rerank_chunks=self.rerank_chunks, llm_model_name="llama3.2:1b"
        )

        # Asserts
        self.assertEqual(result["llm_model_name"], "llama3.2:1b")
        self.assertIn("text-embedding-3-small", result["llm_context"])
        self.assertIn("System Prompt:", result["llm_prompt"])
        self.assertEqual(result["llm_response_raw"], self.mock_ollama_response["response"])
        self.assertEqual(result["llm_answer"], self.mock_ollama_response["response"])
        self.assertEqual(result["llm_tokens_used"], 55)  # 40 + 15
        self.assertEqual(result["llm_latency_ms"], 250.0)  # 250000000 ns / 1_000_000

        # Verify post arguments
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "http://localhost:11434/api/generate")
        payload = kwargs["json"]
        self.assertEqual(payload["model"], "llama3.2:1b")
        self.assertFalse(payload["stream"])

    @patch("httpx.AsyncClient")
    async def test_generate_llm_answer_async_success(self, mock_async_client_cls):
        # Configure mock for async client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_ollama_response

        # Async method mocking
        mock_client.post = AsyncMock(return_value=mock_response)

        # Configure AsyncClient mock context manager
        mock_async_client = mock_async_client_cls.return_value
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=None)

        # Execute
        result = await generate_llm_answer_async(
            query_text=self.query_text, rerank_chunks=self.rerank_chunks, llm_model_name="llama3.2:1b"
        )

        # Asserts
        self.assertEqual(result["llm_model_name"], "llama3.2:1b")
        self.assertEqual(result["llm_answer"], self.mock_ollama_response["response"])
        self.assertEqual(result["llm_tokens_used"], 55)
        self.assertEqual(result["llm_latency_ms"], 250.0)

        # Verify call
        mock_client.post.assert_called_once()

    @patch("app.llm.logger")
    @patch("httpx.post")
    def test_generate_llm_answer_sync_failure(self, mock_post, mock_logger):
        mock_post.side_effect = httpx.RequestError("Connection refused")
        with self.assertRaises(RuntimeError):
            generate_llm_answer(query_text=self.query_text, rerank_chunks=self.rerank_chunks)

    def test_live_ollama_integration(self):
        """
        A live test against local Ollama.
        If local Ollama is running and has llama3.2:1b, this tests actual inference.
        If not running, it skips gracefully to avoid failing CI/tests in headless environments.
        """
        import socket

        # Check if Ollama port is open
        try:
            with socket.create_connection(("127.0.0.1", 11434), timeout=1.0):
                port_open = True
        except Exception:
            port_open = False

        if not port_open:
            self.skipTest("Ollama is not running locally. Skipping live integration test.")

        # If port is open, check if model llama3.2:1b is available
        try:
            import urllib.request
            import json

            response = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2.0)
            data = json.loads(response.read().decode("utf-8"))
            models = [m["name"] for m in data.get("models", [])]
            if "llama3.2:1b" not in models and "llama3.2:latest" not in models:
                self.skipTest("Required llama3.2 model is not downloaded in Ollama. Skipping live test.")

            model_to_use = "llama3.2:1b" if "llama3.2:1b" in models else "llama3.2:latest"
        except Exception as e:
            self.skipTest(f"Failed to query Ollama models: {e}. Skipping live test.")

        # Run real inference
        try:
            result = generate_llm_answer(
                query_text="What color is the sky?",
                rerank_chunks=[{"text": "The sky is blue during a clear day."}],
                llm_model_name=model_to_use,
            )
            self.assertIn("blue", result["llm_answer"].lower())
            self.assertEqual(result["llm_model_name"], model_to_use)
            self.assertGreater(result["llm_tokens_used"], 0)
            self.assertGreater(result["llm_latency_ms"], 0)
        except Exception as e:
            self.fail(f"Live Ollama test failed with error: {e}")


if __name__ == "__main__":
    unittest.main()
