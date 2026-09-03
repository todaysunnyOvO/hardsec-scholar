"""Online embedding adapters."""

from typing import Any

from openai import OpenAI


class OpenAIEmbeddingProvider:
    """Generate embeddings through an OpenAI-compatible API."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        """Initialize a reusable client and embedding model name."""
        if not model.strip():
            raise ValueError("Embedding model must not be empty")
        if not api_key.strip() and client is None:
            raise ValueError("Embedding API key must not be empty")
        self.model = model
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents while preserving API response index ordering."""
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]

    def embed_query(self, text: str) -> list[float]:
        """Embed one non-empty query."""
        if not text.strip():
            raise ValueError("Query must not be empty")
        return self.embed_documents([text])[0]
