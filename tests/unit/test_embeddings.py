"""Tests for the OpenAI-compatible embedding adapter."""

from types import SimpleNamespace

from hardsec_scholar.retrieval.embeddings import OpenAIEmbeddingProvider


class _FakeEmbeddings:
    def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        assert model == "test-embedding"
        data = [
            SimpleNamespace(index=index, embedding=[float(index), float(len(text))])
            for index, text in reversed(list(enumerate(input)))
        ]
        return SimpleNamespace(data=data)


def test_embedding_adapter_restores_api_index_order() -> None:
    client = SimpleNamespace(embeddings=_FakeEmbeddings())
    provider = OpenAIEmbeddingProvider(
        model="test-embedding", api_key="", client=client
    )

    assert provider.embed_documents(["a", "abcd"]) == [[0.0, 1.0], [1.0, 4.0]]
