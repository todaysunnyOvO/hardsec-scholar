"""Compose local storage, retrieval, model, and Agent services for the API."""

from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import Protocol

import chromadb
from chromadb.errors import NotFoundError
from langchain_openai import ChatOpenAI

from hardsec_scholar.agent import (
    AgenticRAGWorkflow,
    AgentRun,
    StructuredAgentReasoner,
    TraceEvent,
)
from hardsec_scholar.api.history import ConversationRepository
from hardsec_scholar.config import AppSettings, RuntimeSettings, load_app_settings
from hardsec_scholar.domain import PaperStatus
from hardsec_scholar.domain.terminology import load_terminology
from hardsec_scholar.generation import StructuredAnswerGenerator
from hardsec_scholar.ingestion import PaperIngestionService
from hardsec_scholar.retrieval.embeddings import OpenAIEmbeddingProvider
from hardsec_scholar.retrieval.hybrid import HybridRetriever
from hardsec_scholar.retrieval.indexing import PaperIndexingService
from hardsec_scholar.retrieval.reranker import FlashRankReranker, IdentityReranker
from hardsec_scholar.retrieval.vector import ChromaVectorIndex
from hardsec_scholar.storage import PaperRepository


class ServiceConfigurationError(RuntimeError):
    """Report missing or unsupported runtime model configuration."""


class ApplicationServices(Protocol):
    """Describe API-facing operations that can be replaced in tests."""

    settings: AppSettings
    repository: PaperRepository
    history: ConversationRepository
    ingestion: PaperIngestionService

    def index_paper(self, paper_id: str) -> int:
        """Create or replace vectors for one parsed paper."""
        ...

    def delete_paper(self, paper_id: str) -> bool:
        """Delete vectors, metadata, chunks, and the controlled PDF copy."""
        ...

    def run_agent(self, question: str, paper_ids: list[str]) -> AgentRun:
        """Run one bounded local-paper Agent question."""
        ...

    def stream_agent(
        self, question: str, paper_ids: list[str]
    ) -> Iterator[TraceEvent | AgentRun]:
        """Stream safe graph events followed by the completed run."""
        ...


class LocalApplicationServices:
    """Lazily create online model clients while keeping paper CRUD offline."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        runtime: RuntimeSettings | None = None,
    ) -> None:
        """Initialize local repositories without requiring API credentials."""
        self.settings = settings or load_app_settings()
        self.runtime = runtime or RuntimeSettings()
        self.repository = PaperRepository(self.settings.paths.database)
        self.history = ConversationRepository(self.settings.paths.database)
        self.ingestion = PaperIngestionService(self.settings, self.repository)
        self.repository.initialize()
        self.history.initialize()

    def _embedding_provider(self) -> OpenAIEmbeddingProvider:
        if self.runtime.embedding_provider.casefold() != "openai":
            raise ServiceConfigurationError(
                "Only an OpenAI-compatible embedding provider is configured."
            )
        secret = self.runtime.embedding_api_key or self.runtime.llm_api_key
        if secret is None:
            raise ServiceConfigurationError(
                "Set EMBEDDING_API_KEY or LLM_API_KEY before indexing papers."
            )
        if not self.runtime.embedding_model.strip():
            raise ServiceConfigurationError(
                "Set EMBEDDING_MODEL before indexing or searching papers."
            )
        return OpenAIEmbeddingProvider(
            model=self.runtime.embedding_model,
            api_key=secret.get_secret_value(),
            base_url=self.runtime.embedding_base_url,
        )

    def _vector_index(self) -> ChromaVectorIndex:
        return ChromaVectorIndex(
            self.settings.paths.chroma,
            self._embedding_provider(),
            collection_name=self.settings.retrieval.collection_name,
        )

    def index_paper(self, paper_id: str) -> int:
        """Embed and index one parsed paper with current runtime credentials."""
        return PaperIndexingService(self.repository, self._vector_index()).index_paper(
            paper_id
        )

    def delete_paper(self, paper_id: str) -> bool:
        """Delete vector state when available, then local metadata and PDF."""
        paper = self.repository.get_paper(paper_id)
        if paper is None:
            return False
        try:
            client = chromadb.PersistentClient(path=str(self.settings.paths.chroma))
            collection = client.get_collection(
                self.settings.retrieval.collection_name,
                embedding_function=None,
            )
            collection.delete(where={"paper_id": paper_id})
        except NotFoundError:
            # A missing local collection means there are no vectors to orphan.
            pass
        deleted = self.repository.delete_paper(paper_id)
        if deleted:
            controlled_root = self.settings.paths.papers.resolve()
            source = Path(paper.file_path).resolve()
            if source.is_file() and source.parent == controlled_root:
                source.unlink()
        return deleted

    @cached_property
    def workflow(self) -> AgenticRAGWorkflow:
        """Build the online Agent graph only when the first question arrives."""
        if self.runtime.llm_provider.casefold() != "openai":
            raise ServiceConfigurationError(
                "The current API composition supports an OpenAI-compatible LLM."
            )
        if self.runtime.llm_api_key is None:
            raise ServiceConfigurationError("Set LLM_API_KEY before asking questions.")
        if not self.runtime.llm_model.strip():
            raise ServiceConfigurationError("Set LLM_MODEL before asking questions.")
        extra_body = None
        if self.runtime.llm_base_url and "deepseek.com" in self.runtime.llm_base_url:
            # DeepSeek V4 enables thinking by default, but thinking mode rejects the
            # forced tool choice LangChain uses for schema-constrained output.
            extra_body = {"thinking": {"type": "disabled"}}
        model = ChatOpenAI(
            model=self.runtime.llm_model,
            api_key=self.runtime.llm_api_key,
            base_url=self.runtime.llm_base_url,
            temperature=0,
            extra_body=extra_body,
        )
        reranker = (
            FlashRankReranker(
                model_name=self.settings.retrieval.reranker_model,
                cache_dir=self.settings.retrieval.reranker_cache,
            )
            if self.settings.retrieval.reranker_enabled
            else IdentityReranker()
        )
        retriever = HybridRetriever(
            repository=self.repository,
            vector_index=self._vector_index(),
            reranker=reranker,
            settings=self.settings.retrieval,
        )
        return AgenticRAGWorkflow(
            retriever=retriever,
            generator=StructuredAnswerGenerator(model),
            reasoner=StructuredAgentReasoner(model),
            terminology=load_terminology(),
            agent_settings=self.settings.agent,
            generation_settings=self.settings.generation,
        )

    def run_agent(self, question: str, paper_ids: list[str]) -> AgentRun:
        """Execute one Agent question against all or selected indexed papers."""
        selected = self._selected_papers(paper_ids)
        return self.workflow.run(question, paper_ids=selected)

    def stream_agent(
        self, question: str, paper_ids: list[str]
    ) -> Iterator[TraceEvent | AgentRun]:
        """Stream one Agent run for the local API's SSE endpoint."""
        selected = self._selected_papers(paper_ids)
        return self.workflow.stream(question, paper_ids=selected)

    def _selected_papers(self, paper_ids: list[str]) -> list[str]:
        """Resolve an empty selection to every currently indexed paper."""
        selected = paper_ids or [
            paper.id
            for paper in self.repository.list_papers()
            if paper.status is PaperStatus.INDEXED
        ]
        if not selected:
            raise ServiceConfigurationError(
                "Index at least one paper before asking a research question."
            )
        return selected
