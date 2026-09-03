"""FastAPI application exposing local paper and Agentic RAG workflows."""

import json
import tempfile
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from hardsec_scholar.agent import AgentRun, TraceEvent
from hardsec_scholar.api.history import (
    ConversationRecord,
    ConversationSummary,
    StoredRun,
)
from hardsec_scholar.api.models import (
    MessageRequest,
    MessageResponse,
    PaperPatch,
    PaperUploadResponse,
    PaperView,
    ReindexResponse,
)
from hardsec_scholar.api.services import (
    ApplicationServices,
    LocalApplicationServices,
    ServiceConfigurationError,
)
from hardsec_scholar.domain import Evidence
from hardsec_scholar.ingestion import DuplicatePaperError


def _paper_view(services: ApplicationServices, paper_id: str) -> PaperView:
    paper = services.repository.get_paper(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return PaperView(
        paper=paper,
        chunk_count=len(services.repository.get_chunks(paper_id)),
    )


def create_app(services: ApplicationServices | None = None) -> FastAPI:
    """Create an injectable local API application."""
    active = services or LocalApplicationServices()
    app = FastAPI(
        title="HardSec Scholar API",
        version="0.1.0",
        description="Local Agentic RAG for hardware-security research papers.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "mode": "local"}

    @app.get("/api/papers", response_model=list[PaperView])
    def list_papers() -> list[PaperView]:
        return [
            PaperView(
                paper=paper,
                chunk_count=len(active.repository.get_chunks(paper.id)),
            )
            for paper in active.repository.list_papers()
        ]

    @app.get("/api/papers/{paper_id}", response_model=PaperView)
    def get_paper(paper_id: str) -> PaperView:
        return _paper_view(active, paper_id)

    @app.post(
        "/api/papers",
        response_model=PaperUploadResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def upload_paper(file: UploadFile = File(...)) -> PaperUploadResponse:
        suffix = Path(file.filename or "paper.pdf").suffix.casefold()
        if suffix != ".pdf":
            raise HTTPException(status_code=415, detail="Only PDF files are accepted")
        temp_root = active.settings.paths.papers.parent / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        max_bytes = active.settings.ingestion.max_file_size_mb * 1024 * 1024
        total_bytes = 0
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".pdf", dir=temp_root, delete=False
            ) as stream:
                temp_path = Path(stream.name)
                while content := file.file.read(1024 * 1024):
                    total_bytes += len(content)
                    if total_bytes > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                "PDF exceeds the configured "
                                f"{active.settings.ingestion.max_file_size_mb} MB limit"
                            ),
                        )
                    stream.write(content)
            result = active.ingestion.ingest(temp_path)
            indexed = False
            warning: str | None = None
            try:
                active.index_paper(result.paper.id)
                indexed = True
            except ServiceConfigurationError as exc:
                warning = str(exc)
            paper = active.repository.get_paper(result.paper.id) or result.paper
            return PaperUploadResponse(
                paper=paper,
                chunk_count=result.chunk_count,
                indexed=indexed,
                indexing_warning=warning,
            )
        except DuplicatePaperError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            file.file.close()
            if temp_path is not None and temp_path.is_file():
                temp_path.unlink()

    @app.patch("/api/papers/{paper_id}", response_model=PaperView)
    def update_paper(paper_id: str, patch: PaperPatch) -> PaperView:
        changes = patch.model_dump(exclude_unset=True)
        updated = active.repository.update_metadata(paper_id, changes)
        if updated is None:
            raise HTTPException(status_code=404, detail="Paper not found")
        return _paper_view(active, paper_id)

    @app.delete("/api/papers/{paper_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_paper(paper_id: str) -> None:
        if not active.delete_paper(paper_id):
            raise HTTPException(status_code=404, detail="Paper not found")

    @app.post("/api/papers/{paper_id}/reindex", response_model=ReindexResponse)
    def reindex_paper(paper_id: str) -> ReindexResponse:
        try:
            count = active.index_paper(paper_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Paper not found") from exc
        except ServiceConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return ReindexResponse(paper_id=paper_id, indexed_chunks=count)

    @app.get("/api/evidence/{evidence_id}", response_model=Evidence)
    def read_evidence(evidence_id: str) -> Evidence:
        chunk_id = evidence_id.removeprefix("evidence_")
        chunk = active.repository.get_chunk(chunk_id)
        if chunk is None:
            raise HTTPException(status_code=404, detail="Evidence not found")
        return Evidence(
            id=f"evidence_{chunk.id}",
            chunk_id=chunk.id,
            paper_id=chunk.paper_id,
            paper_title=chunk.title,
            section=chunk.section,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            text=chunk.text,
        )

    @app.post(
        "/api/conversations",
        response_model=ConversationRecord,
        status_code=status.HTTP_201_CREATED,
    )
    def create_conversation() -> ConversationRecord:
        return active.history.create_conversation()

    @app.get("/api/conversations", response_model=list[ConversationSummary])
    def list_conversations() -> list[ConversationSummary]:
        return active.history.list_conversations()

    @app.get("/api/conversations/{conversation_id}", response_model=ConversationRecord)
    def get_conversation(conversation_id: str) -> ConversationRecord:
        conversation = active.history.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation

    @app.delete(
        "/api/conversations/{conversation_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    def delete_conversation(conversation_id: str) -> None:
        if not active.history.delete_conversation(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")

    @app.post(
        "/api/conversations/{conversation_id}/messages",
        response_model=MessageResponse,
    )
    def send_message(conversation_id: str, request: MessageRequest) -> MessageResponse:
        if active.history.get_conversation(conversation_id) is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        try:
            run = active.run_agent(request.question, request.paper_ids)
        except ServiceConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        stored = active.history.save_run(conversation_id, request.question, run)
        return MessageResponse(run=stored)

    @app.post("/api/conversations/{conversation_id}/messages/stream")
    def stream_message(
        conversation_id: str, request: MessageRequest
    ) -> StreamingResponse:
        if active.history.get_conversation(conversation_id) is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        stream_run_id = f"run_{uuid4().hex}"
        try:
            agent_stream = active.stream_agent(request.question, request.paper_ids)
        except ServiceConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        def generate_live() -> Iterator[str]:
            try:
                for item in agent_stream:
                    if isinstance(item, TraceEvent):
                        payload = {
                            "run_id": stream_run_id,
                            **item.model_dump(mode="json"),
                        }
                        yield (f"event: {item.event}\ndata: {json.dumps(payload)}\n\n")
                    elif isinstance(item, AgentRun):
                        stored = active.history.save_run(
                            conversation_id,
                            request.question,
                            item,
                            run_id=stream_run_id,
                        )
                        payload = stored.model_dump(mode="json")
                        yield f"event: result\ndata: {json.dumps(payload)}\n\n"
            except Exception:
                payload = {
                    "run_id": stream_run_id,
                    "event": "failed",
                    "summary": "The Agent run failed before producing an answer.",
                }
                yield f"event: failed\ndata: {json.dumps(payload)}\n\n"

        return StreamingResponse(generate_live(), media_type="text/event-stream")

    @app.post("/api/research/stream")
    def stream_transient_research(request: MessageRequest) -> StreamingResponse:
        """Run research without writing conversation or run history."""
        stream_run_id = f"run_{uuid4().hex}"
        try:
            agent_stream = active.stream_agent(request.question, request.paper_ids)
        except ServiceConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        def generate_transient() -> Iterator[str]:
            started_at = datetime.now(timezone.utc)
            try:
                for item in agent_stream:
                    if isinstance(item, TraceEvent):
                        payload = {
                            "run_id": stream_run_id,
                            **item.model_dump(mode="json"),
                        }
                        yield (f"event: {item.event}\ndata: {json.dumps(payload)}\n\n")
                    elif isinstance(item, AgentRun):
                        completed_at = datetime.now(timezone.utc)
                        transient = StoredRun(
                            id=stream_run_id,
                            conversation_id="",
                            status=item.answer.status.value,
                            created_at=started_at,
                            completed_at=completed_at,
                            result=item,
                        )
                        payload = transient.model_dump(mode="json")
                        yield f"event: result\ndata: {json.dumps(payload)}\n\n"
            except Exception:
                payload = {
                    "run_id": stream_run_id,
                    "event": "failed",
                    "summary": "The Agent run failed before producing an answer.",
                }
                yield f"event: failed\ndata: {json.dumps(payload)}\n\n"

        return StreamingResponse(generate_transient(), media_type="text/event-stream")

    @app.get("/api/runs/{run_id}", response_model=StoredRun)
    def get_run(run_id: str) -> StoredRun:
        run = active.history.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    @app.get("/api/runs/{run_id}/events")
    def stream_run_events(run_id: str) -> StreamingResponse:
        if active.history.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="Run not found")

        def generate() -> Iterator[str]:
            for event in active.history.list_events(run_id):
                payload = {"run_id": run_id, **event.model_dump()}
                yield f"event: {event.event}\ndata: {json.dumps(payload)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    return app


app = create_app()
