"""Integration tests for local paper, conversation, and trace APIs."""

from collections.abc import Iterator
from pathlib import Path

from fastapi.testclient import TestClient

from hardsec_scholar.agent import AgentRun, QuestionPlan, TraceEvent
from hardsec_scholar.api.app import create_app
from hardsec_scholar.api.history import ConversationRepository
from hardsec_scholar.config import AppSettings
from hardsec_scholar.domain import PaperStatus, QuestionType
from hardsec_scholar.generation import AnswerStatus, GroundedAnswer
from hardsec_scholar.ingestion import PaperIngestionService
from hardsec_scholar.storage import PaperRepository


class _FakeServices:
    def __init__(self, root: Path) -> None:
        self.settings = AppSettings.model_validate(
            {
                "paths": {
                    "papers": root / "papers",
                    "chroma": root / "chroma",
                    "evaluations": root / "evaluations",
                    "database": root / "hardsec.db",
                },
                "chunking": {"chunk_size_tokens": 100, "overlap_tokens": 20},
            }
        )
        self.repository = PaperRepository(self.settings.paths.database)
        self.history = ConversationRepository(self.settings.paths.database)
        self.repository.initialize()
        self.history.initialize()
        self.ingestion = PaperIngestionService(self.settings, self.repository)

    def index_paper(self, paper_id: str) -> int:
        chunks = self.repository.get_chunks(paper_id)
        self.repository.update_status(paper_id, PaperStatus.INDEXED)
        return len(chunks)

    def delete_paper(self, paper_id: str) -> bool:
        paper = self.repository.get_paper(paper_id)
        deleted = self.repository.delete_paper(paper_id)
        if deleted and paper is not None and paper.file_path.is_file():
            paper.file_path.unlink()
        return deleted

    def run_agent(self, question: str, paper_ids: list[str]) -> AgentRun:
        return AgentRun(
            answer=GroundedAnswer(
                status=AnswerStatus.ANSWERED,
                answer="Coverage feedback guides mutation selection. [E1]",
                searched_paper_ids=paper_ids,
            ),
            question_type=QuestionType.MECHANISM,
            plan=QuestionPlan(
                question_type=QuestionType.MECHANISM,
                preferred_sections=["Methodology", "Evaluation"],
            ),
            search_queries=[question],
            rewrite_reasons=[],
            retrieval_retries=0,
            answer_repairs=0,
            trace_events=[
                TraceEvent(
                    sequence=1,
                    event="completed",
                    node="complete",
                    summary="Completed with one verified citation.",
                )
            ],
        )

    def stream_agent(
        self, question: str, paper_ids: list[str]
    ) -> Iterator[TraceEvent | AgentRun]:
        run = self.run_agent(question, paper_ids)
        yield from run.trace_events
        yield run


def test_paper_upload_update_evidence_and_delete(
    tmp_path: Path, sample_paper_pdf: Path
) -> None:
    services = _FakeServices(tmp_path / "api")
    client = TestClient(create_app(services))

    with sample_paper_pdf.open("rb") as stream:
        response = client.post(
            "/api/papers",
            files={"file": ("paper.pdf", stream, "application/pdf")},
        )

    assert response.status_code == 201
    uploaded = response.json()
    paper_id = uploaded["paper"]["id"]
    assert uploaded["indexed"] is True
    assert uploaded["paper"]["status"] == "indexed"
    assert uploaded["chunk_count"] > 0

    listing = client.get("/api/papers")
    assert listing.status_code == 200
    assert listing.json()[0]["paper"]["id"] == paper_id

    patched = client.patch(
        f"/api/papers/{paper_id}",
        json={"title": "Corrected Hardware Fuzzing", "doi": None},
    )
    assert patched.status_code == 200
    assert patched.json()["paper"]["title"] == "Corrected Hardware Fuzzing"
    assert patched.json()["paper"]["doi"] is None

    chunk = services.repository.get_chunks(paper_id)[0]
    evidence = client.get(f"/api/evidence/evidence_{chunk.id}")
    assert evidence.status_code == 200
    assert evidence.json()["paper_title"] == "Corrected Hardware Fuzzing"
    assert evidence.json()["page_start"] >= 1

    deleted = client.delete(f"/api/papers/{paper_id}")
    assert deleted.status_code == 204
    assert client.get(f"/api/papers/{paper_id}").status_code == 404


def test_conversation_run_persistence_and_sse_trace(tmp_path: Path) -> None:
    services = _FakeServices(tmp_path / "api")
    client = TestClient(create_app(services))

    created = client.post("/api/conversations")
    assert created.status_code == 201
    conversation_id = created.json()["id"]

    response = client.post(
        f"/api/conversations/{conversation_id}/messages",
        json={"question": "How does coverage guide mutations?", "paper_ids": []},
    )

    assert response.status_code == 200
    run = response.json()["run"]
    assert run["result"]["answer"]["status"] == "answered"
    run_id = run["id"]

    conversation = client.get(f"/api/conversations/{conversation_id}").json()
    assert [message["role"] for message in conversation["messages"]] == [
        "user",
        "assistant",
    ]

    events = client.get(f"/api/runs/{run_id}/events")
    assert events.status_code == 200
    assert events.headers["content-type"].startswith("text/event-stream")
    assert "event: completed" in events.text
    assert '"run_id":' in events.text


def test_message_stream_emits_trace_and_persists_result(tmp_path: Path) -> None:
    services = _FakeServices(tmp_path / "api")
    client = TestClient(create_app(services))
    conversation_id = client.post("/api/conversations").json()["id"]

    response = client.post(
        f"/api/conversations/{conversation_id}/messages/stream",
        json={"question": "How does coverage guide mutations?"},
    )

    assert response.status_code == 200
    assert "event: completed" in response.text
    assert "event: result" in response.text
    conversation = client.get(f"/api/conversations/{conversation_id}").json()
    assert len(conversation["messages"]) == 2


def test_conversation_history_can_be_listed_viewed_and_deleted(tmp_path: Path) -> None:
    services = _FakeServices(tmp_path / "api")
    client = TestClient(create_app(services))
    conversation_id = client.post("/api/conversations").json()["id"]
    response = client.post(
        f"/api/conversations/{conversation_id}/messages",
        json={"question": "How does coverage guide mutations?"},
    )
    run_id = response.json()["run"]["id"]

    listing = client.get("/api/conversations")

    assert listing.status_code == 200
    assert listing.json() == [
        {
            "id": conversation_id,
            "title": "How does coverage guide mutations?",
            "created_at": listing.json()[0]["created_at"],
            "updated_at": listing.json()[0]["updated_at"],
            "message_count": 2,
        }
    ]
    assert client.get(f"/api/conversations/{conversation_id}").status_code == 200

    deleted = client.delete(f"/api/conversations/{conversation_id}")

    assert deleted.status_code == 204
    assert client.get(f"/api/conversations/{conversation_id}").status_code == 404
    assert client.get(f"/api/runs/{run_id}").status_code == 404
    assert client.get(f"/api/runs/{run_id}/events").status_code == 404
    assert client.delete(f"/api/conversations/{conversation_id}").status_code == 404


def test_transient_research_does_not_persist_history(tmp_path: Path) -> None:
    services = _FakeServices(tmp_path / "api")
    client = TestClient(create_app(services))

    response = client.post(
        "/api/research/stream",
        json={"question": "How does coverage guide mutations?"},
    )

    assert response.status_code == 200
    assert "event: completed" in response.text
    assert "event: result" in response.text
    assert '"conversation_id": ""' in response.text
    assert client.get("/api/conversations").json() == []


def test_api_rejects_web_search_and_non_pdf_upload(tmp_path: Path) -> None:
    services = _FakeServices(tmp_path / "api")
    services.settings.ingestion.max_file_size_mb = 1
    client = TestClient(create_app(services))
    conversation_id = client.post("/api/conversations").json()["id"]

    message = client.post(
        f"/api/conversations/{conversation_id}/messages",
        json={"question": "Search the web", "allow_web_search": True},
    )
    upload = client.post(
        "/api/papers",
        files={"file": ("notes.txt", b"not a pdf", "text/plain")},
    )
    oversized = client.post(
        "/api/papers",
        files={
            "file": ("large.pdf", b"%PDF" + b"0" * (1024 * 1024), "application/pdf")
        },
    )

    assert message.status_code == 422
    assert upload.status_code == 415
    assert oversized.status_code == 413
