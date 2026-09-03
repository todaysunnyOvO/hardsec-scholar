"""SQLite repository for paper metadata and parsed chunks."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from hardsec_scholar.domain import PaperChunk, PaperMetadata, PaperStatus, ResearchArea


class PaperRepository:
    """Persist papers and chunks transactionally in a local SQLite database."""

    def __init__(self, database_path: Path | str) -> None:
        """Store the database location without creating files eagerly."""
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        """Create the database schema when it does not exist."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    authors_json TEXT NOT NULL,
                    year INTEGER,
                    doi TEXT,
                    research_area TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS paper_chunks (
                    id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    section TEXT,
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    source_type TEXT NOT NULL,
                    text TEXT NOT NULL,
                    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
                    UNIQUE (paper_id, chunk_index)
                );

                CREATE INDEX IF NOT EXISTS idx_papers_content_hash
                    ON papers(content_hash);
                CREATE INDEX IF NOT EXISTS idx_chunks_paper_id
                    ON paper_chunks(paper_id);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def save_paper(self, paper: PaperMetadata, chunks: list[PaperChunk]) -> None:
        """Replace one paper and all of its chunks in a single transaction."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO papers (
                    id, content_hash, title, authors_json, year, doi,
                    research_area, file_path, page_count, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content_hash=excluded.content_hash,
                    title=excluded.title,
                    authors_json=excluded.authors_json,
                    year=excluded.year,
                    doi=excluded.doi,
                    research_area=excluded.research_area,
                    file_path=excluded.file_path,
                    page_count=excluded.page_count,
                    status=excluded.status,
                    created_at=excluded.created_at
                """,
                (
                    paper.id,
                    paper.content_hash,
                    paper.title,
                    json.dumps(paper.authors),
                    paper.year,
                    paper.doi,
                    paper.research_area.value,
                    str(paper.file_path),
                    paper.page_count,
                    paper.status.value,
                    paper.created_at.isoformat(),
                ),
            )
            connection.execute(
                "DELETE FROM paper_chunks WHERE paper_id = ?", (paper.id,)
            )
            connection.executemany(
                """
                INSERT INTO paper_chunks (
                    id, paper_id, title, section, page_start, page_end,
                    chunk_index, source_type, text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.id,
                        chunk.paper_id,
                        chunk.title,
                        chunk.section,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.chunk_index,
                        chunk.source_type,
                        chunk.text,
                    )
                    for chunk in chunks
                ],
            )

    def find_by_hash(self, content_hash: str) -> PaperMetadata | None:
        """Return an existing paper with the given content hash."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM papers WHERE content_hash = ?", (content_hash,)
            ).fetchone()
        return self._paper_from_row(row) if row else None

    def get_paper(self, paper_id: str) -> PaperMetadata | None:
        """Return one paper by ID."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM papers WHERE id = ?", (paper_id,)
            ).fetchone()
        return self._paper_from_row(row) if row else None

    def list_papers(self) -> list[PaperMetadata]:
        """List papers in deterministic creation order."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM papers ORDER BY created_at, id"
            ).fetchall()
        return [self._paper_from_row(row) for row in rows]

    def get_chunks(self, paper_id: str) -> list[PaperChunk]:
        """Return all chunks belonging to a paper in source order."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM paper_chunks
                WHERE paper_id = ?
                ORDER BY chunk_index
                """,
                (paper_id,),
            ).fetchall()
        return [self._chunk_from_row(row) for row in rows]

    def get_chunk(self, chunk_id: str) -> PaperChunk | None:
        """Return one chunk by its stable identifier."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM paper_chunks WHERE id = ?", (chunk_id,)
            ).fetchone()
        return self._chunk_from_row(row) if row else None

    def list_chunks(self, paper_ids: list[str] | None = None) -> list[PaperChunk]:
        """List chunks across the corpus, optionally restricted to selected papers."""
        with self._connect() as connection:
            if paper_ids:
                placeholders = ",".join("?" for _ in paper_ids)
                rows = connection.execute(
                    f"""
                    SELECT * FROM paper_chunks
                    WHERE paper_id IN ({placeholders})
                    ORDER BY paper_id, chunk_index
                    """,  # noqa: S608 - placeholders are generated, not user controlled.
                    tuple(paper_ids),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM paper_chunks ORDER BY paper_id, chunk_index"
                ).fetchall()
        return [self._chunk_from_row(row) for row in rows]

    def update_status(self, paper_id: str, status: PaperStatus) -> bool:
        """Update one paper's ingestion/indexing lifecycle status."""
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE papers SET status = ? WHERE id = ?", (status.value, paper_id)
            )
        return cursor.rowcount > 0

    def update_metadata(
        self,
        paper_id: str,
        changes: dict[str, Any],
    ) -> PaperMetadata | None:
        """Update editable paper metadata without changing source identity."""
        paper = self.get_paper(paper_id)
        if paper is None:
            return None
        allowed = {"title", "authors", "year", "doi", "research_area"}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Unsupported metadata fields: {sorted(unknown)}")
        updated = PaperMetadata.model_validate({**paper.model_dump(), **changes})
        chunks = [
            chunk.model_copy(update={"title": updated.title})
            for chunk in self.get_chunks(paper_id)
        ]
        self.save_paper(updated, chunks)
        return updated

    def delete_paper(self, paper_id: str) -> bool:
        """Delete a paper and cascade deletion to its chunks."""
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
        return cursor.rowcount > 0

    @staticmethod
    def _paper_from_row(row: sqlite3.Row) -> PaperMetadata:
        return PaperMetadata(
            id=row["id"],
            content_hash=row["content_hash"],
            title=row["title"],
            authors=json.loads(row["authors_json"]),
            year=row["year"],
            doi=row["doi"],
            research_area=ResearchArea(row["research_area"]),
            file_path=Path(row["file_path"]),
            page_count=row["page_count"],
            status=PaperStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _chunk_from_row(row: sqlite3.Row) -> PaperChunk:
        return PaperChunk(
            id=row["id"],
            paper_id=row["paper_id"],
            title=row["title"],
            section=row["section"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            chunk_index=row["chunk_index"],
            source_type=row["source_type"],
            text=row["text"],
        )
