"""SQLite persistence for conversations, completed runs, and safe trace events."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from hardsec_scholar.agent import AgentRun, TraceEvent


class MessageRecord(BaseModel):
    """Represent one persisted user or assistant conversation message."""

    model_config = ConfigDict(extra="forbid")

    id: str
    role: str
    content: str
    run_id: str | None = None
    created_at: datetime


class ConversationRecord(BaseModel):
    """Represent a conversation and its ordered messages."""

    model_config = ConfigDict(extra="forbid")

    id: str
    created_at: datetime
    messages: list[MessageRecord] = Field(default_factory=list)


class ConversationSummary(BaseModel):
    """Summarize one persisted conversation for history navigation."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = Field(ge=0)


class StoredRun(BaseModel):
    """Represent a completed Agent run and its safe public result."""

    model_config = ConfigDict(extra="forbid")

    id: str
    conversation_id: str
    status: str
    created_at: datetime
    completed_at: datetime
    result: AgentRun


class ConversationRepository:
    """Persist UI history in the same local SQLite database as paper metadata."""

    def __init__(self, database_path: Path | str) -> None:
        """Store the shared local database path."""
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        """Create conversation and trace tables plus query-driven indexes."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    run_id TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS agent_runs (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS run_events (
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event TEXT NOT NULL,
                    node TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    PRIMARY KEY (run_id, sequence),
                    FOREIGN KEY (run_id) REFERENCES agent_runs(id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
                    ON conversation_messages(conversation_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_runs_conversation_created
                    ON agent_runs(conversation_id, created_at);
                """
            )
            connection.execute("PRAGMA optimize")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def create_conversation(self) -> ConversationRecord:
        """Create and return an empty conversation."""
        conversation_id = f"conv_{uuid4().hex}"
        created_at = datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations (id, created_at) VALUES (?, ?)",
                (conversation_id, created_at.isoformat()),
            )
        return ConversationRecord(id=conversation_id, created_at=created_at)

    def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        """Return a conversation with ordered messages."""
        with self._connect() as connection:
            conversation = connection.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if conversation is None:
                return None
            rows = connection.execute(
                """
                SELECT * FROM conversation_messages
                WHERE conversation_id = ?
                ORDER BY created_at, rowid
                """,
                (conversation_id,),
            ).fetchall()
        return ConversationRecord(
            id=conversation["id"],
            created_at=datetime.fromisoformat(conversation["created_at"]),
            messages=[
                MessageRecord(
                    id=row["id"],
                    role=row["role"],
                    content=row["content"],
                    run_id=row["run_id"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in rows
            ],
        )

    def list_conversations(self, *, limit: int = 100) -> list[ConversationSummary]:
        """Return recent conversations with a first-question preview."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    conversations.id,
                    conversations.created_at,
                    COALESCE(MAX(conversation_messages.created_at), conversations.created_at)
                        AS updated_at,
                    COUNT(conversation_messages.id) AS message_count,
                    COALESCE(
                        (
                            SELECT content
                            FROM conversation_messages AS first_message
                            WHERE first_message.conversation_id = conversations.id
                                AND first_message.role = 'user'
                            ORDER BY first_message.created_at, first_message.rowid
                            LIMIT 1
                        ),
                        'New conversation'
                    ) AS title
                FROM conversations
                LEFT JOIN conversation_messages
                    ON conversation_messages.conversation_id = conversations.id
                GROUP BY conversations.id
                ORDER BY updated_at DESC, conversations.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            ConversationSummary(
                id=row["id"],
                title=row["title"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                message_count=row["message_count"],
            )
            for row in rows
        ]

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its cascaded messages, runs, and events."""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            )
        return cursor.rowcount > 0

    def save_run(
        self,
        conversation_id: str,
        question: str,
        run: AgentRun,
        *,
        run_id: str | None = None,
    ) -> StoredRun:
        """Atomically save both messages, final run data, and trace events."""
        if self.get_conversation(conversation_id) is None:
            raise KeyError(f"Unknown conversation: {conversation_id}")
        resolved_run_id = run_id or f"run_{uuid4().hex}"
        created_at = datetime.now(timezone.utc)
        completed_at = datetime.now(timezone.utc)
        status = run.answer.status.value
        user_message_id = f"msg_{uuid4().hex}"
        assistant_message_id = f"msg_{uuid4().hex}"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_runs (
                    id, conversation_id, status, result_json,
                    created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_run_id,
                    conversation_id,
                    status,
                    run.model_dump_json(),
                    created_at.isoformat(),
                    completed_at.isoformat(),
                ),
            )
            connection.executemany(
                """
                INSERT INTO conversation_messages (
                    id, conversation_id, role, content, run_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        user_message_id,
                        conversation_id,
                        "user",
                        question,
                        resolved_run_id,
                        created_at.isoformat(),
                    ),
                    (
                        assistant_message_id,
                        conversation_id,
                        "assistant",
                        run.answer.answer,
                        resolved_run_id,
                        completed_at.isoformat(),
                    ),
                ],
            )
            connection.executemany(
                """
                INSERT INTO run_events (
                    run_id, sequence, event, node, summary
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        resolved_run_id,
                        event.sequence,
                        event.event,
                        event.node,
                        event.summary,
                    )
                    for event in run.trace_events
                ],
            )
        return StoredRun(
            id=resolved_run_id,
            conversation_id=conversation_id,
            status=status,
            created_at=created_at,
            completed_at=completed_at,
            result=run,
        )

    def get_run(self, run_id: str) -> StoredRun | None:
        """Return one completed run by ID."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_runs WHERE id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return StoredRun(
            id=row["id"],
            conversation_id=row["conversation_id"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            completed_at=datetime.fromisoformat(row["completed_at"]),
            result=AgentRun.model_validate_json(row["result_json"]),
        )

    def list_events(self, run_id: str) -> list[TraceEvent]:
        """Return safe trace events in execution order."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM run_events WHERE run_id = ? ORDER BY sequence",
                (run_id,),
            ).fetchall()
        return [
            TraceEvent(
                sequence=row["sequence"],
                event=row["event"],
                node=row["node"],
                summary=row["summary"],
            )
            for row in rows
        ]
