"use client";

import { useEffect, useRef, useState } from "react";
import type { FormEvent, ReactNode } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Paper = {
  id: string;
  title: string;
  authors: string[];
  year: number | null;
  doi: string | null;
  research_area: string;
  page_count: number;
  status: "pending" | "parsed" | "indexed" | "failed";
};

type PaperView = { paper: Paper; chunk_count: number };
type TraceEvent = { sequence: number; event: string; node: string; summary: string };
type Evidence = {
  id: string;
  paper_id: string;
  paper_title: string;
  section: string | null;
  page_start: number;
  page_end: number;
  text: string;
};
type Citation = Evidence & { evidence_id: string; claim: string };
type GroundedAnswer = {
  status: "answered" | "abstained";
  answer: string;
  citations: Citation[];
  evidence: Evidence[];
  missing_evidence: string[];
  verification_errors: string[];
};
type AgentRun = {
  answer: GroundedAnswer;
  plan: {
    question_type: string;
    sub_questions: string[];
    preferred_sections: string[];
    requires_comparison: boolean;
  };
  search_queries: string[];
  rewrite_reasons: string[];
  retrieval_retries: number;
  answer_repairs: number;
  trace_events: TraceEvent[];
};
type StoredRun = { id: string; status: string; result: AgentRun };
type MessageRecord = {
  id: string;
  role: "user" | "assistant";
  content: string;
  run_id: string | null;
  created_at: string;
};
type ConversationRecord = {
  id: string;
  created_at: string;
  messages: MessageRecord[];
};
type ConversationSummary = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
};

const suggestions = [
  "Compare the feedback mechanisms used by the selected fuzzing papers.",
  "What attacker capabilities are assumed in the side-channel threat model?",
  "Which evaluation metrics expose the main architectural security trade-off?",
];

const areaLabels: Record<string, string> = {
  side_channel_attack: "Side-channel",
  architectural_security: "Architecture",
  hardware_fuzzing: "Fuzzing",
  other: "Other",
};

async function readApiError(response: Response): Promise<string> {
  try {
    const payload = await response.json();
    if (typeof payload.detail === "string") return payload.detail;
    if (Array.isArray(payload.detail)) return payload.detail[0]?.msg ?? "Request failed";
  } catch {
    // Fall through to the status message.
  }
  return `${response.status} ${response.statusText}`;
}

function renderAnswer(answer: string, citations: Citation[], onOpen: (id: string) => void) {
  const citationById = new Map(citations.map((item) => [item.evidence_id, item]));
  const parts = answer.split(/\[([^\]]+)\]/g);
  return parts.map((part, index) => {
    if (index % 2 === 0) return <span key={`${index}-${part}`}>{part}</span>;
    const ids = part.split(",").map((value) => value.trim());
    return (
      <span className="citation-group" key={`${index}-${part}`}>
        {ids.map((id) => {
          const citation = citationById.get(id);
          return (
            <button className="citation" key={id} onClick={() => onOpen(id)} type="button">
              {id} · p. {citation?.page_start ?? "?"}
            </button>
          );
        })}
      </span>
    );
  });
}

function SectionTitle({ eyebrow, children }: { eyebrow: string; children: ReactNode }) {
  return (
    <div className="section-title">
      <p>{eyebrow}</p>
      <h2>{children}</h2>
    </div>
  );
}

export default function Home() {
  const [view, setView] = useState<"research" | "library" | "history">("research");
  const [papers, setPapers] = useState<PaperView[]>([]);
  const [selectedPapers, setSelectedPapers] = useState<string[]>([]);
  const [paperMenuOpen, setPaperMenuOpen] = useState(false);
  const [question, setQuestion] = useState(suggestions[0]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [saveHistory, setSaveHistory] = useState(true);
  const [history, setHistory] = useState<ConversationSummary[]>([]);
  const [selectedHistory, setSelectedHistory] = useState<ConversationRecord | null>(null);
  const [historyBusy, setHistoryBusy] = useState(false);
  const [run, setRun] = useState<StoredRun | null>(null);
  const [liveTrace, setLiveTrace] = useState<TraceEvent[]>([]);
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeEvidence, setActiveEvidence] = useState<Evidence | null>(null);
  const [editingPaper, setEditingPaper] = useState<PaperView | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const loadPapers = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/papers`);
      if (!response.ok) throw new Error(await readApiError(response));
      const payload: PaperView[] = await response.json();
      setPapers(payload);
      setSelectedPapers((current) => {
        const available = new Set(payload.filter((item) => item.paper.status === "indexed").map((item) => item.paper.id));
        const retained = current.filter((id) => available.has(id));
        return retained.length ? retained : [...available];
      });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not reach the local API.");
    }
  };

  const loadHistory = async () => {
    setHistoryBusy(true);
    try {
      const response = await fetch(`${API_BASE}/api/conversations`);
      if (!response.ok) throw new Error(await readApiError(response));
      setHistory(await response.json() as ConversationSummary[]);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load conversation history.");
    } finally {
      setHistoryBusy(false);
    }
  };

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${API_BASE}/api/papers`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(await readApiError(response));
        return response.json() as Promise<PaperView[]>;
      })
      .then((payload) => {
        setPapers(payload);
        setSelectedPapers(
          payload
            .filter((item) => item.paper.status === "indexed")
            .map((item) => item.paper.id),
        );
      })
      .catch((cause) => {
        if (cause instanceof DOMException && cause.name === "AbortError") return;
        setError(cause instanceof Error ? cause.message : "Could not reach the local API.");
      });
    return () => controller.abort();
  }, []);

  const updateSaveHistory = (enabled: boolean) => {
    setSaveHistory(enabled);
    setConversationId(null);
  };

  const ensureConversation = async () => {
    if (conversationId) return conversationId;
    const response = await fetch(`${API_BASE}/api/conversations`, { method: "POST" });
    if (!response.ok) throw new Error(await readApiError(response));
    const payload: { id: string } = await response.json();
    setConversationId(payload.id);
    return payload.id;
  };

  const askQuestion = async (event?: FormEvent) => {
    event?.preventDefault();
    if (!question.trim() || busy) return;
    if (!selectedPapers.length) {
      setError("Select at least one indexed paper before running research.");
      return;
    }
    setBusy(true);
    setRun(null);
    setLiveTrace([]);
    setActiveEvidence(null);
    setError(null);
    try {
      const conversation = saveHistory ? await ensureConversation() : null;
      const endpoint = conversation
        ? `${API_BASE}/api/conversations/${conversation}/messages/stream`
        : `${API_BASE}/api/research/stream`;
      const response = await fetch(
        endpoint,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: question.trim(), paper_ids: selectedPapers }),
        },
      );
      if (!response.ok) throw new Error(await readApiError(response));
      if (!response.body) throw new Error("The local API returned no event stream.");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";
        for (const frame of frames) {
          const eventName = frame.match(/^event: (.+)$/m)?.[1];
          const data = frame.match(/^data: (.+)$/m)?.[1];
          if (!eventName || !data) continue;
          const payload = JSON.parse(data);
          if (eventName === "result") {
            setRun(payload as StoredRun);
          } else if (eventName === "failed") {
            throw new Error(payload.summary ?? "The Agent run failed.");
          } else {
            setLiveTrace((current) => [...current, payload as TraceEvent]);
          }
        }
        if (done) break;
      }
      if (saveHistory) await loadHistory();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The research run failed.");
    } finally {
      setBusy(false);
    }
  };

  const viewHistory = async (conversationIdToView: string) => {
    setHistoryBusy(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/api/conversations/${conversationIdToView}`);
      if (!response.ok) throw new Error(await readApiError(response));
      setSelectedHistory(await response.json() as ConversationRecord);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not open conversation history.");
    } finally {
      setHistoryBusy(false);
    }
  };

  const continueHistory = async () => {
    if (!selectedHistory) return;
    const latestUser = [...selectedHistory.messages].reverse().find((message) => message.role === "user");
    const latestRunId = [...selectedHistory.messages].reverse().find((message) => message.role === "assistant" && message.run_id)?.run_id;
    try {
      if (latestRunId) {
        const response = await fetch(`${API_BASE}/api/runs/${latestRunId}`);
        if (!response.ok) throw new Error(await readApiError(response));
        setRun(await response.json() as StoredRun);
      } else {
        setRun(null);
      }
      if (latestUser) setQuestion(latestUser.content);
      setSaveHistory(true);
      setConversationId(selectedHistory.id);
      setLiveTrace([]);
      setView("research");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not restore the latest run.");
    }
  };

  const deleteHistory = async (conversation: ConversationSummary | ConversationRecord) => {
    if (!window.confirm("Delete this conversation and all of its saved runs?")) return;
    const response = await fetch(`${API_BASE}/api/conversations/${conversation.id}`, { method: "DELETE" });
    if (!response.ok) {
      setError(await readApiError(response));
      return;
    }
    if (conversationId === conversation.id) {
      setConversationId(null);
      setRun(null);
      setLiveTrace([]);
    }
    if (selectedHistory?.id === conversation.id) setSelectedHistory(null);
    await loadHistory();
  };

  const uploadPaper = async (file: File) => {
    setUploading(true);
    setError(null);
    try {
      const body = new FormData();
      body.append("file", file);
      const response = await fetch(`${API_BASE}/api/papers`, { method: "POST", body });
      if (!response.ok) throw new Error(await readApiError(response));
      const uploaded = await response.json();
      if (uploaded.indexing_warning) setError(uploaded.indexing_warning);
      await loadPapers();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Paper upload failed.");
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  };

  const reindexPaper = async (paperId: string) => {
    setError(null);
    const response = await fetch(`${API_BASE}/api/papers/${paperId}/reindex`, { method: "POST" });
    if (!response.ok) {
      setError(await readApiError(response));
      return;
    }
    await loadPapers();
  };

  const deletePaper = async (paper: Paper) => {
    if (!window.confirm(`Delete “${paper.title}” and its local index?`)) return;
    const response = await fetch(`${API_BASE}/api/papers/${paper.id}`, { method: "DELETE" });
    if (!response.ok) {
      setError(await readApiError(response));
      return;
    }
    await loadPapers();
  };

  const savePaper = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!editingPaper) return;
    const form = new FormData(event.currentTarget);
    const response = await fetch(`${API_BASE}/api/papers/${editingPaper.paper.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: String(form.get("title") ?? ""),
        year: form.get("year") ? Number(form.get("year")) : null,
        research_area: form.get("research_area"),
      }),
    });
    if (!response.ok) {
      setError(await readApiError(response));
      return;
    }
    setEditingPaper(null);
    await loadPapers();
  };

  const openEvidence = (evidenceId: string) => {
    const evidence = run?.result.answer.evidence.find((item) => item.id === evidenceId);
    setActiveEvidence(evidence ?? null);
  };

  const indexedPapers = papers.filter((item) => item.paper.status === "indexed");
  const traces = run?.result.trace_events ?? liveTrace;
  const answer = run?.result.answer;

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <button className="brand-mark" onClick={() => setView("research")} type="button" aria-label="HardSec Scholar home">HS</button>
        <nav aria-label="Primary navigation">
          <button className={`nav-item ${view === "research" ? "active" : ""}`} onClick={() => setView("research")} type="button"><span>⌁</span>Research</button>
          <button className={`nav-item ${view === "library" ? "active" : ""}`} onClick={() => setView("library")} type="button"><span>▤</span>Paper library</button>
          <button className={`nav-item ${view === "history" ? "active" : ""}`} onClick={() => { setView("history"); void loadHistory(); }} type="button"><span>◷</span>History</button>
        </nav>
        <div className="sidebar-note"><span className="status-dot" />Local corpus only</div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">HARDWARE SECURITY RESEARCH</p>
            <h1>{view === "research" ? "Evidence, not intuition." : view === "library" ? "Your research corpus." : "Saved conversations."}</h1>
          </div>
          {view === "research" ? (
            <div className="paper-filter-wrap">
              <button className="paper-filter" onClick={() => setPaperMenuOpen((value) => !value)} type="button" aria-expanded={paperMenuOpen}>
                <span>Selected papers</span><strong>{selectedPapers.length}</strong>
              </button>
              {paperMenuOpen && (
                <div className="paper-menu">
                  <button type="button" onClick={() => setSelectedPapers(selectedPapers.length === indexedPapers.length ? [] : indexedPapers.map((item) => item.paper.id))}>
                    {selectedPapers.length === indexedPapers.length ? "Clear all" : "Select all indexed"}
                  </button>
                  {indexedPapers.map(({ paper }) => (
                    <label key={paper.id}>
                      <input type="checkbox" checked={selectedPapers.includes(paper.id)} onChange={() => setSelectedPapers((current) => current.includes(paper.id) ? current.filter((id) => id !== paper.id) : [...current, paper.id])} />
                      <span>{paper.title}</span>
                    </label>
                  ))}
                  {!indexedPapers.length && <p>No indexed papers yet.</p>}
                </div>
              )}
            </div>
          ) : view === "library" ? (
            <button className="primary-action" onClick={() => fileInput.current?.click()} disabled={uploading} type="button">{uploading ? "Processing…" : "+ Add paper"}</button>
          ) : <button className="primary-action" onClick={() => void loadHistory()} disabled={historyBusy} type="button">{historyBusy ? "Refreshing…" : "Refresh"}</button>}
        </header>

        {error && <div className="error-banner" role="alert"><span>!</span><p>{error}</p><button onClick={() => setError(null)} type="button">Dismiss</button></div>}

        {view === "research" ? (
          <div className="research-grid">
            <section className="conversation-panel">
              <form className="prompt-card" onSubmit={askQuestion}>
                <label htmlFor="question">Ask across your paper collection</label>
                <textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} rows={3} placeholder="Ask about a mechanism, threat model, experiment, metric, or limitation…" />
                <div className="prompt-actions">
                  <label className="history-toggle">
                    <input type="checkbox" checked={saveHistory} disabled={busy} onChange={(event) => updateSaveHistory(event.target.checked)} />
                    <span>Save conversation history</span>
                  </label>
                  <button disabled={busy || !question.trim()} type="submit">{busy ? "Researching…" : "Run research"}</button>
                </div>
                <p className="storage-note">Agentic retrieval · citations required · web disabled. {saveHistory ? "Questions, answers, runs, and safe traces will be stored in local SQLite." : "This run will stay on this page only and will not be written to conversation history."}</p>
              </form>

              {!answer && !busy && (
                <section className="suggestions">
                  <SectionTitle eyebrow="START WITH A RESEARCH QUESTION">Built for paper-grounded analysis</SectionTitle>
                  <div className="suggestion-list">
                    {suggestions.map((suggestion, index) => <button key={suggestion} onClick={() => setQuestion(suggestion)} type="button"><span>0{index + 1}</span>{suggestion}</button>)}
                  </div>
                </section>
              )}

              {busy && !answer && (
                <article className="answer-card loading-card"><span className="pulse-block" /><div><p>AGENT WORKING</p><h2>{traces.at(-1)?.summary ?? "Preparing the research plan…"}</h2></div></article>
              )}

              {answer && (
                <article className={`answer-card ${answer.status === "abstained" ? "abstained" : ""}`}>
                  <div className="answer-heading"><span className="answer-index">01</span><div><p>{answer.status === "answered" ? "GROUNDED ANSWER" : "EVIDENCE INSUFFICIENT"}</p><h2>{answer.status === "answered" ? "Answer verified against the selected papers." : "The corpus cannot support a reliable answer yet."}</h2></div></div>
                  <div className="answer-copy">{renderAnswer(answer.answer, answer.citations, openEvidence)}</div>
                  {!!answer.missing_evidence.length && <div className="evidence-gap"><strong>Missing evidence</strong>{answer.missing_evidence.map((gap) => <p key={gap}>{gap}</p>)}</div>}
                  <div className="source-strip"><span>{answer.evidence.length} evidence excerpts</span><span>{answer.citations.length} verified citations</span><span>{run?.result.retrieval_retries ?? 0} retrieval retries</span></div>
                </article>
              )}

              {run && (
                <details className="run-details">
                  <summary>Inspect retrieval plan and queries</summary>
                  <div className="detail-grid">
                    <div><p>QUESTION TYPE</p><strong>{run.result.plan.question_type.replaceAll("_", " ")}</strong></div>
                    <div><p>PREFERRED SECTIONS</p><strong>{run.result.plan.preferred_sections.join(" · ") || "All sections"}</strong></div>
                  </div>
                  <h3>Queries</h3>
                  <ol>{run.result.search_queries.map((query) => <li key={query}>{query}</li>)}</ol>
                  {!!run.result.rewrite_reasons.length && <><h3>Why the Agent retried</h3><ul>{run.result.rewrite_reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul></>}
                </details>
              )}
            </section>

            <aside className="trace-panel">
              <div className="trace-title"><div><p>AGENT TRACE</p><h2>{traces.length} steps</h2></div><span className={`run-status ${busy ? "running" : ""}`}>{busy ? "Running" : run ? "Complete" : "Ready"}</span></div>
              <ol className="trace-list">
                {traces.map((trace) => <li key={`${trace.sequence}-${trace.event}`}><span className="step-number">{String(trace.sequence).padStart(2, "0")}</span><div><strong>{trace.node.replaceAll("_", " ")}</strong><p>{trace.summary}</p></div></li>)}
              </ol>
              {!traces.length && <p className="trace-empty">The Agent’s safe execution summary will appear here. Hidden reasoning and full paper text are never logged.</p>}
            </aside>
          </div>
        ) : view === "library" ? (
          <section className="library-view">
            <div className="library-summary">
              <div><strong>{papers.length}</strong><span>Papers</span></div>
              <div><strong>{papers.reduce((total, item) => total + item.chunk_count, 0)}</strong><span>Evidence chunks</span></div>
              <div><strong>{indexedPapers.length}</strong><span>Ready to search</span></div>
            </div>
            <button className="upload-zone" onClick={() => fileInput.current?.click()} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); const file = event.dataTransfer.files[0]; if (file) void uploadPaper(file); }} type="button">
              <span>PDF</span><div><strong>Drop an English hardware-security paper here</strong><p>Text-based PDF · processed and stored on this device</p></div><b>Browse</b>
            </button>
            <div className="paper-table" role="table" aria-label="Paper library">
              <div className="paper-row paper-head" role="row"><span>Paper</span><span>Area</span><span>Pages</span><span>Status</span><span>Actions</span></div>
              {papers.map(({ paper, chunk_count }) => (
                <div className="paper-row" role="row" key={paper.id}>
                  <div className="paper-identity"><span>{areaLabels[paper.research_area]?.slice(0, 2).toUpperCase()}</span><div><strong>{paper.title}</strong><p>{paper.authors.join(", ") || "Authors not detected"} · {paper.year ?? "Year unknown"} · {chunk_count} chunks</p></div></div>
                  <span className="area-tag">{areaLabels[paper.research_area] ?? paper.research_area}</span>
                  <span>{paper.page_count}</span>
                  <span className={`paper-status ${paper.status}`}>{paper.status}</span>
                  <div className="row-actions"><button onClick={() => setEditingPaper({ paper, chunk_count })} type="button">Edit</button><button onClick={() => void reindexPaper(paper.id)} type="button">Reindex</button><button className="danger" onClick={() => void deletePaper(paper)} type="button">Delete</button></div>
                </div>
              ))}
              {!papers.length && <div className="library-empty"><strong>No papers indexed</strong><p>Add your first PDF to begin building the corpus.</p></div>}
            </div>
          </section>
        ) : (
          <section className="history-view">
            <div className="history-list" aria-label="Saved conversations">
              <div className="history-list-heading"><strong>{history.length} conversations</strong><span>Stored in local SQLite</span></div>
              {history.map((conversation) => (
                <article className={`history-item ${selectedHistory?.id === conversation.id ? "active" : ""}`} key={conversation.id}>
                  <button className="history-open" onClick={() => void viewHistory(conversation.id)} type="button">
                    <strong>{conversation.title}</strong>
                    <span>{new Date(conversation.updated_at).toLocaleString()} · {conversation.message_count} messages</span>
                  </button>
                  <button className="history-delete" onClick={() => void deleteHistory(conversation)} type="button" aria-label={`Delete ${conversation.title}`}>Delete</button>
                </article>
              ))}
              {!history.length && !historyBusy && <div className="history-empty"><strong>No saved conversations</strong><p>Enable “Save conversation history” before running research to keep a local copy.</p></div>}
            </div>
            <div className="history-detail">
              {selectedHistory ? (
                <>
                  <div className="history-detail-heading">
                    <div><p className="eyebrow">CONVERSATION</p><h2>{selectedHistory.messages.find((message) => message.role === "user")?.content ?? "Saved conversation"}</h2></div>
                    <div className="history-detail-actions"><button onClick={() => void continueHistory()} type="button">Continue</button><button className="danger" onClick={() => void deleteHistory(selectedHistory)} type="button">Delete</button></div>
                  </div>
                  <ol className="message-history">
                    {selectedHistory.messages.map((message) => <li className={message.role} key={message.id}><span>{message.role}</span><p>{message.content}</p><time>{new Date(message.created_at).toLocaleString()}</time></li>)}
                  </ol>
                </>
              ) : <div className="history-placeholder"><span>◷</span><strong>Select a conversation</strong><p>View its questions and answers, continue the thread, or delete it permanently.</p></div>}
            </div>
          </section>
        )}
      </section>

      <input ref={fileInput} type="file" accept="application/pdf,.pdf" hidden onChange={(event) => { const file = event.target.files?.[0]; if (file) void uploadPaper(file); }} />

      {activeEvidence && (
        <div className="drawer-backdrop">
          <button className="backdrop-dismiss" onClick={() => setActiveEvidence(null)} type="button" aria-label="Close evidence detail" />
          <aside className="evidence-drawer" aria-label="Evidence detail">
            <button className="close-button" onClick={() => setActiveEvidence(null)} type="button" aria-label="Close evidence">×</button>
            <p className="eyebrow">SOURCE EVIDENCE</p><h2>{activeEvidence.paper_title}</h2>
            <div className="evidence-meta"><span>{activeEvidence.section ?? "Unknown section"}</span><span>Pages {activeEvidence.page_start}{activeEvidence.page_end !== activeEvidence.page_start ? `–${activeEvidence.page_end}` : ""}</span><span>{activeEvidence.id}</span></div>
            <blockquote>{activeEvidence.text}</blockquote>
            <p className="privacy-note">This excerpt remains local. Only selected evidence is sent to the configured answer model.</p>
          </aside>
        </div>
      )}

      {editingPaper && (
        <div className="modal-backdrop" role="presentation">
          <form className="edit-modal" onSubmit={savePaper}>
            <button className="close-button" onClick={() => setEditingPaper(null)} type="button" aria-label="Close editor">×</button>
            <p className="eyebrow">CORRECT METADATA</p><h2>Edit paper</h2>
            <label>Title<input name="title" defaultValue={editingPaper.paper.title} required /></label>
            <label>Year<input name="year" type="number" min="1900" max="2100" defaultValue={editingPaper.paper.year ?? ""} /></label>
            <label>Research area<select name="research_area" defaultValue={editingPaper.paper.research_area}>{Object.entries(areaLabels).map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select></label>
            <div className="modal-actions"><button type="button" onClick={() => setEditingPaper(null)}>Cancel</button><button className="primary-action" type="submit">Save changes</button></div>
          </form>
        </div>
      )}
    </main>
  );
}
