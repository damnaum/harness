"""Memory system — episodic, semantic, procedural + TF-IDF vector recall.

- Episodic: past task episodes (task -> output -> score), bounded FIFO.
- Semantic: durable facts/preferences keyed by text, recalled by similarity.
- Procedural: learned "how we solved X" patterns (strategy + model that worked).
- Vector: in-memory TF-IDF cosine similarity (stdlib only, no numpy needed).
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class TfidfIndex:
    """Tiny TF-IDF cosine-similarity index over a list of documents."""

    def __init__(self) -> None:
        self.docs: list[str] = []
        self._tf: list[Counter] = []
        self._df: Counter = Counter()

    def rebuild(self, texts: list[str]) -> None:
        """Re-index from scratch — used after eviction so positions stay valid."""
        self.docs = []
        self._tf = []
        self._df = Counter()
        for text in texts:
            self.add(text)

    def add(self, text: str) -> int:
        self.docs.append(text)
        tf = Counter(_tokens(text))
        self._tf.append(tf)
        for term in tf:
            self._df[term] += 1
        return len(self.docs) - 1

    def _idf(self, term: str) -> float:
        n = len(self.docs) or 1
        return math.log((n + 1) / (self._df.get(term, 0) + 1)) + 1.0

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        if not self.docs:
            return []
        q_tf = Counter(_tokens(query))
        scores: list[tuple[int, float]] = []
        for idx, tf in enumerate(self._tf):
            dot = sum(q_tf[t] * tf.get(t, 0) * self._idf(t) ** 2 for t in q_tf)
            q_norm = math.sqrt(sum((c * self._idf(t)) ** 2 for t, c in q_tf.items())) or 1.0
            d_norm = math.sqrt(sum((c * self._idf(t)) ** 2 for t, c in tf.items())) or 1.0
            scores.append((idx, dot / (q_norm * d_norm)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in scores[:top_k] if s > 0.01]


@dataclass
class Episode:
    task: str
    output: str
    task_type: str = "general"
    strategy: str = ""
    model: str = ""
    score: float = 0.0
    ts: float = field(default_factory=time.time)


@dataclass
class Fact:
    text: str
    source: str = "user"
    ts: float = field(default_factory=time.time)


class MemoryStore:
    def __init__(self, episodic_limit: int = 200, semantic_limit: int = 500):
        self.episodic: list[Episode] = []
        self.facts: list[Fact] = []
        self.procedural: dict[str, dict[str, Any]] = {}
        self.episodic_limit = episodic_limit
        self.semantic_limit = semantic_limit
        self._episode_index = TfidfIndex()
        self._fact_index = TfidfIndex()
        # The HTTP server is threaded, so two runs can remember and recall at the
        # same time. Without this lock an eviction rebuild can re-point the index
        # underneath a concurrent search, returning the wrong episode.
        self._lock = threading.RLock()

    # -- episodic -----------------------------------------------------------
    def remember_episode(self, episode: Episode) -> None:
        with self._lock:
            self._remember_episode_locked(episode)

    def _remember_episode_locked(self, episode: Episode) -> None:
        self.episodic.append(episode)
        self._episode_index.add(f"{episode.task} {episode.output}")
        if len(self.episodic) > self.episodic_limit:
            # Evict oldest, then rebuild: the index maps document position to
            # list index, so dropping a prefix invalidates every offset.
            self.episodic = self.episodic[-self.episodic_limit :]
            self._episode_index.rebuild([f"{e.task} {e.output}" for e in self.episodic])

    def recall_episodes(self, query: str, top_k: int = 3, min_score: float = 0.08) -> list[Episode]:
        """Recall previous episodes, excluding the current task echoing itself."""
        import difflib

        normalised_query = " ".join(query.split()).lower()
        with self._lock:
            hits = self._episode_index.search(query, top_k + 3)
            snapshot = list(self.episodic)  # copy: another thread may evict mid-loop
        out: list[Episode] = []
        for i, score in hits:
            if i >= len(snapshot) or score < min_score:
                continue
            ep = snapshot[i]
            same = difflib.SequenceMatcher(
                None, normalised_query, " ".join(ep.task.split()).lower()
            ).ratio()
            if same > 0.9:  # the same request repeated — not useful context
                continue
            out.append(ep)
            if len(out) >= top_k:
                break
        return out

    # -- semantic ------------------------------------------------------------
    def remember_fact(self, text: str, source: str = "user") -> None:
        # Facts are idempotent: re-remembering the same text must not duplicate.
        normalised = " ".join(text.split()).lower()
        with self._lock:
            if any(" ".join(f.text.split()).lower() == normalised for f in self.facts):
                return
            self.facts.append(Fact(text=text, source=source))
            self._fact_index.add(text)
            if len(self.facts) > self.semantic_limit:
                self.facts = self.facts[-self.semantic_limit :]
                self._fact_index.rebuild([f.text for f in self.facts])

    def recall_facts(self, query: str, top_k: int = 5, min_score: float = 0.08) -> list[Fact]:
        with self._lock:
            hits = self._fact_index.search(query, top_k)
            return [self.facts[i] for i, score in hits if i < len(self.facts) and score >= min_score]

    # -- procedural ------------------------------------------------------------
    def remember_procedure(self, task_type: str, strategy: str, model: str, score: float) -> None:
        with self._lock:
            prev = self.procedural.get(task_type)
            if prev is None or score > prev.get("score", 0):
                self.procedural[task_type] = {"strategy": strategy, "model": model, "score": score}

    def best_procedure(self, task_type: str) -> dict[str, Any] | None:
        with self._lock:
            found = self.procedural.get(task_type)
            return dict(found) if found else None

    # -- persistence ----------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "version": 1,
            "episodic": [
                {"task": e.task, "output": e.output, "task_type": e.task_type,
                 "strategy": e.strategy, "model": e.model, "score": e.score, "ts": e.ts}
                for e in self.episodic
            ],
            "facts": [{"text": f.text, "source": f.source, "ts": f.ts} for f in self.facts],
            "procedural": self.procedural,
        }

    def restore(self, data: dict[str, Any]) -> None:
        with self._lock:
            self._restore_locked(data)

    def _restore_locked(self, data: dict[str, Any]) -> None:
        self.episodic, self.facts, self.procedural = [], [], {}
        self._episode_index = TfidfIndex()
        self._fact_index = TfidfIndex()
        for raw in data.get("episodic", []) or []:
            try:
                self.episodic.append(Episode(
                    task=str(raw.get("task", "")), output=str(raw.get("output", "")),
                    task_type=str(raw.get("task_type", "general")), strategy=str(raw.get("strategy", "")),
                    model=str(raw.get("model", "")), score=float(raw.get("score", 0.0)),
                    ts=float(raw.get("ts", time.time())),
                ))
            except (TypeError, ValueError):
                continue
        self._episode_index.rebuild([f"{e.task} {e.output}" for e in self.episodic])
        for raw in data.get("facts", []) or []:
            try:
                self.facts.append(Fact(text=str(raw.get("text", "")), source=str(raw.get("source", "user")),
                                       ts=float(raw.get("ts", time.time()))))
            except (TypeError, ValueError):
                continue
        self._fact_index.rebuild([f.text for f in self.facts])
        proc = data.get("procedural") or {}
        if isinstance(proc, dict):
            self.procedural = {str(k): dict(v) for k, v in proc.items() if isinstance(v, dict)}

    def save(self, path: str | os.PathLike) -> None:
        """Atomic, durable write: a crash never corrupts and never loses a commit.

        Write to a sibling temp file, ``fsync`` it, then ``os.replace`` (atomic on
        POSIX and Windows). The directory fsync makes the rename itself durable;
        it is skipped silently on platforms that refuse to open a directory.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.snapshot(), ensure_ascii=False, indent=1)
        # Unique temp name per writer: two threads (or processes) saving at once
        # used to collide on "<path>.tmp" and one rename failed with ENOENT.
        unique = f"{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex[:8]}"
        tmp = target.with_suffix(f"{target.suffix}.{unique}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, target)
        except BaseException:
            # Never leave a half-written ".tmp" behind to confuse the next run.
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        try:  # pragma: no cover - platform dependent
            dir_fd = os.open(str(target.parent), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        except OSError:
            pass
        finally:
            os.close(dir_fd)

    def load(self, path: str | os.PathLike) -> bool:
        """Load a snapshot; returns False if missing or unreadable (never raises)."""
        target = Path(path)
        if not target.exists():
            return False
        try:
            self.restore(json.loads(target.read_text(encoding="utf-8")))
            return True
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return False

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "episodes": len(self.episodic),
                "facts": len(self.facts),
                "procedures": len(self.procedural),
            }
