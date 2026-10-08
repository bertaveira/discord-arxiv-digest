"""SQLite state: the seed papers, papers already processed (so repeated runs never
score or post a paper twice), and cached seed-paper embeddings."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Seed:
    name: str  # shown in posts; unique, case-insensitively
    arxiv_id: str
    title: str
    abstract: str

    @property
    def url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"


class Store:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Shared by the event loop (slash commands) and the scoring thread, so every
        # method holds the lock.
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS seen (
                arxiv_id TEXT PRIMARY KEY,
                seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS seeds (
                name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                arxiv_id TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                added_by TEXT,
                added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS seed_embeddings (
                arxiv_id TEXT NOT NULL,
                model TEXT NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (arxiv_id, model)
            );
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )

    def unseen(self, arxiv_ids: Iterable[str]) -> set[str]:
        """The subset of `arxiv_ids` not processed yet."""
        ids = set(arxiv_ids)
        if not ids:
            return set()
        placeholders = ",".join("?" * len(ids))
        with self._lock:
            rows = self._db.execute(f"SELECT arxiv_id FROM seen WHERE arxiv_id IN ({placeholders})", list(ids))
            return ids - {row[0] for row in rows}

    def mark_seen(self, arxiv_ids: Iterable[str]) -> None:
        with self._lock:
            self._db.executemany("INSERT OR IGNORE INTO seen (arxiv_id) VALUES (?)", ((i,) for i in arxiv_ids))
            self._db.commit()

    def seeds(self) -> list[Seed]:
        """All seeds, in the order they were added."""
        with self._lock:
            rows = self._db.execute("SELECT name, arxiv_id, title, abstract FROM seeds ORDER BY rowid")
            return [Seed(*row) for row in rows]

    def seed_by_id(self, arxiv_id: str) -> Seed | None:
        return self._one_seed("arxiv_id = ?", arxiv_id)

    def seed_by_name(self, name: str) -> Seed | None:
        return self._one_seed("name = ?", name)

    def add_seed(self, seed: Seed, added_by: str | None = None) -> bool:
        """False if the name or the paper is already a seed."""
        with self._lock:
            cursor = self._db.execute(
                "INSERT OR IGNORE INTO seeds (name, arxiv_id, title, abstract, added_by) VALUES (?, ?, ?, ?, ?)",
                (seed.name, seed.arxiv_id, seed.title, seed.abstract, added_by),
            )
            self._db.commit()
            return cursor.rowcount == 1

    def remove_seed(self, name: str) -> Seed | None:
        seed = self.seed_by_name(name)
        if seed is not None:
            with self._lock:
                self._db.execute("DELETE FROM seeds WHERE arxiv_id = ?", (seed.arxiv_id,))
                self._db.commit()
        return seed

    def seed_embeddings(self, model: str) -> dict[str, np.ndarray]:
        with self._lock:
            rows = self._db.execute("SELECT arxiv_id, vector FROM seed_embeddings WHERE model = ?", (model,))
            return {arxiv_id: np.frombuffer(vector, dtype=np.float32) for arxiv_id, vector in rows}

    def save_seed_embeddings(self, model: str, vectors: dict[str, np.ndarray]) -> None:
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO seed_embeddings (arxiv_id, model, vector) VALUES (?, ?, ?)",
                ((arxiv_id, model, v.astype(np.float32).tobytes()) for arxiv_id, v in vectors.items()),
            )
            self._db.commit()

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
            return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
            self._db.commit()

    def _one_seed(self, where: str, value: str) -> Seed | None:
        with self._lock:
            row = self._db.execute(f"SELECT name, arxiv_id, title, abstract FROM seeds WHERE {where}", (value,)).fetchone()
            return Seed(*row) if row else None
