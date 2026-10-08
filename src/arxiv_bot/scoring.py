"""Score papers by similarity to the seed papers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import Config
from .embedding import Specter2
from .feed import Paper
from .store import Seed, Store

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str

    def embed(self, papers: list[tuple[str, str]]) -> np.ndarray: ...


@dataclass(frozen=True)
class Match:
    paper: Paper
    score: float  # similarity to the closest seed paper
    seed: str  # that seed's name


def score_papers(papers: list[Paper], config: Config, store: Store, embedder: Embedder | None = None) -> list[Match]:
    """Every paper with a configured announce type, most similar first."""
    candidates = [p for p in papers if p.announce_type in config.announce_types]
    seeds = store.seeds()
    if not seeds:
        log.warning("There are no seed papers; add some with /seeds add")
    if not candidates or not seeds:
        return []
    embedder = embedder or Specter2()

    similarity = embedder.embed([(p.title, p.abstract) for p in candidates]) @ _seed_vectors(seeds, store, embedder).T
    closest = similarity.argmax(axis=1)
    matches = [
        Match(paper, float(similarity[i, closest[i]]), seeds[closest[i]].name) for i, paper in enumerate(candidates)
    ]
    return sorted(matches, key=lambda m: m.score, reverse=True)


def rank_seeds(paper: tuple[str, str], seeds: list[Seed], store: Store, embedder: Embedder) -> list[tuple[Seed, float]]:
    """How similar one (title, abstract) is to each seed, most similar first."""
    similarity = embedder.embed([paper])[0] @ _seed_vectors(seeds, store, embedder).T
    return sorted(zip(seeds, map(float, similarity)), key=lambda pair: pair[1], reverse=True)


def _seed_vectors(seeds: list[Seed], store: Store, embedder: Embedder) -> np.ndarray:
    """Seed embeddings, embedding (and caching) any seed not seen before."""
    cached = store.seed_embeddings(embedder.name)
    if missing := [s for s in seeds if s.arxiv_id not in cached]:
        vectors = dict(zip((s.arxiv_id for s in missing), embedder.embed([(s.title, s.abstract) for s in missing])))
        store.save_seed_embeddings(embedder.name, vectors)
        cached.update(vectors)
    return np.stack([cached[s.arxiv_id] for s in seeds])
