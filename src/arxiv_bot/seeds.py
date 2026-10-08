"""Managing the seed papers: parsing arXiv IDs, naming, adding, and the first-run
import from config.toml. Errors meant for the person asking are SeedErrors."""

from __future__ import annotations

import re
from dataclasses import dataclass

from discord.utils import escape_markdown

from .config import Config
from .feed import fetch_abstracts
from .scoring import Embedder, rank_seeds
from .store import Seed, Store

NAME_LENGTH = 40
NOT_AN_ID = "That doesn't look like an arXiv ID or link (for example 2308.04079)."

# New-style IDs (2308.04079) and old-style ones (cs/0112017, math.GT/0309136),
# with or without a version, anywhere in the text: bare, "arXiv:...", or a link.
_ARXIV_ID = re.compile(r"(?<![\w.])(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?(?!\d)")


class SeedError(Exception):
    """A problem with a requested seed, worded for the person who asked."""


def parse_arxiv_id(text: str) -> str | None:
    match = _ARXIV_ID.search(text)
    return match.group(1) if match else None


def default_name(title: str) -> str:
    """The part before a colon ("EDGS: Eliminating ..." -> "EDGS"), else the title, shortened."""
    head = title.split(":", 1)[0].strip()
    return head if len(head) <= NAME_LENGTH else head[: NAME_LENGTH - 1].rstrip() + "…"


def check_new_seed(store: Store, text: str, name: str | None) -> str:
    """The arXiv ID in `text`, after the checks that need no network."""
    arxiv_id = parse_arxiv_id(text)
    if arxiv_id is None:
        raise SeedError(NOT_AN_ID)
    if existing := store.seed_by_id(arxiv_id):
        raise SeedError(f"{arxiv_id} is already a seed, named **{escape_markdown(existing.name)}**.")
    if name and store.seed_by_name(name):
        raise SeedError(f"There is already a seed named **{escape_markdown(name)}**. Pick another name.")
    return arxiv_id


def add_seed(store: Store, arxiv_id: str, name: str | None, added_by: str | None) -> Seed:
    """Look the paper up on arXiv and add it. Blocks on the network."""
    found = fetch_abstracts([arxiv_id])
    if arxiv_id not in found:
        raise SeedError(f"arXiv has no paper {arxiv_id}.")
    title, abstract = found[arxiv_id]
    seed = Seed(name or default_name(title), arxiv_id, title, abstract)
    if not store.add_seed(seed, added_by):
        raise SeedError(
            f"There is already a seed named **{escape_markdown(seed.name)}**. Add it again with the `name` option."
        )
    return seed


@dataclass(frozen=True)
class PaperScore:
    arxiv_id: str
    title: str
    itself: Seed | None  # set when the paper is already a seed
    closest: list[tuple[Seed, float]]  # the other seeds, most similar first

    @property
    def score(self) -> float:
        return self.closest[0][1]


def score_paper(store: Store, embedder: Embedder, arxiv_id: str) -> PaperScore:
    """Score one paper against the seeds, as the daily post would. Blocks on the
    network and the model."""
    found = fetch_abstracts([arxiv_id])
    if arxiv_id not in found:
        raise SeedError(f"arXiv has no paper {arxiv_id}.")
    title, abstract = found[arxiv_id]
    # A seed scores 1.0 against itself; compare it with the others instead.
    others = [s for s in store.seeds() if s.arxiv_id != arxiv_id]
    if not others:
        raise SeedError("There are no other seed papers to compare it with. Add one with `/seeds add`.")
    return PaperScore(arxiv_id, title, store.seed_by_id(arxiv_id), rank_seeds((title, abstract), others, store, embedder))


def import_initial_seeds(store: Store, config: Config) -> None:
    """Fill the seed list from config.toml the first time the bot runs. After that
    the list is managed from Discord, so this never runs again (even if every seed
    is later removed)."""
    if store.get_meta("initial_seeds_imported") or not config.initial_seeds:
        return
    ids = list(config.initial_seeds.values())
    found = fetch_abstracts(ids)
    if unknown := [i for i in ids if i not in found]:
        raise ValueError(f"Seed papers in config.toml not found on arXiv: {', '.join(unknown)}")
    for name, arxiv_id in config.initial_seeds.items():
        store.add_seed(Seed(name, arxiv_id, *found[arxiv_id]), added_by="config.toml")
    store.set_meta("initial_seeds_imported", "1")
