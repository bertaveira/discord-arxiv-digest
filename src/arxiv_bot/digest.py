"""Render papers and seeds as Discord embed text, split to fit Discord's limits."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import TypeVar

from discord.utils import escape_markdown

from .scoring import Match
from .store import Seed

T = TypeVar("T")

# Discord allows 4096 characters in an embed description; leave some headroom.
DESCRIPTION_LIMIT = 4000

ARXIV_RED = 0xB31B1B
GREY = 0x6B7280

FOOTER = "Each paper: similarity score · most similar seed paper"


def relevant_title(categories: list[str], announced: date, count: int) -> str:
    papers = "paper" if count == 1 else "papers"
    day = f"{announced:%a} {announced.day} {announced:%b %Y}"
    return f"📄 arXiv {' / '.join(categories)} · {day} · {count} {papers}"


def borderline_title(count: int, low: float, high: float) -> str:
    papers = "paper" if count == 1 else "papers"
    return f"Probably not relevant · {count} {papers} scoring {low:.3f}–{high:.3f}"


def link(title: str, url: str) -> str:
    # Square brackets in the title would close the masked link early.
    return f"[{escape_markdown(title.replace('[', '(').replace(']', ')'))}]({url})"


def line(match: Match) -> str:
    return f"• {link(match.paper.title, match.paper.url)} · {match.score:.3f} · {escape_markdown(match.seed)}"


def seed_line(seed: Seed) -> str:
    return f"**{escape_markdown(seed.name)}** · {link(seed.title, seed.url)}"


def render(items: list[T], render_line: Callable[[T], str] = line) -> str:
    return "\n".join(render_line(item) for item in items)


def paginate(items: list[T], render_line: Callable[[T], str] = line, limit: int = DESCRIPTION_LIMIT) -> list[list[T]]:
    """Group items so that each group renders within one embed description."""
    pages: list[list[T]] = []
    size = limit  # forces a new page for the first item
    for item in items:
        length = len(render_line(item)) + 1  # +1 for the newline
        if size + length > limit:
            pages.append([])
            size = 0
        pages[-1].append(item)
        size += length
    return pages
