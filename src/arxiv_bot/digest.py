"""Render papers and seeds as Discord embed text, split to fit Discord's limits."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import TypeVar

import discord
from discord.utils import escape_markdown

from .config import Config
from .scoring import Match
from .store import Seed

T = TypeVar("T")

# Discord allows 4096 characters in an embed description; leave some headroom.
DESCRIPTION_LIMIT = 4000

ARXIV_RED = 0xB31B1B
GREY = 0x6B7280

FOOTER = "Each paper: similarity score · most similar seed paper"


def pages(matches: list[Match], config: Config) -> list[tuple[discord.Embed, list[Match]]]:
    """The embeds of a daily post (relevant, then probably not relevant), each with
    the matches it shows. Empty when nothing reaches the borderline cutoff."""
    relevant = [m for m in matches if m.score >= config.relevant]
    borderline = [m for m in matches if config.borderline <= m.score < config.relevant]
    sections = []
    if relevant:
        title = relevant_title(config.categories, relevant[0].paper.announced, len(relevant))
        sections.append((title, relevant, ARXIV_RED))
    if borderline:
        sections.append((borderline_title(len(borderline), config.borderline, config.relevant), borderline, GREY))
    result = [
        (discord.Embed(title=title if i == 0 else None, description=render(page), color=color), page)
        for title, section, color in sections
        for i, page in enumerate(paginate(section))
    ]
    if result:
        result[-1][0].set_footer(text=FOOTER)
    return result


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
