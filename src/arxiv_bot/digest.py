"""Render papers and seeds as Discord embed text, split to fit Discord's limits.

A daily post is one channel message listing the relevant papers, plus a thread
on it holding the "probably not relevant" list.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import TypeVar

import discord
from discord.utils import escape_markdown

from .config import Config
from .scoring import Match
from .store import Seed

T = TypeVar("T")

# Discord allows 4096 characters in an embed description; leave some headroom.
# Longer lists continue in further messages.
DESCRIPTION_LIMIT = 4000

MAX_AUTHORS = 10

ARXIV_RED = 0xB31B1B
GREY = 0x6B7280

FOOTER = "Score = similarity to the closest seed paper · check any paper with /seeds score"

Page = tuple[discord.Embed, list[Match]]  # an embed and the matches it shows


@dataclass
class Post:
    main: list[Page]  # the channel message(s) with the relevant papers; never empty
    thread_name: str
    thread: list[Page]  # the probably-not-relevant list; may be empty


def build_post(matches: list[Match], config: Config) -> Post | None:
    """The daily post for these scored papers, or None when none is relevant (the
    probably-not-relevant list only appears as a thread on a post). `matches` are
    all scored papers, so its length is the number of papers considered."""
    relevant = [m for m in matches if m.score >= config.relevant]
    borderline = [m for m in matches if config.borderline <= m.score < config.relevant]
    if not relevant:
        return None

    lead = f"-# {len(relevant)} relevant from {len(matches)} new {' / '.join(config.categories)} papers"
    main = _pages(relevant, entry, lead, ARXIV_RED, title=f"arXiv · {day(matches[0].paper.announced)}")
    main[-1][0].set_footer(text=FOOTER)

    thread_lead = f"-# Scoring {config.borderline:.3f}–{config.relevant:.3f}. These didn't make the main list."
    thread = _pages(borderline, compact, thread_lead, GREY) if borderline else []
    return Post(main, f"Probably not relevant · {_count(len(borderline))}", thread)


def _pages(matches: list[Match], render_line: Callable[[Match], str], lead: str, color: int, title: str | None = None) -> list[Page]:
    pages = paginate(matches, render_line, DESCRIPTION_LIMIT - len(lead) - 1)
    return [
        (
            discord.Embed(
                title=title if i == 0 else None,
                description="\n".join(([lead] if i == 0 else []) + [render_line(m) for m in page]),
                color=color,
            ),
            page,
        )
        for i, page in enumerate(pages)
    ]


def day(announced: date) -> str:
    return f"{announced:%A} {announced.day} {announced:%B}"


def _count(n: int) -> str:
    return f"{n} paper" if n == 1 else f"{n} papers"


def link(title: str, url: str) -> str:
    # Square brackets in the title would close the masked link early.
    return f"[{escape_markdown(title.replace('[', '(').replace(']', ')'))}]({url})"


def author_list(authors: tuple[str, ...]) -> str:
    """All authors, or the first MAX_AUTHORS and "+N more" for large collaborations."""
    names = [re.sub(r"\s*\(.*?\)", "", a).strip() for a in authors]  # drop "(Affiliation)"
    shown = ", ".join(names[:MAX_AUTHORS])
    return shown if len(names) <= MAX_AUTHORS else f"{shown} +{len(names) - MAX_AUTHORS} more"


def entry(match: Match) -> str:
    """The title as a bold link, then the authors and then the closest seed and score,
    each on its own line in small grey text."""
    lines = [f"**{link(match.paper.title, match.paper.url)}**"]
    if authors := author_list(match.paper.authors):
        lines.append(f"-# {escape_markdown(authors)}")
    lines.append(f"-# ≈ {escape_markdown(match.seed)} · {match.score:.3f}")
    return "\n".join(lines)


def compact(match: Match) -> str:
    return f"{link(match.paper.title, match.paper.url)} · {match.score:.3f}"


def seed_line(seed: Seed) -> str:
    return f"**{escape_markdown(seed.name)}** · {link(seed.title, seed.url)}"


def render(items: list[T], render_line: Callable[[T], str]) -> str:
    return "\n".join(render_line(item) for item in items)


def paginate(items: list[T], render_line: Callable[[T], str], limit: int = DESCRIPTION_LIMIT) -> list[list[T]]:
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
