"""Sending a post to Discord: the relevant list as messages, then the probably-not-
relevant list in a thread started on the last of them. Shared by the daily post
and /digest so both look the same."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

import discord

from .digest import Post
from .scoring import Match

log = logging.getLogger(__name__)

Send = Callable[[discord.Embed], Awaitable[discord.Message]]


async def deliver(
    post: Post,
    channel: discord.abc.Messageable,
    send: Send,
    mark: Callable[[list[Match]], None] = lambda page: None,
) -> None:
    """Post `post` in `channel`. `send` posts one embed and returns the message (a
    channel send, or an interaction follow-up); `mark` is called with each page's
    papers once that page is posted. If the bot may not use threads, the thread
    list is posted with `send` as well, so it is never lost."""
    for embed, page in post.main:
        message = await send(embed)
        mark(page)
    if not post.thread:
        return

    thread = await _start_thread(channel, message, post.thread_name)
    target: Send = (lambda embed: thread.send(embed=embed)) if thread else send
    for embed, page in post.thread:
        try:
            await target(embed)
        except discord.Forbidden as e:
            if target is send:
                raise
            log.warning("Couldn't post in the thread (%s); posting the list without it", e.text)
            target = send
            await target(embed)
        mark(page)


async def _start_thread(channel: discord.abc.Messageable, message: discord.Message, name: str) -> discord.Thread | None:
    create_thread = getattr(channel, "create_thread", None)
    if create_thread is None:  # e.g. /digest run inside a thread
        return None
    try:
        return await create_thread(name=name, message=message, auto_archive_duration=1440)
    except discord.HTTPException as e:
        # Most likely missing Create Public Threads.
        log.warning("Couldn't create the thread (%s); posting the list without it", e.text)
        return None
