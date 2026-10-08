"""Discord client that posts each day's matching papers to one channel and
handles the /seeds commands."""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, time, timedelta
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import tasks

from . import digest
from .commands import SeedCommands, digest_command
from .config import Config
from .embedding import SharedSpecter2
from .feed import Paper, fetch_feed, parse_feed
from .scoring import Match, score_papers
from .seeds import import_initial_seeds
from .store import Store

log = logging.getLogger(__name__)


class ArxivBot(discord.Client):
    def __init__(self, channel_id: int, config_path: Path, store: Store):
        super().__init__(intents=discord.Intents(guilds=True))
        self.channel_id = channel_id
        self.config_path = config_path
        self.store = store
        self.embedder = SharedSpecter2()
        self.tree = app_commands.CommandTree(self)
        self._lock = asyncio.Lock()
        self._catch_up: asyncio.Task | None = None

        config = Config.load(config_path)
        self._post_time = config.post_time
        times = [_add_hours(config.post_time, h) for h in range(config.retry_hours + 1)]
        self.daily = tasks.loop(time=times)(self.post_new_papers)

    async def setup_hook(self) -> None:
        try:
            await asyncio.to_thread(import_initial_seeds, self.store, Config.load(self.config_path))
        except Exception:
            log.exception("Importing the seeds from config.toml failed; retrying before the next post")

        # Register the commands for this server only, where they appear immediately.
        channel = await self.fetch_channel(self.channel_id)
        guild = discord.Object(id=channel.guild.id)
        self.tree.add_command(SeedCommands(self.store, self.embedder, self.config_path), guild=guild)
        self.tree.add_command(digest_command(self.store, self.embedder, self.config_path), guild=guild)
        await self.tree.sync(guild=guild)

        self.daily.start()
        # If we (re)start after today's post time, catch up instead of waiting a day.
        now = datetime.now(self._post_time.tzinfo)
        if now.time() >= self._post_time.replace(tzinfo=None):
            self._catch_up = asyncio.create_task(self._post_when_ready())

    async def _post_when_ready(self) -> None:
        await self.wait_until_ready()
        await self.post_new_papers()

    async def post_new_papers(self) -> None:
        async with self._lock:
            try:
                await self._post_new_papers()
            except Exception:
                # An uncaught exception would stop the daily loop for good.
                log.exception("Posting papers failed")

    async def _post_new_papers(self) -> None:
        config = Config.load(self.config_path)
        # Embedding takes a minute or two on a small CPU; keep it off the event loop
        # so the connection to Discord stays alive.
        new, matches = await asyncio.to_thread(self._score_new_papers, config)
        pages = digest.pages(matches, config)
        log.info("%d new papers, %d posted", len(new), sum(len(page) for _, page in pages))
        if pages:
            channel = self.get_channel(self.channel_id) or await self.fetch_channel(self.channel_id)
            for embed, page in pages:
                await channel.send(embed=embed)
                self.store.mark_seen(m.paper.arxiv_id for m in page)
        self.store.mark_seen(p.arxiv_id for p in new)

    def _score_new_papers(self, config: Config) -> tuple[list[Paper], list[Match]]:
        import_initial_seeds(self.store, config)  # no-op once done
        papers = parse_feed(fetch_feed(config.categories))
        unseen = self.store.unseen(p.arxiv_id for p in papers)
        new = [p for p in papers if p.arxiv_id in unseen]
        return new, score_papers(new, config, self.store, self.embedder)


def _add_hours(t: time, hours: int) -> time:
    return (datetime.combine(date.today(), t) + timedelta(hours=hours)).timetz()


def run(token: str, channel_id: int, config_path: Path, db_path: Path) -> None:
    bot = ArxivBot(channel_id, config_path, Store(db_path))
    bot.run(token, root_logger=True)
