"""Slash commands: /seeds list, /seeds add, /seeds remove, /seeds score for the
seed papers, and /digest to post today's papers on demand.

Anyone in the server can use them; admins can restrict them per role or channel
under Server Settings → Integrations without code changes.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import discord
from discord import app_commands
from discord.utils import escape_markdown

from . import digest
from .config import Config
from .feed import fetch_feed, parse_feed
from .posting import deliver
from .scoring import Embedder, score_papers
from .seeds import (
    NAME_LENGTH,
    NOT_AN_ID,
    PaperScore,
    SeedError,
    add_seed,
    check_new_seed,
    parse_arxiv_id,
    score_paper,
)
from .store import Store

log = logging.getLogger(__name__)

# Seed names are typed by users; never let one ping anybody.
NO_PINGS = discord.AllowedMentions.none()


class SeedCommands(app_commands.Group):
    def __init__(self, store: Store, embedder: Embedder, config_path: Path):
        super().__init__(name="seeds", description="The papers that new arXiv papers are compared against")
        self.store = store
        self.embedder = embedder
        self.config_path = config_path

    @app_commands.command(name="list", description="Show the seed papers")
    async def list_seeds(self, interaction: discord.Interaction) -> None:
        seeds = self.store.seeds()
        if not seeds:
            await interaction.response.send_message("There are no seed papers yet. Add one with `/seeds add`.", ephemeral=True)
            return
        pages = digest.paginate(seeds, digest.seed_line)
        for i, page in enumerate(pages):
            embed = discord.Embed(
                title=f"{len(seeds)} seed papers" if i == 0 else None,
                description=digest.render(page, digest.seed_line),
                color=digest.ARXIV_RED,
            )
            if i == len(pages) - 1:
                embed.set_footer(text="Add one with /seeds add · remove one with /seeds remove")
            send = interaction.followup.send if i else interaction.response.send_message
            await send(embed=embed, ephemeral=True)

    @app_commands.command(name="add", description="Add a paper that new arXiv papers are compared against")
    @app_commands.describe(
        paper="arXiv ID or link, for example 2308.04079",
        name="Short name shown in posts (default: the title up to its colon)",
    )
    async def add(
        self,
        interaction: discord.Interaction,
        paper: str,
        name: Optional[app_commands.Range[str, 1, NAME_LENGTH]] = None,
    ) -> None:
        name = (name or "").strip() or None
        try:
            arxiv_id = check_new_seed(self.store, paper, name)
        except SeedError as e:
            await interaction.response.send_message(str(e), ephemeral=True, allowed_mentions=NO_PINGS)
            return

        await interaction.response.defer(thinking=True)  # the arXiv lookup can take a few seconds
        try:
            seed = await asyncio.to_thread(add_seed, self.store, arxiv_id, name, str(interaction.user))
        except SeedError as e:
            await interaction.followup.send(str(e), allowed_mentions=NO_PINGS)
            return
        except Exception:
            log.exception("Adding seed %s failed", arxiv_id)
            await interaction.followup.send("Couldn't reach arXiv. Try again in a minute.")
            return
        await interaction.followup.send(
            f"Added **{escape_markdown(seed.name)}**: {digest.link(seed.title, seed.url)} "
            f"({len(self.store.seeds())} seeds). It counts from the next daily post.",
            allowed_mentions=NO_PINGS,
            suppress_embeds=True,
        )

    @app_commands.command(name="remove", description="Remove a seed paper")
    @app_commands.describe(name="The seed's name, as shown by /seeds list")
    async def remove(self, interaction: discord.Interaction, name: str) -> None:
        seed = self.store.remove_seed(name)
        if seed is None:
            await interaction.response.send_message(
                f"There is no seed named **{escape_markdown(name)}**. See `/seeds list`.",
                ephemeral=True,
                allowed_mentions=NO_PINGS,
            )
            return
        await interaction.response.send_message(
            f"Removed **{escape_markdown(seed.name)}**: {digest.link(seed.title, seed.url)} "
            f"({len(self.store.seeds())} seeds left).",
            allowed_mentions=NO_PINGS,
            suppress_embeds=True,
        )

    @remove.autocomplete("name")
    async def complete_name(self, interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
        current = current.lower()
        return [
            app_commands.Choice(name=f"{s.name} · {s.title}"[:100], value=s.name)
            for s in self.store.seeds()
            if current in s.name.lower() or current in s.title.lower()
        ][:25]

    @app_commands.command(name="score", description="Score a paper against the seeds, to see whether it's worth adding")
    @app_commands.describe(paper="arXiv ID or link, for example 2308.04079")
    async def score(self, interaction: discord.Interaction, paper: str) -> None:
        arxiv_id = parse_arxiv_id(paper)
        if arxiv_id is None:
            await interaction.response.send_message(NOT_AN_ID, ephemeral=True)
            return

        # Loading the model can take half a minute on a small CPU.
        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            result = await asyncio.to_thread(score_paper, self.store, self.embedder, arxiv_id)
        except SeedError as e:
            await interaction.followup.send(str(e), ephemeral=True, allowed_mentions=NO_PINGS)
            return
        except Exception:
            log.exception("Scoring %s failed", arxiv_id)
            await interaction.followup.send("Couldn't score that paper right now. Try again in a minute.", ephemeral=True)
            return
        await interaction.followup.send(embed=score_embed(result, Config.load(self.config_path)), ephemeral=True)


def digest_command(store: Store, embedder: Embedder, config_path: Path) -> app_commands.Command:
    @app_commands.command(name="digest", description="Run today's paper check now and post the result here")
    async def run_digest(interaction: discord.Interaction) -> None:
        # Scoring the whole feed takes a minute or two on a small CPU.
        await interaction.response.defer(thinking=True)
        config = Config.load(config_path)
        try:
            papers = parse_feed(await asyncio.to_thread(fetch_feed, config.categories))
            matches = await asyncio.to_thread(score_papers, papers, config, store, embedder)
        except Exception:
            log.exception("/digest failed")
            await interaction.followup.send("Couldn't check today's papers. Try again in a minute.")
            return

        post = digest.build_post(matches, config)
        if not papers:
            await interaction.followup.send(
                "arXiv's feed is empty right now. It has no announcements on Friday and Saturday nights (New York time)."
            )
        elif post is None:
            below = sum(1 for m in matches if m.score >= config.borderline)
            await interaction.followup.send(
                f"Nothing to post today: none of the {len(matches)} new papers scores {config.relevant:.3f} "
                f"or more ({below} {'is' if below == 1 else 'are'} between {config.borderline:.3f} and {config.relevant:.3f})."
            )
        else:
            # The same post as the daily one, but nothing is marked as seen: the daily post still comes as usual.
            await deliver(post, interaction.channel, lambda embed: interaction.followup.send(embed=embed, wait=True))

    return run_digest


def score_embed(result: PaperScore, config: Config) -> discord.Embed:
    """What the daily post would do with this paper, and what adding it as a seed would change."""
    seed, score = result.closest[0]
    if score >= config.relevant:
        verdict = f"**Relevant**: it would be in the main list (≥ {config.relevant:.3f})."
        advice = "Your seeds already cover it, so adding it would change little."
    elif score >= config.borderline:
        verdict = f"**Probably not relevant**: it would be in the thread under the post ({config.borderline:.3f}–{config.relevant:.3f})."
        advice = "Adding it as a seed would move papers like it into the main list."
    else:
        verdict = f"**Not posted**: it scores below {config.borderline:.3f}."
        advice = "Adding it as a seed would start bringing in papers like it."
    if result.itself:
        advice = f"It is already a seed, named **{escape_markdown(result.itself.name)}**; this compares it with the other seeds."

    closest = "\n".join(f"{s:.3f} · {escape_markdown(seed.name)}" for seed, s in result.closest[:3])
    return discord.Embed(
        title=result.title[:256],
        url=f"https://arxiv.org/abs/{result.arxiv_id}",
        description=f"**{score:.3f}** · closest seed: **{escape_markdown(seed.name)}**\n{verdict}\n{advice}\n\n"
        f"**Closest seeds**\n{closest}",
        color=digest.ARXIV_RED,
    )
