"""Entry point: `arxiv-bot run` starts the bot, `arxiv-bot dry-run` previews scores."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .config import Config
from .feed import fetch_feed, parse_feed
from .scoring import score_papers
from .seeds import import_initial_seeds
from .store import Store


def main() -> None:
    parser = argparse.ArgumentParser(prog="arxiv-bot")
    parser.add_argument(
        "--config", type=Path, default=Path(os.environ.get("ARXIV_BOT_CONFIG", "config/config.toml"))
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run", help="start the Discord bot")
    dry_run = commands.add_parser("dry-run", help="score today's papers and print them without posting")
    dry_run.add_argument("--feed-file", type=Path, help="read a saved feed instead of downloading it")
    dry_run.add_argument("--below", type=int, default=10, help="also show this many papers below the borderline cutoff")
    args = parser.parse_args()

    db_path = Path(os.environ.get("ARXIV_BOT_DB", "data/arxiv-bot.db"))
    if args.command == "run":
        _run(args.config, db_path)
    else:
        _dry_run(args.config, db_path, args.feed_file, args.below)


def _run(config_path: Path, db_path: Path) -> None:
    from .bot import run

    token = os.environ.get("DISCORD_TOKEN")
    channel_id = os.environ.get("DISCORD_CHANNEL_ID")
    if not token or not channel_id:
        sys.exit("DISCORD_TOKEN and DISCORD_CHANNEL_ID must be set")
    run(token, int(channel_id), config_path, db_path)


def _dry_run(config_path: Path, db_path: Path, feed_file: Path | None, below: int) -> None:
    config = Config.load(config_path)
    store = Store(db_path)
    import_initial_seeds(store, config)
    papers = parse_feed(feed_file.read_bytes() if feed_file else fetch_feed(config.categories))
    # Uses the bot's seed list, but marks nothing as seen.
    matches = score_papers(papers, config, store)

    sections = [
        (f"Relevant (≥ {config.relevant:.3f})", lambda s: s >= config.relevant),
        (f"Probably not relevant ({config.borderline:.3f}–{config.relevant:.3f})",
         lambda s: config.borderline <= s < config.relevant),
    ]
    for heading, in_section in sections:
        section = [m for m in matches if in_section(m.score)]
        print(f"\n== {heading}: {len(section)}")
        for m in section:
            print(f"  {m.score:.3f}  {m.paper.title}\n         {m.paper.url}  ≈ {m.seed}")
    print(f"\n== Next {below} below the cutoff (not posted)")
    for m in [m for m in matches if m.score < config.borderline][:below]:
        print(f"  {m.score:.3f}  {m.paper.title}  ≈ {m.seed}")
    print(f"\n{len(matches)} papers scored")
