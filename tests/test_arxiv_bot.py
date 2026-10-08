import asyncio
import time as clock
from dataclasses import replace
from datetime import date, time
from pathlib import Path
from unittest import mock

import discord
import numpy as np
import pytest

from arxiv_bot import bot as bot_module
from arxiv_bot import commands, digest, seeds
from arxiv_bot.commands import SeedCommands, digest_command, score_embed
from arxiv_bot.config import Config
from arxiv_bot.feed import detex, parse_abstracts, parse_feed
from arxiv_bot.scoring import Match, score_papers
from arxiv_bot.embedding import SharedSpecter2
from arxiv_bot.seeds import (
    PaperScore,
    SeedError,
    add_seed,
    check_new_seed,
    default_name,
    import_initial_seeds,
    parse_arxiv_id,
    score_paper,
)
from arxiv_bot.store import Seed, Store

ROOT = Path(__file__).parent.parent
FIXTURE = Path(__file__).parent / "fixtures" / "feed.xml"
CONFIG = ROOT / "config" / "config.toml"  # cutoffs 0.93 / 0.91

SPLATS = Seed("Splats", "9999.00001", "Splat seed", "...")
NERF = Seed("NeRF", "9999.00002", "NeRF seed", "...")
ARXIV = {
    "9999.00001": ("Splat seed", "..."),
    "9999.00002": ("NeRF seed", "..."),
    "2504.13204": ("EDGS: Eliminating Densification for Efficient Convergence of 3DGS", "..."),
}


class FakeEmbedder:
    """Looks vectors up by title, so scores are known in advance."""

    name = "fake"
    vectors = {
        "Splat seed": [1, 0, 0],
        "Far seed": [-1, 0, 0],
        "NeRF seed": [0, 1, 0],
        "Fast Gaussian Splatting for [Large] Scenes": [0.9, 0.1, 0],
        "Relighting Neural Radiance Fields": [0.1, 0.95, 0],
        "A Revised Ray-Tracing Paper": [1, 0, 0],
        "An Article on Image Classification": [0.1, 0.1, 1],
        "EDGS: Eliminating Densification for Efficient Convergence of 3DGS": [0.92, 0.39, 0],
    }

    def __init__(self):
        self.calls = []

    def embed(self, papers):
        self.calls.append([title for title, _ in papers])
        v = np.array([self.vectors[title] for title, _ in papers], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class FakeInteraction:
    user = "alice"

    def __init__(self):
        self.response = mock.AsyncMock()
        self.followup = mock.AsyncMock()


@pytest.fixture
def papers():
    return parse_feed(FIXTURE.read_bytes())


@pytest.fixture
def config():
    return Config(
        categories=["cs.CV", "cs.GR"],
        announce_types=["new", "cross"],
        post_time=time(8, 0),
        retry_hours=0,
        relevant=0.95,
        borderline=0.9,
        initial_seeds={"Splats": "9999.00001", "NeRF": "9999.00002"},
    )


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "bot.db")


@pytest.fixture
def seeded(store):
    store.add_seed(SPLATS)
    store.add_seed(NERF)
    return store


@pytest.fixture
def arxiv(monkeypatch):
    """Stand-in for the arXiv API; records which IDs were requested."""
    requested = []

    def fetch(ids):
        requested.append(ids)
        return {i: ARXIV[i] for i in ids if i in ARXIV}

    monkeypatch.setattr(seeds, "fetch_abstracts", fetch)
    return requested


# Feed


def test_parse_feed(papers):
    first = papers[0]
    assert [p.arxiv_id for p in papers] == ["2610.00001", "2610.00002", "2601.00003", "2610.00004"]
    assert [p.announce_type for p in papers] == ["new", "cross", "replace", "new"]
    assert first.title == "Fast Gaussian Splatting for [Large] Scenes"
    assert first.abstract == "We speed up 3DGS training on city-scale captures."
    assert first.authors == ("Ada Lovelace", "Alan Turing")
    assert first.categories == ("cs.CV", "cs.GR")
    assert first.announced == date(2026, 10, 8)
    assert first.url == "https://arxiv.org/abs/2610.00001"


def test_parse_abstracts():
    xml = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry>
        <id>http://arxiv.org/abs/2308.04079v1</id>
        <title>3D Gaussian Splatting for
          Real-Time Radiance Field Rendering</title>
        <summary>  Radiance Field methods have recently revolutionized novel-view synthesis.  </summary>
    </entry></feed>"""
    assert parse_abstracts(xml) == {
        "2308.04079": (
            "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
            "Radiance Field methods have recently revolutionized novel-view synthesis.",
        )
    }


# Scoring


def test_score_papers(papers, config, seeded):
    matches = score_papers(papers, config, seeded, FakeEmbedder())
    # The "replace" entry is skipped; the rest are sorted by score.
    assert [(m.paper.arxiv_id, m.seed) for m in matches] == [
        ("2610.00002", "NeRF"),
        ("2610.00001", "Splats"),
        ("2610.00004", "Splats"),
    ]
    assert matches[0].score == pytest.approx(0.95 / np.hypot(0.1, 0.95))
    assert matches[-1].score < 0.2


def test_seed_embeddings_are_cached(papers, config, seeded):
    score_papers(papers, config, seeded, FakeEmbedder())
    embedder = FakeEmbedder()
    score_papers(papers, config, seeded, embedder)
    assert embedder.calls == [[p.title for p in papers if p.announce_type != "replace"]]


def test_no_seeds_or_no_candidates_skip_the_model(papers, config, store, seeded):
    assert score_papers(papers, config, Store(":memory:")) == []
    assert score_papers([p for p in papers if p.announce_type == "replace"], config, seeded) == []


# Seeds


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2404.09591", "2404.09591"),
        ("  arXiv:2404.09591v3 ", "2404.09591"),
        ("https://arxiv.org/abs/2404.09591v2", "2404.09591"),
        ("https://arxiv.org/pdf/2404.09591v1.pdf", "2404.09591"),
        ("https://arxiv.org/abs/cs/0112017", "cs/0112017"),
        ("math.GT/0309136", "math.GT/0309136"),
        ("3DGS", None),
        ("12345", None),
    ],
)
def test_parse_arxiv_id(text, expected):
    assert parse_arxiv_id(text) == expected


def test_default_name():
    assert default_name("EDGS: Eliminating Densification for Efficient Convergence of 3DGS") == "EDGS"
    long = default_name("3D Gaussian Splatting for Real-Time Radiance Field Rendering")
    assert long == "3D Gaussian Splatting for Real-Time Rad…"
    assert len(long) == seeds.NAME_LENGTH


def test_check_new_seed(seeded):
    assert check_new_seed(seeded, "arxiv.org/abs/2504.13204", None) == "2504.13204"
    with pytest.raises(SeedError, match="doesn't look like"):
        check_new_seed(seeded, "EDGS", None)
    with pytest.raises(SeedError, match="already a seed, named \\*\\*Splats"):
        check_new_seed(seeded, "9999.00001v2", None)
    with pytest.raises(SeedError, match="already a seed named"):
        check_new_seed(seeded, "2504.13204", "splats")  # names are case-insensitive


def test_add_seed(seeded, arxiv):
    seed = add_seed(seeded, "2504.13204", None, "alice")
    assert seed.name == "EDGS"
    assert seeded.seeds()[-1] == seed
    with pytest.raises(SeedError, match="no paper 2612.99999"):
        add_seed(seeded, "2612.99999", None, "alice")


def test_remove_seed(seeded):
    assert seeded.remove_seed("nerf") == NERF
    assert seeded.remove_seed("NeRF") is None
    assert seeded.seeds() == [SPLATS]


def test_import_initial_seeds_runs_once(config, store, arxiv):
    import_initial_seeds(store, config)
    assert [s.name for s in store.seeds()] == ["Splats", "NeRF"]
    store.remove_seed("Splats")
    store.remove_seed("NeRF")
    import_initial_seeds(store, config)
    assert store.seeds() == []
    assert len(arxiv) == 1


def test_import_initial_seeds_rejects_unknown_ids(config, store, arxiv):
    with pytest.raises(ValueError, match="1234.56789"):
        import_initial_seeds(store, replace(config, initial_seeds={"Typo": "1234.56789"}))
    assert store.get_meta("initial_seeds_imported") is None


# Slash commands


def run(command, group, *args):
    asyncio.run(group.get_command(command).callback(group, *args))


def test_command_add(seeded, arxiv):
    group, interaction = SeedCommands(seeded, FakeEmbedder(), CONFIG), FakeInteraction()
    run("add", group, interaction, "https://arxiv.org/abs/2504.13204", None)
    interaction.response.defer.assert_awaited_once()
    message = interaction.followup.send.call_args.args[0]
    assert message.startswith("Added **EDGS**: [EDGS: Eliminating Densification")
    assert "(3 seeds)" in message


def test_command_add_rejects_duplicates_privately(seeded, arxiv):
    group, interaction = SeedCommands(seeded, FakeEmbedder(), CONFIG), FakeInteraction()
    run("add", group, interaction, "9999.00001", None)
    interaction.response.send_message.assert_awaited_once()
    assert interaction.response.send_message.call_args.kwargs["ephemeral"] is True
    assert arxiv == []


def test_command_list_and_remove(seeded):
    group, interaction = SeedCommands(seeded, FakeEmbedder(), CONFIG), FakeInteraction()
    run("list", group, interaction)
    embed = interaction.response.send_message.call_args.kwargs["embed"]
    assert embed.title == "2 seed papers"
    assert "**NeRF** · [NeRF seed](https://arxiv.org/abs/9999.00002)" in embed.description

    interaction = FakeInteraction()
    run("remove", group, interaction, "NeRF")
    assert "Removed **NeRF**" in interaction.response.send_message.call_args.args[0]
    assert [s.name for s in seeded.seeds()] == ["Splats"]


def test_command_score(seeded, arxiv):
    group, interaction = SeedCommands(seeded, FakeEmbedder(), CONFIG), FakeInteraction()
    run("score", group, interaction, "arxiv.org/abs/2504.13204")
    assert interaction.response.defer.call_args.kwargs["ephemeral"] is True
    embed = interaction.followup.send.call_args.kwargs["embed"]
    assert embed.title.startswith("EDGS: Eliminating")
    assert embed.description.startswith("**0.921** · closest seed: **Splats**")
    assert "in the thread under the post" in embed.description
    assert seeded.seed_by_id("2504.13204") is None  # scoring never adds it


def test_score_paper_compares_a_seed_with_the_others(seeded, arxiv):
    result = score_paper(seeded, FakeEmbedder(), "9999.00001")
    assert result.itself == SPLATS
    assert [seed.name for seed, _ in result.closest] == ["NeRF"]


@pytest.mark.parametrize(
    "score, expected",
    [(0.95, "**Relevant**"), (0.92, "**Probably not relevant**"), (0.80, "**Not posted**")],
)
def test_score_embed_verdicts(score, expected):
    result = PaperScore("2504.13204", "EDGS", None, [(SPLATS, score), (NERF, 0.5)])
    description = score_embed(result, Config.load(CONFIG)).description
    assert expected in description
    assert f"{score:.3f} · Splats\n0.500 · NeRF" in description


def test_shared_embedder_loads_once_and_frees_when_idle():
    loads = []

    def load():
        loads.append(1)
        return FakeEmbedder()

    shared = SharedSpecter2(idle_seconds=0.05, load=load)
    shared.embed([("Splat seed", "")])
    shared.embed([("NeRF seed", "")])
    assert len(loads) == 1
    clock.sleep(0.2)
    shared.embed([("Splat seed", "")])
    assert len(loads) == 2


def low_borderline_config(tmp_path):
    """The repo config with borderline 0.05, so the fixture's classification paper (0.099) lands in the thread."""
    path = tmp_path / "config.toml"
    path.write_text(CONFIG.read_text().replace("borderline = 0.91", "borderline = 0.05"))
    return path


def test_command_digest_shows_todays_post_privately(tmp_path, papers, seeded, monkeypatch):
    monkeypatch.setattr(commands, "fetch_feed", lambda categories: FIXTURE.read_bytes())
    interaction = FakeInteraction()
    asyncio.run(digest_command(seeded, FakeEmbedder(), low_borderline_config(tmp_path)).callback(interaction))
    (main, thread) = interaction.followup.send.call_args_list
    assert main.args[0] == "Today's post as it would look (nothing was posted). In the channel:"
    assert main.kwargs["embed"].description.startswith("-# 2 relevant from 3 new cs.CV / cs.GR papers")
    assert thread.args[0] == "In a thread named **Probably not relevant · 1 paper**:"
    assert all(call.kwargs["ephemeral"] for call in (main, thread))
    assert seeded.unseen(p.arxiv_id for p in papers) == {p.arxiv_id for p in papers}


def test_command_digest_on_an_empty_feed(seeded, monkeypatch):
    monkeypatch.setattr(commands, "fetch_feed", lambda categories: b"<rss><channel></channel></rss>")
    interaction = FakeInteraction()
    asyncio.run(digest_command(seeded, FakeEmbedder(), CONFIG).callback(interaction))
    assert interaction.followup.send.call_args.args[0].startswith("arXiv's feed is empty right now")


# Posting


class FakeThread:
    def __init__(self, refuse=False):
        self.embeds, self.refuse = [], refuse

    async def send(self, embed):
        if self.refuse:
            raise discord.Forbidden(mock.Mock(status=403, reason="Forbidden"), "Missing Permissions")
        self.embeds.append(embed)


class FakeChannel:
    def __init__(self, thread=None, thread_error=None):
        self.embeds, self.threads = [], []
        self.thread, self.thread_error = thread, thread_error

    async def send(self, embed):
        self.embeds.append(embed)
        channel = self

        class Message:
            async def create_thread(self, name, auto_archive_duration):
                if channel.thread_error:
                    raise channel.thread_error
                channel.threads.append(name)
                return channel.thread

        return Message()


def post_with(tmp_path, seeded, monkeypatch, channel):
    seeded.set_meta("initial_seeds_imported", "1")
    monkeypatch.setattr(bot_module, "fetch_feed", lambda categories: FIXTURE.read_bytes())

    async def run_once():
        bot = bot_module.ArxivBot(1, low_borderline_config(tmp_path), seeded)
        bot.embedder = FakeEmbedder()
        monkeypatch.setattr(bot, "get_channel", lambda channel_id: channel)
        await bot._post_new_papers()

    asyncio.run(run_once())


def test_bot_posts_relevant_papers_with_the_rest_in_a_thread(tmp_path, papers, seeded, monkeypatch):
    thread = FakeThread()
    channel = FakeChannel(thread)
    post_with(tmp_path, seeded, monkeypatch, channel)
    assert [e.title for e in channel.embeds] == ["arXiv · Thursday 8 October"]
    assert "Relighting Neural Radiance Fields" in channel.embeds[0].description
    assert channel.threads == ["Probably not relevant · 1 paper"]
    assert "An Article on Image Classification" in thread.embeds[0].description
    assert seeded.unseen(p.arxiv_id for p in papers) == set()


@pytest.mark.parametrize(
    "channel",
    [
        FakeChannel(thread_error=discord.Forbidden(mock.Mock(status=403, reason="Forbidden"), "Missing Permissions")),
        FakeChannel(thread=FakeThread(refuse=True)),
    ],
    ids=["cannot create thread", "cannot post in thread"],
)
def test_bot_falls_back_to_the_channel_without_thread_permissions(tmp_path, papers, seeded, monkeypatch, channel):
    post_with(tmp_path, seeded, monkeypatch, channel)
    assert len(channel.embeds) == 2
    assert "An Article on Image Classification" in channel.embeds[1].description
    assert seeded.unseen(p.arxiv_id for p in papers) == set()


# Digest


def test_build_post(papers):
    config = replace(Config.load(CONFIG), relevant=0.95, borderline=0.9)
    matches = [Match(papers[0], 0.97, "3DGS"), Match(papers[1], 0.92, "NeRF"), Match(papers[3], 0.5, "Other")]
    post = digest.build_post(matches, config)
    ((main, shown),) = post.main
    assert [m.seed for m in shown] == ["3DGS"]
    assert main.title == "arXiv · Thursday 8 October"
    assert main.description == (
        "-# 1 relevant from 3 new cs.CV / cs.GR papers\n"
        "**[Fast Gaussian Splatting for (Large) Scenes](https://arxiv.org/abs/2610.00001)**\n"
        "-# Ada Lovelace, Alan Turing\n"
        "-# ≈ 3DGS · 0.970"
    )
    assert main.footer.text == digest.FOOTER
    assert post.thread_name == "Probably not relevant · 1 paper"
    ((thread, _),) = post.thread
    assert thread.description == (
        "-# Scoring 0.900–0.950. These didn't make the main list.\n"
        "[Relighting Neural Radiance Fields](https://arxiv.org/abs/2610.00002) · 0.920"
    )


def test_no_post_without_relevant_papers(papers):
    config = replace(Config.load(CONFIG), relevant=0.95, borderline=0.9)
    assert digest.build_post([Match(papers[1], 0.92, "NeRF")], config) is None
    assert digest.build_post([Match(papers[3], 0.5, "Other")], config) is None


def test_command_digest_says_why_nothing_would_be_posted(seeded, monkeypatch):
    monkeypatch.setattr(commands, "fetch_feed", lambda categories: FIXTURE.read_bytes())
    interaction = FakeInteraction()
    seeded.remove_seed("Splats")
    seeded.remove_seed("NeRF")
    seeded.add_seed(Seed("Far", "9999.00009", "Far seed", "..."))
    asyncio.run(digest_command(seeded, FakeEmbedder(), CONFIG).callback(interaction))
    assert interaction.followup.send.call_args.args[0].startswith(
        "Nothing would be posted today: none of the 3 new papers scores 0.930 or more"
    )


def test_author_list():
    assert digest.author_list(("Ada Lovelace",)) == "Ada Lovelace"
    assert digest.author_list(("Ada Lovelace", "Alan Turing (Bletchley Park)")) == "Ada Lovelace, Alan Turing"
    many = tuple(f"Author {i}" for i in range(1, 35))
    assert digest.author_list(many) == ", ".join(f"Author {i}" for i in range(1, 11)) + " +24 more"


def test_long_posts_continue_in_more_messages(papers):
    config = replace(Config.load(CONFIG), relevant=0.95, borderline=0.9)
    authors = tuple(f"Author Number {i}" for i in range(12))
    many = [Match(replace(papers[0], arxiv_id=f"2610.{i:05d}", authors=authors), 0.96, "3DGS") for i in range(60)]
    post = digest.build_post(many, config)
    assert len(post.main) > 1
    embeds = [embed for embed, _ in post.main]
    assert [e.title for e in embeds] == ["arXiv · Thursday 8 October"] + [None] * (len(embeds) - 1)
    assert embeds[0].description.startswith("-# 60 relevant") and not embeds[1].description.startswith("-#")
    assert [e.footer.text for e in embeds] == [None] * (len(embeds) - 1) + [digest.FOOTER]
    assert all(len(e.description) <= digest.DESCRIPTION_LIMIT for e in embeds)
    assert sum(len(shown) for _, shown in post.main) == 60


def test_detex_author_names():
    assert detex(r'Fabian L\"oschner') == "Fabian Löschner"
    assert detex(r"Ram\'on Garc\'{\i}a") == "Ramón García"
    assert detex(r"\v{S}imon {\o}stergaard, Erd\H{o}s") == "Šimon østergaard, Erdős"


def test_digest_paginate_respects_limit(papers):
    many = [Match(replace(papers[0], arxiv_id=f"2610.{i:05d}"), 0.95, "3DGS") for i in range(200)]
    pages = digest.paginate(many, digest.entry, limit=1000)
    assert sum(len(p) for p in pages) == 200
    assert all(len(digest.render(p, digest.entry)) <= 1000 for p in pages)


# Store and config


def test_store_seen(store):
    assert store.unseen(["a", "b"]) == {"a", "b"}
    store.mark_seen(["a"])
    store.mark_seen(["a"])
    assert store.unseen(["a", "b"]) == {"b"}


def test_store_seed_embeddings(tmp_path):
    Store(tmp_path / "bot.db").save_seed_embeddings("m", {"x": np.array([0.6, 0.8])})
    cached = Store(tmp_path / "bot.db").seed_embeddings("m")
    assert list(cached) == ["x"]
    np.testing.assert_allclose(cached["x"], [0.6, 0.8])
    assert Store(tmp_path / "bot.db").seed_embeddings("other") == {}


def test_config_file_loads():
    config = Config.load(ROOT / "config" / "config.toml")
    assert config.borderline <= config.relevant
    assert config.initial_seeds["3DGS"] == "2308.04079"


def test_config_rejects_inverted_cutoffs(tmp_path):
    text = (ROOT / "config" / "config.toml").read_text().replace("borderline = 0.91", "borderline = 0.99")
    (tmp_path / "config.toml").write_text(text)
    with pytest.raises(ValueError, match="borderline"):
        Config.load(tmp_path / "config.toml")
