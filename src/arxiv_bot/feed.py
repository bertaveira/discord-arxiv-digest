"""Fetch and parse arXiv's daily announcement feed (rss.arxiv.org) and, for seed
papers, titles and abstracts by ID from the arXiv API."""

from __future__ import annotations

import re
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from email.utils import parsedate_to_datetime

FEED_URL = "https://rss.arxiv.org/rss/{}"
API_URL = "https://export.arxiv.org/api/query?{}"
USER_AGENT = "discord-arxiv-bot/0.1"
_NS = {"arxiv": "http://arxiv.org/schemas/atom", "dc": "http://purl.org/dc/elements/1.1/"}
_ATOM = {"a": "http://www.w3.org/2005/Atom"}


@dataclass(frozen=True)
class Paper:
    arxiv_id: str  # without version, e.g. "2610.08813"
    title: str
    abstract: str
    authors: tuple[str, ...]
    categories: tuple[str, ...]
    announce_type: str  # "new", "cross", "replace" or "replace-cross"
    announced: date

    @property
    def url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"


def fetch_feed(categories: list[str]) -> bytes:
    return _get(FEED_URL.format("+".join(categories)))


def parse_feed(xml: bytes) -> list[Paper]:
    return [_parse_item(item) for item in ET.fromstring(xml).iter("item")]


def fetch_abstracts(arxiv_ids: list[str]) -> dict[str, tuple[str, str]]:
    """(title, abstract) for each ID that exists on arXiv."""
    query = urllib.parse.urlencode({"id_list": ",".join(arxiv_ids), "max_results": len(arxiv_ids)})
    return parse_abstracts(_get(API_URL.format(query)))


def parse_abstracts(xml: bytes) -> dict[str, tuple[str, str]]:
    abstracts = {}
    for entry in ET.fromstring(xml).findall("a:entry", _ATOM):
        arxiv_id = re.sub(r"v\d+$", "", entry.findtext("a:id", "", _ATOM).split("/abs/")[-1])
        abstracts[arxiv_id] = (_squash(entry.findtext("a:title", "", _ATOM)), _squash(entry.findtext("a:summary", "", _ATOM)))
    return abstracts


def _get(url: str, timeout: float = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _parse_item(item: ET.Element) -> Paper:
    # <description> reads "arXiv:2610.08813v1 Announce Type: new \nAbstract: ..."
    _, _, abstract = item.findtext("description", "").partition("Abstract:")
    authors = item.findtext("dc:creator", "", _NS).split(",")
    return Paper(
        arxiv_id=re.sub(r"v\d+$", "", item.findtext("link", "").split("/abs/", 1)[-1]),
        title=_squash(item.findtext("title", "")),
        abstract=_squash(abstract),
        authors=tuple(detex(a.strip()) for a in authors if a.strip()),
        categories=tuple(c.text for c in item.findall("category") if c.text),
        announce_type=item.findtext("arxiv:announce_type", "", _NS),
        announced=parsedate_to_datetime(item.findtext("pubDate", "")).date(),
    )


def _squash(text: str) -> str:
    return " ".join(text.split())


# arXiv keeps author names as typed, often with LaTeX accents: L\"oschner, Garc\'{\i}a, \v{S}imon.
_TEX_LETTERS = {"i": "i", "j": "j", "o": "ø", "O": "Ø", "l": "ł", "L": "Ł", "ss": "ß", "aa": "å", "AA": "Å", "ae": "æ", "AE": "Æ"}
_TEX_ACCENTS = {
    '"': "\u0308", "'": "\u0301", "`": "\u0300", "^": "\u0302", "~": "\u0303", "=": "\u0304", ".": "\u0307",
    "v": "\u030c", "c": "\u0327", "u": "\u0306", "H": "\u030b", "k": "\u0328",
}


def detex(text: str) -> str:
    """Turn LaTeX accents and special letters into Unicode ("L\\"oschner" -> "Löschner")."""
    text = re.sub(r"\\(ss|aa|AA|ae|AE|[ijoOlL])(?![A-Za-z])\s?", lambda m: _TEX_LETTERS[m.group(1)], text)
    text = re.sub(
        r"""\\(["'`^~=.]|[vcuHk](?=[\s{]))\s*\{?\s*([A-Za-z])\}?""",
        lambda m: m.group(2) + _TEX_ACCENTS[m.group(1)],
        text,
    )
    return unicodedata.normalize("NFC", text.replace("{", "").replace("}", ""))
