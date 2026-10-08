"""Behaviour settings from config.toml. Secrets come from the environment instead."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class Config:
    categories: list[str]
    announce_types: list[str]
    post_time: time  # timezone-aware
    retry_hours: int
    relevant: float
    borderline: float
    initial_seeds: dict[str, str]  # name -> arXiv ID (without version); imported on first run

    @classmethod
    def load(cls, path: str | Path) -> Config:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        hour, minute = map(int, raw["post_time"].split(":"))
        config = cls(
            categories=raw["categories"],
            announce_types=raw["announce_types"],
            post_time=time(hour, minute, tzinfo=ZoneInfo(raw["timezone"])),
            retry_hours=raw.get("retry_hours", 0),
            relevant=raw["similarity"]["relevant"],
            borderline=raw["similarity"]["borderline"],
            initial_seeds={
                name: re.sub(r"v\d+$", "", arxiv_id.strip()) for name, arxiv_id in raw.get("seeds", {}).items()
            },
        )
        if config.borderline > config.relevant:
            raise ValueError("similarity.borderline must not be higher than similarity.relevant")
        return config
