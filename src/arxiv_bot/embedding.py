"""SPECTER2 paper embeddings on CPU.

SPECTER2 is trained on citation links to place related papers close together,
using the "proximity" adapter for paper-to-paper similarity.
"""

from __future__ import annotations

import gc
import threading

import numpy as np


class Specter2:
    name = "allenai/specter2"

    def __init__(self, batch_size: int = 16):
        # Imported here: torch and transformers are slow to import and only needed
        # once there is something to embed.
        import torch
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer

        self._torch = torch
        self._batch_size = batch_size
        self._tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self._model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        # Logs "There are adapters available but none are activated" while loading;
        # that is before set_active takes effect and is harmless.
        self._model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
        self._model.eval()

    def embed(self, papers: list[tuple[str, str]]) -> np.ndarray:
        """Unit-length embeddings for (title, abstract) pairs."""
        texts = [title + self._tokenizer.sep_token + abstract for title, abstract in papers]
        batches = []
        with self._torch.inference_mode():
            for i in range(0, len(texts), self._batch_size):
                tokens = self._tokenizer(
                    texts[i : i + self._batch_size],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                batches.append(self._model(**tokens).last_hidden_state[:, 0].numpy())
        vectors = np.concatenate(batches)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class SharedSpecter2:
    """One SPECTER2 for the whole bot (daily posts and /seeds score). It loads on
    first use and is freed after `idle_seconds` unused, so the model's ~1.4 GB
    isn't held all day for a job that runs once a day. Calls run one at a time."""

    name = Specter2.name

    def __init__(self, idle_seconds: float = 600, load=Specter2):
        self._idle_seconds = idle_seconds
        self._load = load
        self._model = None
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def embed(self, papers: list[tuple[str, str]]) -> np.ndarray:
        with self._lock:
            if self._timer:
                self._timer.cancel()
            if self._model is None:
                self._model = self._load()
            try:
                return self._model.embed(papers)
            finally:
                self._timer = threading.Timer(self._idle_seconds, self._release)
                self._timer.daemon = True
                self._timer.start()

    def _release(self) -> None:
        with self._lock:
            self._model = None
        gc.collect()
