"""Tiny helpers for the edges: turn values into sample field bytes and back.

Kept deliberately minimal — fields are opaque bytes and users bring their
own codecs (soundfile, PIL, ...). These cover the ubiquitous cases.
"""

from __future__ import annotations

import io
import json
from typing import Any


def json_bytes(obj: Any) -> bytes:
    """Compact UTF-8 JSON."""
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def json_from(data: bytes) -> Any:
    return json.loads(data.decode("utf-8"))


def npy_bytes(array: Any) -> bytes:
    """numpy array -> .npy bytes (np.save into a BytesIO). Imports numpy
    lazily so the core library has no hard numpy dependency."""
    import numpy as np

    buf = io.BytesIO()
    np.save(buf, array)
    return buf.getvalue()


def npy_from(data: bytes) -> Any:
    """.npy bytes -> numpy array (allow_pickle=False)."""
    import numpy as np

    return np.load(io.BytesIO(data), allow_pickle=False)


def text_bytes(s: str) -> bytes:
    return s.encode("utf-8")


def text_from(data: bytes) -> str:
    return data.decode("utf-8")
