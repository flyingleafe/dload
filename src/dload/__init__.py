"""dload — dataset management, storage, caching and streaming for ML training.

Typical usage:

    import dload

    repo = dload.Repository.open()          # from env / dload.toml config
    ds = repo.dataset("librispeech")        # resolves dload.lock / latest

    pipe = (
        ds.samples()
        .shuffle(seed=0)
        .map(my_decode)
        .batch(32)
    )
    for batch in pipe:                      # prefetches, caches, streams
        ...
"""

from . import codecs
from .config import Config, format_size, parse_size
from .errors import (
    CacheFullError,
    ConfigError,
    DloadError,
    InfeasiblePipelineError,
    IntegrityError,
    NotFoundError,
    PackFormatError,
    RemoteError,
)
from .manifest import Manifest, ShardInfo
from .pipeline import (
    Pipeline,
    PlanReport,
    choice,
    concat,
    from_iterable,
    mix,
    random_stream,
    seeded,
    seeded_rng,
    select,
    zip_streams,
    zip_with,
)
from .pipeline import zip_streams as zip  # noqa: A004 — deliberate: dload.zip
from .repo import Dataset, Repository

__version__ = "0.3.0"

__all__ = [
    "CacheFullError",
    "Config",
    "ConfigError",
    "Dataset",
    "DloadError",
    "InfeasiblePipelineError",
    "IntegrityError",
    "Manifest",
    "NotFoundError",
    "PackFormatError",
    "Pipeline",
    "PlanReport",
    "RemoteError",
    "Repository",
    "ShardInfo",
    "choice",
    "codecs",
    "concat",
    "format_size",
    "from_iterable",
    "mix",
    "parse_size",
    "random_stream",
    "seeded",
    "seeded_rng",
    "select",
    "zip",
    "zip_streams",
    "zip_with",
    "__version__",
]
