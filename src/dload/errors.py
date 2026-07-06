"""Exception hierarchy for dload."""


class DloadError(Exception):
    """Base class for all dload errors."""


class ConfigError(DloadError):
    """Missing or invalid configuration (remote endpoint, bucket, sizes)."""


class RemoteError(DloadError):
    """Storage backend failure (network, auth, missing object)."""


class NotFoundError(RemoteError):
    """A dataset, version, ref or object does not exist."""


class PackFormatError(DloadError):
    """A pack file is corrupt or has an unsupported format."""


class IntegrityError(DloadError):
    """Downloaded content does not match its expected digest."""


class CacheFullError(DloadError):
    """The cache budget cannot accommodate the working set (all pinned)."""


class InfeasiblePipelineError(DloadError):
    """The pipeline's required resident set exceeds the local cache budget.

    Raised at iteration start with an explanation of what to change
    (larger budget, smaller prefetch window, windowed instead of full
    shuffle, ...).
    """
