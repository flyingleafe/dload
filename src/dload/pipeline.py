"""Lazy streaming pipelines over datasets.

A `Pipeline` wraps an immutable DAG of nodes. Operators return new
pipelines; nothing touches storage until iteration. `iter(pipeline)`:

1. plans shard visit orders for every dataset source (shuffled per epoch
   when a `shuffle` sits above the source),
2. checks feasibility: the required resident set (prefetch windows, or the
   whole dataset for `shuffle(full=True)`) must fit the cache budget —
   violations raise `InfeasiblePipelineError` *before* any data moves,
3. executes with a background prefetcher: shards are downloaded ahead of
   the consumer, pinned while in the window, and released to LRU eviction
   once consumed — the sliding-window behavior on small disks, and plain
   warm-cache reads once everything fits locally.

Randomness model: `shuffle(buffer_size=N, seed=s)` shuffles the shard visit
order of every source below it (per epoch), shuffles sample order inside
each shard, and applies a size-N streaming shuffle buffer at its own
position. `shuffle(full=True)` yields a true uniform permutation of the
whole dataset and therefore requires it to fit the cache budget; it must be
applied directly to `Dataset.samples()`.

Combinator architecture, three layers: (0) planner-aware nodes that cannot
be stream combinators — `SourceNode` (shard-backed, worker-sharded) and
`ShuffleNode` (shard visit order + feasibility), plus `RepeatNode`
(re-opens its child per cycle) and `MixNode` (reactive re-weighting when a
stream dries up); (1) the fundamental core — `TransformNode` (the n-ary
`fn(*iters) -> iterator` substrate), `IterableSourceNode`/`from_iterable`
(non-shard leaf), `RandomNode`/`random_stream` (randomness as a stream),
and `scan`/`flat_map`/`zip_with`/`select`/`through`; (2) everything else
(`map`, `filter`, `batch`, `take`, `concat`, `window`, `choice`, `maybe`,
`star_map`, `zip`) — thin wrappers or pure compositions of layer 1.

Determinism: with a seed set, sample order is a pure function of
(seed, epoch, worker); epoch advances on every re-iteration (or is set
explicitly via the torch adapter's `set_epoch`). Without a seed, each
shuffle draws a random seed once at construction time, so orders differ
between pipelines but remain consistent across the DataLoader worker
processes that share one pipeline.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import random
from collections import OrderedDict, deque
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

from .config import format_size
from .errors import InfeasiblePipelineError
from .pack import PackReader, Sample

if TYPE_CHECKING:
    from .cache import PinnedShard, ShardCache
    from .manifest import ShardInfo
    from .repo import Dataset

logger = logging.getLogger("dload")

_MAX_POOL_WORKERS = 32
_MAX_OPEN_READERS = 64


def seeded(*parts: object) -> int:
    """Deterministic 64-bit seed hashed from `parts` (anything with a stable
    `repr`). The canonical per-key randomness idiom: derive every stochastic
    decision from the sample key plus a purpose salt, so it reproduces per
    item with no shared state and no dependence on iteration order:

        random.Random(dload.seeded(key, "crop")).randrange(n)
        np.random.default_rng(dload.seeded(key, "snr")).uniform(0, 20)
    """
    material = repr(parts).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def seeded_rng(*parts: object) -> random.Random:
    """A `random.Random` seeded with `seeded(*parts)`."""
    return random.Random(seeded(*parts))


def _derive_rng(seed: int | None, *salt: object) -> random.Random:
    """Deterministic child RNG from (seed, *salt); fresh entropy if seed is None."""
    if seed is None:
        return random.Random()
    return random.Random(seeded(seed, *salt))


# --------------------------------------------------------------------------
# nodes


class Node:
    children: tuple["Node", ...] = ()

    def iterate(self, ctx: "_Ctx") -> Iterator[Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class _Directive:
    """Planner's instruction to one source for one iteration."""

    order: str  # "sequential" | "shuffled" | "global"
    rng: random.Random | None  # shard/sample order rng (same across workers)


class SourceNode(Node):
    def __init__(self, dataset: "Dataset", prefetch: int = 3) -> None:
        self.dataset = dataset
        self.prefetch = max(2, prefetch)

    def iterate(self, ctx: "_Ctx") -> Iterator[Sample]:
        d = ctx.directives[self]
        shards = list(self.dataset.manifest.shards)
        if d.order == "global":
            # Workers split the *sample* permutation, not the shard list —
            # slicing shards first would degrade the promised whole-dataset
            # permutation into per-worker permutations of disjoint subsets.
            yield from self._iterate_global(ctx, shards, d)
        else:
            if d.order == "shuffled":
                assert d.rng is not None
                d.rng.shuffle(shards)
            yield from self._iterate_windowed(
                ctx, shards[ctx.worker :: ctx.num_workers], d
            )

    def _iterate_windowed(
        self, ctx: "_Ctx", shards: list["ShardInfo"], d: _Directive
    ) -> Iterator[Sample]:
        repo = self.dataset.repo
        pending: deque[Future[PinnedShard]] = deque()
        idx = 0
        try:
            while idx < len(shards) or pending:
                while idx < len(shards) and len(pending) < self.prefetch:
                    pending.append(ctx.pool.submit(repo.open_shard, shards[idx]))
                    idx += 1
                pinned = pending.popleft().result()
                try:
                    with PackReader(pinned.path) as reader:
                        order: Sequence[int] = range(len(reader))
                        if d.order == "shuffled":
                            assert d.rng is not None
                            order = d.rng.sample(range(len(reader)), len(reader))
                        for i in order:
                            yield reader.read(i)
                finally:
                    pinned.release()
        finally:
            for f in pending:
                if not f.cancel():
                    try:
                        f.result().release()
                    except Exception:
                        pass

    def _iterate_global(
        self, ctx: "_Ctx", shards: list["ShardInfo"], d: _Directive
    ) -> Iterator[Sample]:
        """True uniform permutation across the (locally materialized) dataset."""
        repo = self.dataset.repo
        assert d.rng is not None
        prefetches = [
            ctx.pool.submit(lambda s=s: repo.open_shard(s).release())
            for s in shards[ctx.worker :: ctx.num_workers]
        ]
        # The permutation is built over the whole dataset with the shared
        # (seed, epoch)-derived rng, so every worker computes the same one;
        # each then serves its stripe of it.
        order = [(si, j) for si, s in enumerate(shards) for j in range(s.num_samples)]
        d.rng.shuffle(order)
        order = order[ctx.worker :: ctx.num_workers]
        readers: OrderedDict[int, tuple[PackReader, PinnedShard]] = OrderedDict()
        try:
            for si, j in order:
                entry = readers.get(si)
                if entry is None:
                    pinned = repo.open_shard(shards[si])
                    entry = (PackReader(pinned.path), pinned)
                    readers[si] = entry
                    if len(readers) > _MAX_OPEN_READERS:
                        old_reader, old_pin = readers.popitem(last=False)[1]
                        old_reader.close()
                        old_pin.release()
                else:
                    readers.move_to_end(si)
                yield entry[0].read(j)
        finally:
            for reader, pinned in readers.values():
                reader.close()
                pinned.release()
            for f in prefetches:
                f.cancel()


class IterableSourceNode(Node):
    """Non-shard leaf: lifts any iterable (or iterable factory) into the DAG.

    Invisible to the planner — no shards, zero feasibility cost. By default
    the stream is REPLICATED in every DataLoader worker (each worker starts
    it from scratch), which is right for generative/control/random side
    streams and wrong for a primary dataset; `shard=True` gives each worker
    an interleaved `islice` of one logical stream instead.
    """

    def __init__(
        self, source: Callable[[], Iterable[Any]] | Iterable[Any], shard: bool
    ) -> None:
        self.source = source
        self.shard = shard

    def iterate(self, ctx: "_Ctx") -> Iterator[Any]:
        src = self.source() if callable(self.source) else self.source
        it = iter(src)
        if self.shard and ctx.num_workers > 1:
            yield from itertools.islice(it, ctx.worker, None, ctx.num_workers)
        else:
            yield from it


class RandomNode(Node):
    """Endless uniform floats in [0, 1) — randomness as a stream.

    Draws come from `ctx.rng_for(self)`, i.e. deterministic per
    (seed, epoch, worker, node position) — the same derivation `mix` and
    the shuffle buffer use. Per-worker divergence is intentional: this node
    never participates in shard-order derivation, which is the only flow
    that must stay worker-free.
    """

    def __init__(self, seed: int | None) -> None:
        self.seed = seed

    def iterate(self, ctx: "_Ctx") -> Iterator[float]:
        rng = ctx.rng_for(self)
        while True:
            yield rng.random()


class TransformNode(Node):
    """The universal n-ary combinator substrate: `fn(*child_iters) -> iterator`.

    Every derived operator (`map`, `filter`, `batch`, `scan`, `zip_with`,
    `select`, ...) is a thin wrapper over this node. Child generators are
    created eagerly (free — a generator does no work until first pulled)
    and closed here in one place, so `.close()` propagates through any
    combinator chain. `fn` must be picklable for DataLoader workers:
    module-level functions or `functools.partial` over them.
    """

    def __init__(
        self, children: Sequence[Node], fn: Callable[..., Iterator[Any]]
    ) -> None:
        self.children = tuple(children)
        self.fn = fn

    def iterate(self, ctx: "_Ctx") -> Iterator[Any]:
        iters = [c.iterate(ctx) for c in self.children]
        try:
            yield from self.fn(*iters)
        finally:
            for it in iters:
                it.close()  # type: ignore[attr-defined]


# -- combinator bodies (module-level so pipelines pickle to spawn workers) ----


def _run_map(fn: Callable[[Any], Any], it: Iterator[Any]) -> Iterator[Any]:
    for item in it:
        yield fn(item)


def _run_star_map(fn: Callable[..., Any], it: Iterator[Any]) -> Iterator[Any]:
    for item in it:
        yield fn(*item)


def _run_filter(pred: Callable[[Any], bool], it: Iterator[Any]) -> Iterator[Any]:
    for item in it:
        if pred(item):
            yield item


def _run_scan(
    fn: Callable[[Any, Any], Any], init: Any, it: Iterator[Any]
) -> Iterator[Any]:
    state = init
    for item in it:
        state = fn(state, item)
        yield state


def _run_flat_map(
    fn: Callable[[Any], Iterable[Any]], it: Iterator[Any]
) -> Iterator[Any]:
    for item in it:
        yield from fn(item)


def _run_batch(
    size: int,
    collate: Callable[[list[Any]], Any] | None,
    drop_last: bool,
    it: Iterator[Any],
) -> Iterator[Any]:
    batch: list[Any] = []
    for item in it:
        batch.append(item)
        if len(batch) == size:
            yield collate(batch) if collate is not None else batch
            batch = []
    if batch and not drop_last:
        yield collate(batch) if collate is not None else batch


def _run_window(
    size: int, stride: int, drop_last: bool, it: Iterator[Any]
) -> Iterator[list[Any]]:
    buf: list[Any] = []
    skip = 0
    emitted = False
    for item in it:
        if skip:
            skip -= 1
            continue
        buf.append(item)
        if len(buf) == size:
            yield list(buf)
            emitted = True
            if stride >= size:
                buf = []
                skip = stride - size
            else:
                del buf[:stride]
    # A trailing partial window only makes sense when it holds items no full
    # window has emitted: non-overlapping strides, or a stream shorter than
    # one window.
    if not drop_last and buf and (stride >= size or not emitted):
        yield buf


def _run_take(n: int, it: Iterator[Any]) -> Iterator[Any]:
    for _, item in zip(range(n), it):
        yield item


def _run_concat(*its: Iterator[Any]) -> Iterator[Any]:
    for it in its:
        yield from it


def _run_zip_with(fn: Callable[..., Any], *its: Iterator[Any]) -> Iterator[Any]:
    yield from map(fn, *its)  # builtin map stops at the shortest stream


def _run_select(idx_it: Iterator[Any], *stream_its: Iterator[Any]) -> Iterator[Any]:
    n = len(stream_its)
    for i in idx_it:
        if not 0 <= i < n:
            raise ValueError(f"select: index {i} out of range for {n} streams")
        try:
            yield next(stream_its[i])
        except StopIteration:
            return


def _tuple_of(*xs: Any) -> tuple[Any, ...]:
    return xs


def _choice_index(n: int, p: Any, r: float) -> int:
    """Map a uniform draw `r` to a stream index under probability spec `p`
    (scalar = probability of stream 0, valid for 2 streams; else weights)."""
    weights = [p, 1.0 - p] if isinstance(p, (int, float)) else list(p)
    if len(weights) != n:
        raise ValueError(f"choice: got {len(weights)} weights for {n} streams")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("choice: weights must sum to > 0")
    x = r * total
    for i, w in enumerate(weights):
        x -= w
        if x < 0:
            return i
    return n - 1


def _maybe_apply(
    fn: Callable[[Any], Any],
    p: float | Callable[[Any], float],
    item: Any,
    r: float,
) -> Any:
    pt = p(item) if callable(p) else p
    return fn(item) if r < pt else item


def _maybe_apply_stream(
    fn: Callable[[Any], Any], item: Any, r: float, pt: float
) -> Any:
    return fn(item) if r < pt else item


class ShuffleNode(Node):
    def __init__(
        self, child: Node, buffer_size: int, seed: int | None, full: bool
    ) -> None:
        self.children = (child,)
        self.buffer_size = buffer_size
        self.seed = seed
        self.full = full
        # Shard order must agree across DataLoader worker processes even
        # without a user seed, or slicing shards[worker::n] duplicates and
        # drops shards. Drawn once here, shipped to workers via pickling.
        self.auto_seed = random.getrandbits(63)

    @property
    def order_seed(self) -> int:
        return self.auto_seed if self.seed is None else self.seed

    def iterate(self, ctx: "_Ctx") -> Iterator[Any]:
        upstream = self.children[0].iterate(ctx)
        if self.full or self.buffer_size <= 1:
            yield from upstream
            return
        rng = ctx.rng_for(self)
        buf: list[Any] = []
        for item in upstream:
            if len(buf) < self.buffer_size:
                buf.append(item)
                continue
            i = rng.randrange(len(buf))
            buf[i], item = item, buf[i]
            yield item
        rng.shuffle(buf)
        yield from buf


class RepeatNode(Node):
    def __init__(self, child: Node, times: int | None) -> None:
        self.children = (child,)
        self.times = times

    def iterate(self, ctx: "_Ctx") -> Iterator[Any]:
        count = 0
        while self.times is None or count < self.times:
            yield from self.children[0].iterate(ctx)
            count += 1


class MixNode(Node):
    """Sample-level stochastic mixing of several streams."""

    def __init__(
        self,
        children: Sequence[Node],
        weights: Sequence[float] | None,
        seed: int | None,
        until: str,
    ) -> None:
        if until not in ("any", "all"):
            raise ValueError("until must be 'any' or 'all'")
        self.children = tuple(children)
        self.weights = (
            list(weights) if weights is not None else [1.0] * len(self.children)
        )
        if len(self.weights) != len(self.children):
            raise ValueError("weights must match the number of pipelines")
        self.seed = seed
        self.until = until

    def iterate(self, ctx: "_Ctx") -> Iterator[Any]:
        rng = ctx.rng_for(self)
        streams = [c.iterate(ctx) for c in self.children]
        alive = list(range(len(streams)))
        try:
            while alive:
                (k,) = rng.choices(alive, weights=[self.weights[i] for i in alive])
                try:
                    yield next(streams[k])
                except StopIteration:
                    if self.until == "any":
                        return
                    alive.remove(k)
        finally:
            for s in streams:
                s.close()  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# planning & execution


class _Ctx:
    def __init__(
        self,
        directives: dict[SourceNode, _Directive],
        node_ids: dict[Node, int],
        pool: ThreadPoolExecutor,
        epoch: int,
        worker: int,
        num_workers: int,
    ) -> None:
        self.directives = directives
        self._node_ids = node_ids
        self.pool = pool
        self.epoch = epoch
        self.worker = worker
        self.num_workers = num_workers

    def rng_for(self, node: Node) -> random.Random:
        seed = getattr(node, "seed", None)
        return _derive_rng(seed, self.epoch, self.worker, self._node_ids[node])

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


@dataclass
class PlanReport:
    """Result of the feasibility check (`Pipeline.check()`)."""

    ok: bool
    lines: list[str] = field(default_factory=list)

    @property
    def message(self) -> str:
        return "\n".join(self.lines)


def _walk(
    node: Node,
    node_ids: dict[Node, int],
    shuffle_above: ShuffleNode | None,
    sources: list[tuple[SourceNode, ShuffleNode | None]],
) -> None:
    node_ids.setdefault(node, len(node_ids))
    if isinstance(node, SourceNode):
        sources.append((node, shuffle_above))
        return
    if isinstance(node, ShuffleNode):
        if node.full and not isinstance(node.children[0], SourceNode):
            raise InfeasiblePipelineError(
                "shuffle(full=True) must be applied directly to Dataset.samples() — "
                "it re-orders raw storage access and cannot sit above other operators"
            )
        shuffle_above = node
    for child in node.children:
        _walk(child, node_ids, shuffle_above, sources)


class Pipeline:
    """A lazy, re-iterable stream. Each `iter()` is a fresh epoch."""

    def __init__(self, node: Node) -> None:
        self._node = node
        self._epochs = 0

    # -- operators (each returns a new Pipeline) -----------------------------

    def map(self, fn: Callable[[Any], Any]) -> "Pipeline":
        """Apply `fn` to every item."""
        return Pipeline(TransformNode((self._node,), partial(_run_map, fn)))

    def star_map(self, fn: Callable[..., Any]) -> "Pipeline":
        """Apply `fn(*item)` to every item — sugar for tuple streams
        (e.g. after `dload.zip`)."""
        return Pipeline(TransformNode((self._node,), partial(_run_star_map, fn)))

    def filter(self, pred: Callable[[Any], bool]) -> "Pipeline":
        """Keep items where `pred` is true."""
        return Pipeline(TransformNode((self._node,), partial(_run_filter, pred)))

    def scan(self, fn: Callable[[Any, Any], Any], init: Any) -> "Pipeline":
        """Stateful map: fold `state = fn(state, item)` over the stream,
        yielding each new state. For stateful 1:N expansion (windowing,
        packing), have `fn` return `(carry, emitted)` and follow with
        `.flat_map(lambda s: s[1])`."""
        return Pipeline(TransformNode((self._node,), partial(_run_scan, fn, init)))

    def flat_map(self, fn: Callable[[Any], Iterable[Any]]) -> "Pipeline":
        """Expand each item into zero or more items (`fn` returns an
        iterable)."""
        return Pipeline(TransformNode((self._node,), partial(_run_flat_map, fn)))

    def through(self, fn: Callable[[Iterator[Any]], Iterator[Any]]) -> "Pipeline":
        """Escape hatch: pass the whole stream through `fn(iterator) ->
        iterator`, staying inside the DAG (so `.shuffle()/.batch()/...`
        remain available downstream)."""
        return Pipeline(TransformNode((self._node,), fn))

    def zip_with(self, fn: Callable[..., Any], *others: "Pipeline") -> "Pipeline":
        """Method form of `dload.zip_with(fn, self, *others)`."""
        return zip_with(fn, self, *others)

    def maybe(
        self,
        fn: Callable[[Any], Any],
        p: "float | Callable[[Any], float] | Pipeline",
        *,
        seed: int | None = None,
    ) -> "Pipeline":
        """Apply `fn` to a `p`-fraction of items, pass the rest through.
        `p` may be a float, a `fn(item) -> float`, or a Pipeline of floats
        (probability as a stream). Single-pass over the upstream — never
        implemented via `choice`, which would open the upstream twice."""
        r = random_stream(seed)
        if isinstance(p, Pipeline):
            return zip_with(partial(_maybe_apply_stream, fn), self, r, p)
        return zip_with(partial(_maybe_apply, fn, p), self, r)

    def window(
        self, size: int, stride: int | None = None, *, drop_last: bool = False
    ) -> "Pipeline":
        """Group items into (possibly overlapping) list windows of `size`,
        advancing by `stride` (default: `size`, i.e. non-overlapping). A
        trailing partial window is yielded only when its items appeared in
        no full window (`stride >= size`, or a stream shorter than `size`);
        `drop_last=True` suppresses it."""
        if size < 1:
            raise ValueError("window: size must be >= 1")
        stride = size if stride is None else stride
        if stride < 1:
            raise ValueError("window: stride must be >= 1")
        return Pipeline(
            TransformNode((self._node,), partial(_run_window, size, stride, drop_last))
        )

    def shuffle(
        self, buffer_size: int = 4096, *, seed: int | None = None, full: bool = False
    ) -> "Pipeline":
        """Randomize order: shard-order shuffle at the sources below plus a
        streaming buffer of `buffer_size` samples here. `full=True` asks for
        a true uniform permutation instead, which requires the dataset to
        fit in the local cache budget (checked at iteration start)."""
        return Pipeline(ShuffleNode(self._node, buffer_size, seed, full))

    def batch(
        self,
        size: int,
        *,
        collate: Callable[[list], Any] | None = None,
        drop_last: bool = False,
    ) -> "Pipeline":
        """Group items into lists of `size` (or `collate(list)` if given)."""
        return Pipeline(
            TransformNode((self._node,), partial(_run_batch, size, collate, drop_last))
        )

    def take(self, n: int) -> "Pipeline":
        """Stop after `n` items."""
        return Pipeline(TransformNode((self._node,), partial(_run_take, n)))

    def repeat(self, times: int | None = None) -> "Pipeline":
        """Loop the upstream `times` times (forever when None)."""
        return Pipeline(RepeatNode(self._node, times))

    # -- planning -------------------------------------------------------------

    def _plan(
        self, epoch: int
    ) -> tuple[dict[SourceNode, _Directive], dict[Node, int], PlanReport]:
        node_ids: dict[Node, int] = {}
        found: list[tuple[SourceNode, ShuffleNode | None]] = []
        _walk(self._node, node_ids, None, found)

        directives: dict[SourceNode, _Directive] = {}
        for source, shuf in found:
            if shuf is None:
                directives[source] = _Directive("sequential", None)
            else:
                rng = _derive_rng(shuf.order_seed, epoch, node_ids[source])
                order = "global" if shuf.full else "shuffled"
                directives[source] = _Directive(order, rng)

        by_cache: dict[int, tuple[ShardCache, list[str], int]] = {}
        for source, _ in found:
            cache = source.dataset.repo.cache
            m = source.dataset.manifest
            d = directives[source]
            if d.order == "global":
                need = m.total_bytes
                how = "full local materialization (shuffle(full=True))"
            else:
                need = source.prefetch * max((s.size for s in m.shards), default=0)
                how = f"window of {source.prefetch} shards"
            _, lines, total = by_cache.setdefault(id(cache), (cache, [], 0))
            lines.append(f"  {m.name}@{m.version[:12]}: {format_size(need)} ({how})")
            by_cache[id(cache)] = (cache, lines, total + need)

        report = PlanReport(ok=True)
        for cache, lines, total in by_cache.values():
            budget = cache.budget
            header = f"cache {cache.root}: requires {format_size(total)} resident, budget {format_size(budget)}"
            report.lines.append(header)
            report.lines.extend(lines)
            if budget is not None and total > budget:
                report.ok = False
                report.lines.append(
                    "  → infeasible: raise the cache budget (DLOAD_CACHE_BUDGET), lower "
                    "prefetch windows, or use a windowed shuffle instead of full=True"
                )
        return directives, node_ids, report

    def check(self) -> PlanReport:
        """Dry-run the feasibility check without touching storage."""
        return self._plan(epoch=self._epochs)[2]

    # -- execution -------------------------------------------------------------

    def __iter__(self) -> Generator[Any, None, None]:
        return self._iterate(worker=0, num_workers=1)

    def _iterate(
        self, worker: int, num_workers: int, epoch: int | None = None
    ) -> Generator[Any, None, None]:
        if epoch is None:
            epoch = self._epochs
        self._epochs = epoch + 1
        directives, node_ids, report = self._plan(epoch)
        if not report.ok:
            raise InfeasiblePipelineError(
                "pipeline cannot run on this machine:\n" + report.message
            )
        logger.debug(
            "pipeline plan (epoch %d, worker %d/%d):\n%s",
            epoch,
            worker,
            num_workers,
            report.message,
        )

        window_slots = sum(s.prefetch for s in directives) or 1
        pool = ThreadPoolExecutor(
            max_workers=min(max(8, window_slots), _MAX_POOL_WORKERS),
            thread_name_prefix="dload-prefetch",
        )
        ctx = _Ctx(directives, node_ids, pool, epoch, worker, num_workers)

        def run() -> Generator[Any, None, None]:
            try:
                yield from self._node.iterate(ctx)
            finally:
                ctx.close()

        return run()


def mix(
    pipelines: Sequence[Pipeline],
    weights: Sequence[float] | None = None,
    *,
    seed: int | None = None,
    until: str = "any",
) -> Pipeline:
    """Stochastically interleave pipelines with the given weights. Stops when
    the first stream runs dry (`until="any"`) or continues with the rest
    (`until="all"`)."""
    return Pipeline(MixNode([p._node for p in pipelines], weights, seed, until))


def concat(pipelines: Sequence[Pipeline]) -> Pipeline:
    """Chain pipelines end to end."""
    return Pipeline(TransformNode([p._node for p in pipelines], _run_concat))


def from_iterable(
    source: Callable[[], Iterable[Any]] | Iterable[Any], *, shard: bool = False
) -> Pipeline:
    """Lift any iterable into a Pipeline — a generative model's output, a
    parameter schedule, an in-memory list. Pass a factory (called fresh each
    epoch) or a re-iterable container; a raw one-shot generator survives only
    one epoch.

    Contributes nothing to the feasibility check, and is REPLICATED in every
    DataLoader worker by default — right for side/control streams, wrong for
    a primary dataset (`shard=True` slices one logical stream across
    workers instead)."""
    return Pipeline(IterableSourceNode(source, shard))


def random_stream(seed: int | None = None) -> Pipeline:
    """An endless Pipeline of uniform floats in [0, 1). Seeded: a pure
    function of (seed, epoch, worker, position in the DAG). The primitive
    that lets stochastic combinators (`choice`, `.maybe`) be built by
    composition — randomness is just another stream."""
    return Pipeline(RandomNode(seed))


def zip_with(fn: Callable[..., Any], *pipelines: Pipeline) -> Pipeline:
    """Pair streams positionally and yield `fn(a, b, ...)`; stops at the
    shortest. This pairs (unlike `mix`, which interleaves); pairing happens
    within each DataLoader worker, so zip a worker-sharded stream only
    against stateless/generative side streams, or keep aligned pairs inside
    one sample."""
    if not pipelines:
        raise ValueError("zip_with: needs at least one pipeline")
    return Pipeline(
        TransformNode([p._node for p in pipelines], partial(_run_zip_with, fn))
    )


def zip_streams(*pipelines: Pipeline) -> Pipeline:
    """Pair streams positionally into tuples (exported as `dload.zip`)."""
    return zip_with(_tuple_of, *pipelines)


def select(index: Pipeline, *pipelines: Pipeline) -> Pipeline:
    """Demultiplex by a control stream: pull `i` from `index`, yield the
    next item of `pipelines[i]` — only the selected stream advances. Stops
    when `index` or the selected stream is exhausted. The fundamental
    combinator beneath `choice`; for reactive re-weighting over streams
    that run dry, use `mix(until="all")`."""
    if not pipelines:
        raise ValueError("select: needs at least one pipeline")
    return Pipeline(
        TransformNode([index._node, *(p._node for p in pipelines)], _run_select)
    )


def choice(
    pipelines: Sequence[Pipeline],
    p: float | Sequence[float] | Pipeline = 0.5,
    *,
    seed: int | None = None,
) -> Pipeline:
    """Per item, pick ONE stream at random and yield its next item (the
    others don't advance). `p` is the probability of `pipelines[0]` (two
    streams), a weight per stream, or a Pipeline of either — probabilities
    that vary over the course of iteration. Built on
    `select(zip_with(...), random_stream(seed), ...)`; stops when any
    selected stream runs dry (combine with `.repeat()` for endless
    branches)."""
    n = len(pipelines)
    if n < 2:
        raise ValueError("choice: needs at least two pipelines")
    r = random_stream(seed)
    if isinstance(p, Pipeline):
        index = zip_with(partial(_choice_index, n), p, r)
    else:
        _choice_index(n, p, 0.0)  # validate the static spec now, not mid-epoch
        index = r.map(partial(_choice_index, n, p))
    return select(index, *pipelines)
