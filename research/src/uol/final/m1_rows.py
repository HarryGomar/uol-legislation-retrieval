"""M1 historical-memory columns on a variant release that adds rows (DEV-B; D113, 24 Oct 2026).

``uol.round3.m1_memory`` reads its neighbours from a cache built once for the main release's development rows and
refuses any other row set. The DEV-B variant release keeps every main row at its index and appends the recovered
sources (``variants.devb_new_rows``). Here the cached neighbours serve the rows they were built for, and each
recovered row gets no historical neighbours: vote, count and logrank are 0, as for a case whose older *tesis* cite
nothing it is scored on. Every other row's columns are exactly ``m1_memory.columns_for``'s. Any other row difference
(a missing main row, or extra rows that are not DEV-B's recovered sources) still stops the job.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from uol.final import context


def _cached_rows_and_extra(key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(cache rows, cache idx, cache sim, positions of the cache rows in the current dev rows)."""
    from uol.round3 import common as R3, m1_memory as M
    z = np.load(M.neighbours_path(key))
    rows, dev = z["rows"].astype(np.int64), np.asarray(R3.dev_rows(), dtype=np.int64)
    pos = {int(r): i for i, r in enumerate(dev)}
    missing = [int(r) for r in rows if int(r) not in pos]
    extra = sorted(set(dev.tolist()) - set(rows.tolist()))
    if missing:
        raise RuntimeError(f"neighbour cache rows missing from the dev rows ({len(missing)}, e.g. {missing[:3]})")
    allowed = set()
    if extra and context.active().variant == "devb":
        from uol.final import variants
        allowed = set(variants.devb_new_rows())
    if set(extra) - allowed:
        raise RuntimeError(f"dev rows without cached neighbours that are not DEV-B's recovered rows: {sorted(set(extra) - allowed)[:5]}")
    return rows, z["idx"], z["sim"], np.asarray([pos[int(r)] for r in rows], dtype=np.int64)


def config_votes(cfg: dict) -> tuple[sparse.csr_matrix, sparse.csr_matrix, dict]:
    """(vote, count) over the current dev rows; rows without cached neighbours are all zero."""
    from uol.round3 import common as R3, m1_memory as M
    rows, idx, sim, at = _cached_rows_and_extra(cfg["key"])
    mem = M.memory()
    pd_ = R3.publication_dates()
    qd = np.asarray([pd_[int(r)] for r in rows], dtype="datetime64[D]")
    V, Cn = M.votes_from(idx, sim, qd, mem["date"], mem["cited"], int(cfg["k"]), M.DECAYS[cfg["decay"]],
                         len(R3.release().candidate_ids))
    n = len(R3.dev_rows())
    P = sparse.csr_matrix((np.ones(len(at), dtype=np.float32), (at, np.arange(len(at)))), shape=(n, len(at)))
    return (P @ V).tocsr(), (P @ Cn).tocsr(), {"rows_without_neighbours": int(n - len(at))}


def columns_for(cfg: dict, g, h) -> tuple[np.ndarray, dict]:
    """``m1_memory.columns_for`` for a release whose dev rows add DEV-B's recovered rows to the cached ones."""
    from uol.round3 import common as R3, m1_memory as M
    if getattr(g, "is_lockbox", False) or not str(g.role).startswith("dev_fold"):
        raise PermissionError("M1 columns are built for dev groups only")
    V, Cn, info = config_votes(cfg)
    cols = h.empty_columns(len(M.COLUMNS))
    for i, r in enumerate(R3.dev_rows()):
        sl, pc = h.pair_slice(int(r))
        cols[sl] = M.row_columns(V.getrow(i).toarray().ravel(), Cn.getrow(i).toarray().ravel(), pc)
    return cols, info
