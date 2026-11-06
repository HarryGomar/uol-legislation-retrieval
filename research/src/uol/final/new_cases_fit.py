"""All-development learned rankers for the new cases (E13, D161, D162): fitted once before collection and saved.

On the new cases the three learned rankers (S0, S_full, S_full_ext) are models trained on all 4,812 development cases:
the role the frozen embedding-push run calls the lockbox group (group 5, training folds 0-4, tree seed
``fusion_ltr.SEED + 5000``, pair-sampling seed + 999). They are fitted with the record's own code
(``adapters.ltr.LTRRefit``'s harness, ``round3.features.Harness.fit_group``, on the development pair table) in the
record's bundle profile (the bundle on Drive, corrected labels, 4 threads). Every configuration comes from the record or
the frozen run, so no label is read to choose it:

* HGB: the frozen lockbox-group hyper-parameters of the pipeline manifest (``LTR_noemb_hgb`` for S0, ``LTR_emb_hgb``
  for S_full and S_full_ext). The bundle's copies keep the five folds only, so the entry is read from the frozen run.
* Topic columns: per family, the grid point with the highest mean inner objective over the record's fifteen inner
  selections (three rankers x five folds): LSA 1,024, LDA similarity 400, LDA document model 100, doc2vec 300.
  Fitted once on the catalogue and all development facts, shared by the three rankers, saved.
* M1 (S_full_ext): dense key, half-life 5 years, k 10: the best configuration on all development cases in the M1
  arm's table (``configs_tried``), and the one three of the five folds chose.

Outputs (``<models>``): per ranker the pickled HGB (``model.pkl``), ``info.json`` and ``dry.npz`` (the batch output of
the fit for the dry-run cases), the topic-model pickles, the instrument-prior row of the group, and ``MODELS.json``
with every sha256.

``NewCaseRanker`` is the per-case path: a case's pool, the frozen feature columns, the topic columns (one single-case
call per family with the inference seed reset first, so a case's columns never depend on the cases scored before it),
the M1 columns and the saved model. ``dryrun`` checks it on development cases against the saved model's own output,
and ``spread`` refits fold 0 the record's way so its development nDCG@10 can be set beside the record's.

    py -3.12 -m uol.final.new_cases fit|dryrun-rankers|spread   (see uol.final.new_cases)
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pickle
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RANKERS = ("S0", "S_full", "S_full_ext")
LOCKBOX = 5                                   # group index of the all-development models (embedding_push, round 2)
THREADS = 4                                   # the record's fits (receipts: OMP_NUM_THREADS = UOL_FINAL_THREADS = 4)
DRY_N = 20                                    # development cases of the dry run (DEV fold 0, the first score rows)
TOPIC_SIZES = {"lsa": {"dim": 1024}, "lda_sim": {"topics": 400},
               "lda_dm": {"topics": 100, "lam": 0.7, "mu": 1000.0}, "doc2vec": {"size": 300}}
M1_CFG = {"key": "dense", "decay": "hl5", "k": 10}
RECORD = ROOT / "experiments" / "runs" / "final_20261025" / "store_dev_20261025b"
FROZEN_PIPES = ROOT / "experiments" / "runs" / "embedding_push_20261004" / "fusion" / "pipelines"
M1_ARM = ROOT / "experiments" / "runs" / "round3_best_case_20261013" / "arms" / "M1" / "RESULT.json"
BUNDLE_DATA = Path(os.environ.get("UOL_NEW_BUNDLE_DATA", r"bundle\UOL_FINAL_RUN\data"))
# the M1 memory (older tesis only, no case row) as staged and verified for the timing job (T2, timing_20261026c)
M1_STAGE = ROOT / ".cache" / "timing_stage_20261026c"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")


def log(msg: str, path: Path | None = None) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# --------------------------------------------------------------------------------------------- configurations


def topic_choice(receipts: Path = RECORD / "receipts") -> dict[str, dict[str, Any]]:
    """Per topic family, the grid point with the highest mean inner objective over the record's inner selections of
    the three rankers' five folds (ties to the earlier grid point, as ``adapters.base.select``)."""
    by: dict[str, dict[str, list[float]]] = {}
    grid: dict[str, list[str]] = {}
    for sid in RANKERS:
        rec = json.loads((receipts / f"{sid}__DEV.json").read_text(encoding="utf-8"))
        for f in rec["folds"]:
            for kind in TOPIC_SIZES:
                for h in f["fit"][kind]["selection"]:
                    key = json.dumps(h["params"], sort_keys=True)
                    by.setdefault(kind, {}).setdefault(key, []).append(float(h["objective"]))
                    if key not in grid.setdefault(kind, []):
                        grid[kind].append(key)
    out = {}
    for kind, vals in by.items():
        means = {k: float(np.mean(v)) for k, v in vals.items()}
        best = max(means.values())
        pick = next(k for k in grid[kind] if means[k] >= best - 1e-12)
        out[kind] = {"params": json.loads(pick), "mean_inner_objective": means, "n": {k: len(v) for k, v in vals.items()}}
    return out


def m1_choice(path: Path = M1_ARM) -> dict[str, Any]:
    """The M1 configuration with the best score on all development cases in the arm's table."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    best = max(d["configs_tried"], key=lambda x: x["dev_score"])
    picks = [json.dumps(v, sort_keys=True) for v in d["selected_config"].values()]
    return {"config": best["config"], "dev_score": best["dev_score"],
            "folds_choosing_it": picks.count(json.dumps(best["config"], sort_keys=True))}


def frozen_lockbox_cfg(pipeline: str) -> dict[str, Any]:
    man = FROZEN_PIPES / pipeline / "manifest.json"
    rec = json.loads(man.read_text(encoding="utf-8"))["configuration"]["lockbox"]
    return {"config": dict(rec["config"]), "manifest": str(man), "manifest_sha256": sha256(man),
            "frozen_final_train_pairs": (rec.get("evidence") or {}).get("final_train_pairs")}


def check_choices() -> dict[str, Any]:
    """The constants above equal what the record and the frozen files give (raises otherwise)."""
    t = topic_choice()
    bad = {k: v["params"] for k, v in t.items() if v["params"] != TOPIC_SIZES[k]}
    m = m1_choice()
    if bad or m["config"] != M1_CFG:
        raise RuntimeError(f"configuration constants differ from the record: topic {bad}, M1 {m['config']}")
    return {"topic": t, "m1": m, "hgb": {p: frozen_lockbox_cfg(p) for p in ("LTR_noemb_hgb", "LTR_emb_hgb")}}


# --------------------------------------------------------------------------------------------- profile


def activate(work: Path):
    """The record's bundle profile (corrected labels) with its work folder ``work``; the M1 memory from the staging."""
    if os.environ.get("UOL_FINAL_THREADS") != str(THREADS):
        raise RuntimeError(f"set UOL_FINAL_THREADS={THREADS} before the process starts (topic workers are read at import)")
    os.environ["UOL_FINAL_LABELS"] = "corrected"
    from uol.final import bundle_overlay, context, cost
    bundle_overlay.install(BUNDLE_DATA)          # the packed arrays served from their verified local originals
    prof = context.activate_bundle(BUNDLE_DATA, store=Path(work) / "store", work=Path(work))
    cost.set_threads(THREADS)
    point_m1_stage()
    return prof


def point_m1_stage() -> dict[str, str]:
    """``timing_sources.staged_path('m1/...')`` -> the verified staging folder (sha256 checked against its MANIFEST)."""
    from uol.final import timing_sources as TS
    man = json.loads((M1_STAGE / "MANIFEST.json").read_text(encoding="utf-8"))
    files = man.get("data") or man.get("files") or {}
    got = {}
    for rel in ("m1/dense_memory.npz", "m1/dense_memory_vectors.npy.zsh"):
        want = files.get(rel) or files.get("data/" + rel)
        want = want.get("sha256") if isinstance(want, dict) else want
        have = sha256(M1_STAGE / "data" / rel)
        if want is None or want != have:
            raise RuntimeError(f"staged {rel}: sha256 {have} != MANIFEST {want}")
        got[rel] = have
        ent = files.get(rel)
        if isinstance(ent, dict) and ent.get("unpacked"):
            unpacked = M1_STAGE / "data" / ent["unpacked"]
            if unpacked.exists() and sha256(unpacked) != ent["unpacked_sha256"]:
                raise RuntimeError(f"staged {ent['unpacked']}: sha256 differs from MANIFEST")
    if not getattr(TS.staged_path, "_new_cases", False):
        orig = TS.staged_path

        def staged_path(rel: str) -> Path:
            return M1_STAGE / "data" / rel if rel.startswith("m1/") else orig(rel)
        staged_path._new_cases = True
        TS.staged_path = staged_path
    return got


def dev_population():
    from uol.final import populations
    return populations.load("DEV", None)


def dry_rows(pop) -> np.ndarray:
    from uol.final import guard
    f0 = next(f for f in pop.folds() if f.k == 0)
    return guard.dev_rows_only(np.asarray(f0.score_rows[:DRY_N]), "dry-run rows")


def all_dev_fold(pop, score_rows: np.ndarray):
    """The fold object of the all-development models: every development row trains, ``score_rows`` are scored."""
    from uol.embedding_push import common as ep
    from uol.final import guard, populations
    train = guard.dev_rows_only(ep.dev_rows(), "all development rows")
    tune = populations.labelled(ep.population_rows(pop.eval_population, train))
    return populations.FoldSpec(LOCKBOX, "lockbox", np.asarray(score_rows), train, tune, [])


def prepared(sid: str, pop):
    """The record's scorer for ``sid``, prepared, with the all-development configurations installed."""
    from uol.final import registry
    sc = registry.make(sid)
    sc.prepare(pop)
    if sc.method != "hgb" or sc.table != "dev" or sc.seed_offset != 0:
        raise RuntimeError(f"{sid}: expected the record's hgb refit on the development table")
    for m in sc.extra:
        m.grid = [dict(TOPIC_SIZES[m.kind])]
    sc.h.cfgs["lockbox"] = frozen_lockbox_cfg(sc.h.name)["config"]
    return sc


def lockbox_group(sc, rows: np.ndarray):
    from uol.embedding_push import fusion_ltr as fl
    g = replace(sc.h.ctx.groups[LOCKBOX], rows=np.asarray(rows))
    if g.role != "lockbox" or g.idx != LOCKBOX or tuple(g.train_folds) != (0, 1, 2, 3, 4):
        raise RuntimeError(f"group {LOCKBOX} is not the all-development group: {g.role} {g.train_folds}")
    if len(fl.train_rows_of(sc.h.ctx, g)) != len(sc.h.ctx.dev):
        raise RuntimeError("the all-development group does not train on every development row")
    return g


def extra_columns(sc, fold, g) -> tuple[np.ndarray, dict[str, Any]]:
    """``LTRRefit._extra_columns`` for the all-development group: the topic members fitted on all development facts
    and scored on every development row; M1 with the all-development configuration (its columns are the same for
    every development group, ``m1_memory.columns_for``)."""
    from uol.embedding_push import common as ep
    from uol.final import guard
    from uol.final.adapters import ltr
    from uol.round3 import e2_adapter as e2, m1_memory
    info: dict[str, Any] = {}
    blocks = []
    dev = guard.dev_rows_only(ep.dev_rows(), "pair-table rows")
    for m in sc.extra:
        info[m.sid] = m.fit(fold)
        cols = sc.h.empty_columns(3)
        with ltr._pair_table(sc.h.F, sc.table):
            for s in range(0, len(dev), 256):
                rows = dev[s:s + 256]
                pos, c = e2.full_columns(m.score(rows), rows)
                cols[pos] = c
        blocks.append(cols)
    if sc.extension == "m1":
        g0 = next(x for x in sc.h.groups if x.role == "dev_fold0")
        blocks.append(np.asarray(m1_memory.columns_for(M1_CFG, g0, sc.h), dtype=np.float32))
        info["m1"] = dict(M1_CFG)
    return np.hstack(blocks).astype(np.float32), info


def topic_key(m) -> tuple[str, Path]:
    from uol.final.adapters import topic
    _, key = topic.model_key(m.kind, topic.param_value(m.kind, m.params), m.articles, m.extra)
    return key, topic._cache_dir() / f"{key}.pkl"


# --------------------------------------------------------------------------------------------- fit


def fit(models: Path, work: Path, rankers: Sequence[str] = RANKERS) -> dict[str, Any]:
    """Fit and save the all-development rankers (once: refuses a ranker whose model exists)."""
    models = Path(models)
    logf = models / "fit.log"
    choices = check_choices()
    activate(work)
    from uol.embedding_push import common as ep, fusion_ltr as fl
    pop = dev_population()
    dry = dry_rows(pop)
    fold = all_dev_fold(pop, dry)
    out_all: dict[str, Any] = {"choices": choices, "dry_rows": [int(r) for r in dry], "rankers": {}}
    table = None
    for sid in rankers:
        out = models / sid
        if (out / "model.pkl").exists():
            raise FileExistsError(f"{out / 'model.pkl'} exists: the all-development models are fitted once")
        t0 = time.perf_counter()
        sc = prepared(sid, pop)
        g = lockbox_group(sc, dry)
        log(f"{sid}: prepared ({time.perf_counter() - t0:.0f}s), fitting topic / M1 columns", logf)
        extra, info = extra_columns(sc, fold, g)
        log(f"{sid}: extra columns {extra.shape} ({time.perf_counter() - t0:.0f}s)", logf)
        idx, scores = sc.h.fit_group(g, extra, seed_offset=0)
        model, Xout, idx_out = sc.h.last_fit
        tr_idx = sc.h._base["lockbox"][0]
        fit_s = time.perf_counter() - t0
        out.mkdir(parents=True, exist_ok=False)
        with (out / "model.pkl").open("wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        np.savez(out / "dry.npz", rows=dry, idx_out=idx_out, pq=np.asarray(sc.h.F.pq[idx_out]),
                 pc=np.asarray(sc.h.F.pc[idx_out]), X=Xout, decision=scores)
        topics = {}
        for m in sc.extra:
            key, src = topic_key(m)
            dst = models / "topic" / f"{key}.pkl"
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
            if sha256(dst) != sha256(src):
                raise RuntimeError(f"topic model {key}: saved copy differs from the cache")
            topics[m.kind] = {"key": key, "params": m.params, "file": f"topic/{key}.pkl", "sha256": sha256(dst),
                              "model": info[m.sid]["model"]}
        if table is None:
            table, _ = fl.instrument_tables(sc.h.ctx)
            np.save(models / "instrument_prior_lockbox.npy", np.asarray(table[LOCKBOX], dtype=np.float32))
        rec = {"system": sid, "variant": sc.variant, "pipeline": sc.h.name, "config": sc.h.cfgs["lockbox"],
               "seed": int(fl.SEED + 1000 * LOCKBOX), "pair_seed": int(fl.SEED + 1000 * LOCKBOX + 999),
               "train_rows": int(len(fold.train_rows)), "train_pairs": int(len(tr_idx)),
               "frozen_lockbox_train_pairs": frozen_lockbox_cfg(sc.h.name)["frozen_final_train_pairs"],
               "feature_columns": int(Xout.shape[1]), "hgb_cols": int(len(sc.h.v.hgb_cols)),
               "pool_ids": list(sc.h.v.pool_ids), "feature_ids": list(sc.h.v.feature_ids),
               "extra": [m.sid for m in sc.extra], "extension": sc.extension, "m1": info.get("m1"), "topics": topics,
               "fit_seconds": round(fit_s, 1), "n_iter": int(getattr(model, "n_iter_", -1)),
               "files": {"model.pkl": sha256(out / "model.pkl"), "dry.npz": sha256(out / "dry.npz")}}
        write_json(out / "info.json", rec)
        out_all["rankers"][sid] = rec
        log(f"{sid}: fitted in {fit_s:.0f}s, {len(tr_idx)} training pairs, {Xout.shape[1]} columns", logf)
        sc.h._base = {}
        del sc, extra, Xout
    out_all["instrument_prior_lockbox.npy"] = sha256(models / "instrument_prior_lockbox.npy")
    out_all["bundle"] = {"data": str(BUNDLE_DATA), "BUNDLE.json": sha256(BUNDLE_DATA / "BUNDLE.json"),
                         "label_corrections.json": sha256(BUNDLE_DATA / "label_corrections.json")}
    out_all["m1_stage"] = point_m1_stage()
    from uol.final import harness
    out_all["versions"], out_all["code_sha256"] = harness.versions(), harness.code_hashes()
    out_all["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    prev = models / "MODELS.json"
    if prev.exists():                                    # a later ranker added to an earlier fit: keep both records
        old = json.loads(prev.read_text(encoding="utf-8"))
        old["rankers"].update(out_all["rankers"])
        out_all = {**old, "rankers": old["rankers"], "utc_last": out_all["utc"]}
    write_json(prev, out_all)
    return out_all


# --------------------------------------------------------------------------------------------- per-case path


@contextlib.contextmanager
def _seeded_lda(model) -> Iterator[None]:
    from uol.final.adapters import topic
    state = model.lda.random_state
    model.lda.random_state = np.random.RandomState(topic.SEED)
    try:
        yield
    finally:
        model.lda.random_state = state


def load_topic_models(models: Path, info: dict) -> dict[str, tuple[Any, dict]]:
    out = {}
    for kind, t in info["topics"].items():
        p = Path(models) / t["file"]
        if sha256(p) != t["sha256"]:
            raise RuntimeError(f"topic model {p.name}: sha256 differs from info.json")
        with p.open("rb") as f:
            out[kind] = (pickle.load(f), dict(t["params"]))
    return out


def topic_row(kind: str, model, params: dict, text: str) -> np.ndarray:
    """One case's [N] topic scores: a single-case call with the inference seed reset (doc2vec's ``score`` reseeds)."""
    if kind == "lsa":
        return np.asarray(model.score([text]), dtype=np.float32)[0]
    if kind == "lda_sim":
        with _seeded_lda(model):
            return np.asarray(model.similarity([text]), dtype=np.float32)[0]
    if kind == "lda_dm":
        with _seeded_lda(model):
            return np.asarray(model.doc_model([text], params["lam"], params["mu"]), dtype=np.float32)[0]
    if kind == "doc2vec":
        return np.asarray(model.score([text]), dtype=np.float32)[0]
    raise KeyError(kind)


class NewCaseRanker:
    """A saved all-development ranker applied to one case at a time (``timing.CaseRanker`` without the fold fit)."""

    def __init__(self, sc, model, topics: dict[str, tuple[Any, dict]], m1=None):
        from uol.embedding_push import fusion_ltr as fl
        self.sc, self.h, self.v, self.model = sc, sc.h, sc.h.v, model
        self.by_id = {s.id: s for s in self.h.F.sources}
        ctx = self.h.ctx
        self.table, _ = fl.instrument_tables(ctx)
        self.log_len = np.log1p(np.asarray([len(t) for t in ctx.data.candidate_texts], dtype=np.float32))
        self.kinds = [m.kind for m in sc.extra]
        self.topics = topics
        self.m1 = m1 if sc.extension == "m1" else None
        if sc.extension == "m1" and m1 is None:
            raise ValueError(f"{sc.sid} needs the M1 memory")

    @property
    def sources(self) -> list[str]:
        return list(dict.fromkeys(list(self.v.pool_ids) + list(self.v.feature_ids)))

    def pool(self, src: dict[str, Any]) -> np.ndarray:
        from uol.final import timing
        return timing.CaseRanker.pool(self, src)

    def features(self, text: str, src: dict[str, Any], pc: np.ndarray, group: int) -> np.ndarray:
        """The frozen columns for the pool (sources in feature order, then length, the instrument prior of training
        group ``group`` and the query length)."""
        from uol.embedding_push import fusion_ltr as fl
        from uol.final import timing
        cols = []
        for sid in self.v.feature_ids:
            s = self.by_id[sid]
            if s.kind == "full":
                cols.append(timing.full_cols(fl.sanitize(np.asarray(src[sid], dtype=np.float32)[None, :])[0], pc, 0))
            else:
                cols.append(timing.pairs_cols(src[sid]["cidx"], src[sid]["score"], pc))
        shared = np.stack([self.log_len[pc], self.table[int(group), pc],
                           np.full(len(pc), np.log1p(np.float32(len(text))))], axis=1)
        return np.hstack(cols + [shared.astype(np.float32)]).astype(np.float32)

    def topic_rows(self, text: str) -> dict[str, np.ndarray]:
        return {k: topic_row(k, self.topics[k][0], self.topics[k][1], text) for k in self.kinds}

    def extras(self, topic_rows: dict[str, np.ndarray], pc: np.ndarray, m1_cols: np.ndarray | None) -> np.ndarray:
        from uol.final import timing
        blocks = [timing.full_cols(topic_rows[k], pc, 0) for k in self.kinds]
        if self.m1 is not None:
            blocks.append(np.asarray(m1_cols, dtype=np.float32))
        return np.hstack(blocks).astype(np.float32)

    def m1_columns(self, q: np.ndarray, date: str, pc: np.ndarray, key: int) -> np.ndarray:
        """M1 columns of a case from its unit Qwen3-8B vector and publication date (``timing.M1Case`` arithmetic)."""
        from uol.round3 import m1_memory as M1
        self.m1.dates[int(key)] = np.datetime64(str(date)[:10], "D")
        idx, sim = self.m1.neighbours(np.asarray(q, dtype=np.float32), int(key))
        v, c = self.m1.votes(idx, sim, int(key))
        return M1.row_columns(v, c, pc)

    def rank(self, text: str, src: dict[str, Any], group: int, *, topic_rows: dict[str, np.ndarray] | None = None,
             m1_cols: np.ndarray | None = None, extra_cols: np.ndarray | None = None) -> dict[str, Any]:
        """Pool, columns, decision and the pooled [N] score row (``fusion_ltr.write_pooled`` for one row).
        ``extra_cols`` replaces the topic and M1 columns (dry run: the fit's own batch columns)."""
        from uol.final import timing
        pc = self.pool(src)
        X = self.features(text, src, pc, group)
        if extra_cols is None:
            extra_cols = self.extras(topic_rows if topic_rows is not None else self.topic_rows(text), pc, m1_cols)
        Xf = np.hstack([X, extra_cols]).astype(np.float32)
        dec = self.model.decision_function(Xf).astype(np.float32)
        row = timing.CaseRanker.pooled_row(self, pc, dec, np.asarray(src["L_star"]))
        return {"pc": pc, "X": Xf, "decision": dec, "row": row}


def m1_case():
    from uol.final import timing
    return timing.M1Case(dict(M1_CFG))


# --------------------------------------------------------------------------------------------- dry run


def frozen_sources(sources: Sequence[str], row: int) -> dict[str, Any]:
    """The record's inputs for a development row: frozen full score rows and the frozen ColBERT pairs."""
    from uol.embedding_push import common as ep
    from uol.final import timing_sources as TS
    out: dict[str, Any] = {}
    for sid in sources:
        if sid == "colbert_bge":
            f = TS.frozen_pairs()
            m = f["qidx"] == int(row)
            out[sid] = {"cidx": f["cidx"][m], "score": f["score"][m]}
        else:
            out[sid] = np.asarray(ep.open_scores(sid)[int(row)], dtype=np.float32)
    if "L_star" not in out:
        out["L_star"] = np.asarray(ep.open_scores("L_star")[int(row)], dtype=np.float32)
    return out


def dryrun(models: Path, work: Path, out: Path, rankers: Sequence[str] = RANKERS) -> dict[str, Any]:
    """Saved models against their own batch output on the dry-run development cases (no label read):

    * the pickle gives the fit's batch decision values bit for bit;
    * the per-case pool equals the pair table's, the frozen and M1 columns equal the fit's;
    * with the fit's topic columns the per-case top-10 equals the batch top-10 (gated, every case);
    * the single-case topic call is deterministic (two calls equal; gated) and its top-10 agreement with the batch
      top-10 is reported (the batch columns came from 256-row calls whose LDA state moved between rows)."""
    from uol.embedding_push import common as ep
    from uol.final import timing
    models, out = Path(models), Path(out)
    activate(work)
    manifest = json.loads((models / "MODELS.json").read_text(encoding="utf-8"))
    pop = dev_population()
    d = ep.release()
    res: dict[str, Any] = {"rankers": {}, "fails": []}
    m1 = None
    for sid in rankers:
        info = json.loads((models / sid / "info.json").read_text(encoding="utf-8"))
        if info != manifest["rankers"][sid]:
            res["fails"].append(f"{sid}: info.json differs from MODELS.json")
        for name, h in info["files"].items():
            if sha256(models / sid / name) != h:
                res["fails"].append(f"{sid}: {name} sha256 differs")
        with (models / sid / "model.pkl").open("rb") as f:
            model = pickle.load(f)
        z = np.load(models / sid / "dry.npz")
        r: dict[str, Any] = {"pickle_decision_identical": bool(np.array_equal(
            model.decision_function(z["X"]).astype(np.float32), z["decision"]))}
        if not r["pickle_decision_identical"]:
            res["fails"].append(f"{sid}: the saved model does not give its batch decision values")
        sc = prepared(sid, pop)
        if sc.extension == "m1" and m1 is None:
            m1 = m1_case()
        rk = NewCaseRanker(sc, model, load_topic_models(models, info), m1)
        nbase = len(sc.h.v.hgb_cols)
        ntopic = 3 * len(rk.kinds)
        cases = []
        qc = None
        if rk.m1 is not None:
            from uol.embedding_push import qwen_endpoint as qe
            qc = qe.query_vectors("c")
        for row in [int(x) for x in z["rows"]]:
            sel = z["pq"] == row
            fit_pc, fit_X, fit_dec = z["pc"][sel].astype(np.int64), z["X"][sel], z["decision"][sel]
            src = frozen_sources(rk.sources, row)
            text = d.queries[row].text
            group = int(sc.h.ctx.fold[row])
            batch_row = timing.CaseRanker.pooled_row(rk, fit_pc, fit_dec, src["L_star"])
            c: dict[str, Any] = {"row": row, "query_id": d.queries[row].query_id}
            pc = rk.pool(src)
            c["pool_equal"] = bool(np.array_equal(pc, fit_pc))
            if not c["pool_equal"]:
                cases.append(c)
                continue
            X = rk.features(text, src, pc, group)
            diff = np.abs(np.nan_to_num(X, nan=1e9) - np.nan_to_num(fit_X[:, :nbase], nan=1e9))
            c["base_max_abs_diff"] = float(diff.max()) if diff.size else 0.0
            if rk.m1 is not None:
                q = np.asarray(qc[row], dtype=np.float32)
                q = q / max(float(np.linalg.norm(q)), 1e-12)
                m1c, m1info = rk.m1.columns(q, row, pc, cached=False)
                dm = np.abs(np.nan_to_num(m1c, nan=1e9) - np.nan_to_num(fit_X[:, nbase + ntopic:], nan=1e9))
                c["m1_max_abs_diff"] = float(dm.max()) if dm.size else 0.0
                c["m1_same_top_k"] = m1info.get("same_top_k")
            replay = rk.rank(text, src, group, extra_cols=fit_X[:, nbase:])
            c["replay_decision_max_abs_diff"] = float(np.max(np.abs(replay["decision"] - fit_dec)))
            c["replay_top10_equal"] = timing.top10(replay["row"]) == timing.top10(batch_row)
            t1, t2 = rk.topic_rows(text), rk.topic_rows(text)
            c["topic_single_case_deterministic"] = all(np.array_equal(t1[k], t2[k]) for k in t1)
            m1_cols = fit_X[:, nbase + ntopic:] if rk.m1 is not None else None
            fresh = rk.rank(text, src, group, topic_rows=t1, m1_cols=m1_cols)
            tdiff = np.abs(np.nan_to_num(fresh["X"][:, nbase:nbase + ntopic], nan=1e9)
                           - np.nan_to_num(fit_X[:, nbase:nbase + ntopic], nan=1e9))
            c["topic_single_vs_batch_max_abs_diff"] = {k: float(tdiff[:, 3 * i:3 * i + 3].max())
                                                      for i, k in enumerate(rk.kinds)}
            c["single_case_topic_top10_equal_batch"] = timing.top10(fresh["row"]) == timing.top10(batch_row)
            cases.append(c)
        r["cases"] = cases
        r["n"] = len(cases)
        for key in ("pool_equal", "replay_top10_equal", "topic_single_case_deterministic"):
            r[key] = float(np.mean([bool(c.get(key)) for c in cases]))
            if r[key] < 1.0:
                res["fails"].append(f"{sid}: {key} {r[key]:.3f} < 1")
        r["base_max_abs_diff"] = max((c.get("base_max_abs_diff", 0.0) for c in cases), default=None)
        if r["base_max_abs_diff"] is not None and r["base_max_abs_diff"] > 1e-5:
            res["fails"].append(f"{sid}: frozen feature columns differ by {r['base_max_abs_diff']:.3g}")
        if rk.m1 is not None:
            r["m1_max_abs_diff"] = max(c.get("m1_max_abs_diff", 0.0) for c in cases)
            if r["m1_max_abs_diff"] > 1e-5:
                res["fails"].append(f"{sid}: M1 columns differ by {r['m1_max_abs_diff']:.3g}")
        r["single_case_topic_top10_equal_batch"] = float(np.mean([bool(c.get("single_case_topic_top10_equal_batch"))
                                                                  for c in cases]))
        res["rankers"][sid] = r
        log(f"{sid}: dry run {json.dumps({k: v for k, v in r.items() if k != 'cases'})}", out / "dryrun.log")
        sc.h._base = {}
    res["pass"] = not res["fails"]
    write_json(out / "dryrun_rankers.json", res)
    return res


def lstar_dryrun(work: Path, out: Path) -> dict[str, Any]:
    """lstar_headings (P1) per case on the dry-run cases against the record's top 200 (exact: same candidates, same
    order); the record's fold-0 selection, the frozen V2 heading text."""
    import pandas as pd
    from uol.embedding_push import common as ep
    from uol.final import registry
    out = Path(out)
    activate(work)
    pop = dev_population()
    rows = dry_rows(pop)
    sc = registry.make("lstar_headings")
    sc.prepare(pop)
    f0 = next(f for f in pop.folds() if f.k == 0)
    sc.fit(f0)
    ref = pd.read_parquet(RECORD / "runs" / "lstar_headings" / "DEV" / "top200.parquet")
    d = ep.release()
    exclude, key = None, None
    from uol.final import timing
    exclude, key = timing.eval_view()
    from uol.facts_to_law import metrics
    cases = []
    for r in rows:
        S = sc.score(np.asarray([int(r)]))[0]
        res = metrics.evaluate_matrix(np.asarray(S, dtype=np.float64)[None], [[]], exclude_idx=exclude, top_k=200,
                                      tie_order=key)
        got = [int(c) for c in res[0]["topk"]]
        want = ref[ref["query_id"] == d.queries[int(r)].query_id].sort_values("rank")["candidate_idx"].astype(int).tolist()
        cases.append({"row": int(r), "top200_equal": got == want, "n_ref": len(want)})
    share = float(np.mean([c["top200_equal"] for c in cases]))
    res = {"cases": cases, "top200_equal": share, "selection": sc.params, "pass": share == 1.0}
    write_json(out / "dryrun_lstar_headings.json", res)
    log(f"lstar_headings: top200 equal {share:.3f} on {len(cases)} cases", out / "dryrun.log")
    return res


# --------------------------------------------------------------------------------------------- spread


def spread(work: Path, out: Path, rankers: Sequence[str] = RANKERS) -> dict[str, Any]:
    """One fold-0 refit per ranker the record's way (``harness.run`` on DEV fold 0, into this work store) and its
    component-macro nDCG@10 beside the record's on the same cases: run-to-run spread, reported, never gated."""
    import pandas as pd
    from uol.final import harness, measures
    out = Path(out)
    prof = activate(work)
    res: dict[str, Any] = {}
    for sid in rankers:
        t0 = time.perf_counter()
        harness.run(sid, "DEV", store=prof.store, folds=[0])
        mine = pd.read_parquet(prof.store / "runs" / sid / "DEV" / "per_case.parquet")
        rec = pd.read_parquet(RECORD / "runs" / sid / "DEV" / "per_case.parquet")
        rec0 = rec[rec["fold"] == 0]
        if sorted(mine["query_id"]) != sorted(rec0["query_id"]):
            raise RuntimeError(f"{sid}: the refit's fold-0 cases differ from the record's")
        a, b = measures.summary(mine), measures.summary(rec0)
        top_m = pd.read_parquet(prof.store / "runs" / sid / "DEV" / "top200.parquet")
        top_r = pd.read_parquet(RECORD / "runs" / sid / "DEV" / "top200.parquet")
        t10 = lambda t: {q: g.sort_values("rank")["candidate_idx"].tolist()[:10] for q, g in t.groupby("query_id")}
        tm, tr = t10(top_m), t10(top_r)
        same = float(np.mean([tm[q] == tr[q] for q in tm]))
        res[sid] = {"cases": int(len(mine)), "groups": int(mine["component_id"].nunique()),
                    "refit_ndcg_at_10": a["ndcg_at_10"], "record_ndcg_at_10": b["ndcg_at_10"],
                    "difference": a["ndcg_at_10"] - b["ndcg_at_10"], "top10_identical_share": same,
                    "seconds": round(time.perf_counter() - t0, 1)}
        log(f"{sid}: fold-0 refit nDCG@10 {a['ndcg_at_10']:.4f} vs record {b['ndcg_at_10']:.4f}", out / "dryrun.log")
    write_json(out / "spread_fold0.json", res)
    return res
