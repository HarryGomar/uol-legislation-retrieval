"""The whole run (research/programs/overnight_20261105/PLAN.md, "what landing a system means"; D210).

One command takes a case's facts and returns the final list. Nothing stored about the case is in its path: no source row, no
column, no query vector, no model answer. The stored score files and the pair table are refused at the file level, so a read
of one stops the run. Development cases are treated as new and answered by the fold-0 model, so each result can be compared
with the stored one.

    py -3.12 -m uol.final.whole_run fit --setup ranker --go       fold-0 model and what the case needs from training
    py -3.12 -m uol.final.whole_run package --setup ranker --go   the columns' order and the instrument prior of the fold
    py -3.12 -m uol.final.whole_run run --setup system --encoder card --cases 50 --tag a --go
    py -3.12 -m uol.final.whole_run compare --setup system --encoder card --tag a --go

Setups: ``ranker`` (rt_lm_tesis), ``noenc`` (rk_noenc), ``system`` (the ranker's list, the model's list, the merge, the final
reranking). Encoders: ``none`` (no encoder in the setup), ``stored`` (a check of the path, not a timing), ``card``, ``cpu``.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pickle
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PROG = ROOT / "research" / "programs" / "whole_run_20261105"
WORK = Path(os.environ.get("UOL_WHOLE_WORK", ROOT / ".cache" / "whole_run"))
SIDS = {"ranker": "rt_lm_tesis", "noenc": "rk_noenc", "system": "rt_lm_tesis"}
FOLD = 0
STORE = ROOT / "experiments" / "runs" / "footing_20261101" / "dev" / "store"
MODEL_THREADS = 4
REFUSED = ("embedding_push_20261004/scores/", "/d58/x_control", "/d58/cache/top200_", "/footing_20261101/dev/store/runs/", "/agents/alone", "/agents_rk/")
KIND = "gpt61sol"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ what is fitted once


def fit(setup: str) -> int:
    """The fold-0 model, as ``ranker_tests.fit_group`` fits it with its setting chosen; checked against the stored lists."""
    from uol.embedding_push import fusion_ltr as fl
    from uol.final import lean_rerun as LR, populations, ranker_tests as RT, registry
    from uol.gap_tests import ltr_listwise as LW
    LR.D.activate(STORE)
    LR._overlay_stat_glob()
    sid = SIDS[setup]
    pop = populations.load("DEV", None)
    sc = registry.make(sid)
    sc.prepare(pop)
    fold = next(f for f in pop.folds() if f.k == FOLD)
    g, _ = sc._group(fold)
    extra, info = sc._extra_columns(fold, g)
    receipt = json.loads((STORE / "receipts" / f"{sid}__DEV.json").read_text(encoding="utf-8"))
    stored_fit = next(f for f in receipt["folds"] if f["k"] == FOLD)["fit"]
    cfg = stored_fit["config"]
    RT.LM_THREADS = MODEL_THREADS
    seed = fl.SEED + 1000 * g.idx
    rng = np.random.default_rng(seed + 999)
    tr = RT.kept_rows(sc.h.ctx, fl.train_rows_of(sc.h.ctx, g), None)
    idx, y, counts = LW.training_lists(sc.h.ctx, sc.h.F, sc.h.v, tr, rng, n_neg=cfg.get("n_neg", LW.N_NEG))
    t0 = time.perf_counter()
    model = RT._fit("lgbm", cfg, RT._matrix(sc.h.F, sc.h.v, idx, extra), y, counts, seed)
    seconds = round(time.perf_counter() - t0, 1)
    out_idx = fl.row_pairs(sc.h.v, g.rows, False, sc.h.F)
    same = reproduces(sc, g, out_idx, RT._predict(model, sc.h.F, sc.h.v, out_idx, extra), sid)
    folder = WORK / "models" / sid
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "model.pkl").open("wb") as f:
        pickle.dump(model, f)
    meta = {"sid": sid, "fold": FOLD, "role": g.role, "setting": cfg, "m1": info.get("m1") or stored_fit.get("m1"), "reproduces_stored": same, "fit_seconds": seconds,
            "train_cases": int(len(tr)), "sha256": sha256(folder / "model.pkl")}
    (folder / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print(json.dumps(meta, indent=1, default=str), flush=True)
    return 0


def reproduces(sc, g, out_idx, score, sid: str) -> dict:
    """Top 10 and top 200 of the fold's cases, written as the harness writes them, against the stored run."""
    import pandas as pd
    from uol.embedding_push import common as ep, fusion_ltr as fl
    from uol.final.adapters.ltr import _RowSink
    d = ep.release()
    rows = np.asarray(g.rows)
    sink = _RowSink(rows, sc.N)
    fl.write_pooled(sc.h.ctx, sc.h.F, sink, rows, out_idx, score, sc.lstar)
    t = pd.read_parquet(STORE / "runs" / sid / "DEV" / "top200.parquet")
    by = {q: x.sort_values("rank")["candidate_idx"].astype(int).tolist() for q, x in t.groupby("query_id")}
    generic = {int(i) for i in d.generic_idx}
    n = same10 = 0
    over = []
    for i, r in enumerate(rows):
        q = d.queries[int(r)].query_id
        if q not in by:
            continue
        s = by[q]
        got = [int(c) for c in np.argsort(-sink.out[i], kind="stable")[:len(s) + len(generic)] if int(c) not in generic][:len(s)]
        n += 1
        same10 += got[:10] == s[:10]
        over.append(len(set(got) & set(s)) / len(s))
    return {"cases": n, "top10_same": same10, "top200_overlap": float(np.mean(over))}


def package(setup: str) -> int:
    """What a case needs from training, written once with the pair table present; a run then reads only these."""
    from uol.embedding_push import fusion_ltr as fl
    from uol.final import new_cases_fit as NF, registry, timing_ranker as TR
    TR.activate()
    sid = SIDS[setup]
    sc = registry.make(sid)
    sc.prepare(NF.dev_population())
    v, F = sc.h.v, sc.h.F
    table, _ = fl.instrument_tables(sc.h.ctx)
    fold0 = np.asarray([int(sc.h.ctx.fold[r]) for r in next(f for f in NF.dev_population().folds() if f.k == FOLD).score_rows[:50]])
    if len(set(fold0.tolist())) != 1:
        raise RuntimeError("the fold's cases do not share a training group")
    group = int(fold0[0])
    # columns: three of each full source, in feature order, then the three shared; the model must take this order
    kinds = {s.id: s.kind for s in F.sources}
    if any(kinds[s] != "full" for s in v.feature_ids):
        raise RuntimeError("a pair source in the features")
    meta = json.loads((WORK / "models" / sid / "meta.json").read_text(encoding="utf-8"))
    n_model = len(v.hgb_cols) + (3 if meta["m1"] else 0)
    out = WORK / "models" / sid
    np.save(out / "instrument.npy", np.asarray(table[group], dtype=np.float32))
    w = None
    if setup == "system":
        w = float(json.loads((ROOT / ".cache" / "footing" / "dev" / "llm" / f"{KIND}_rerank_rt_lm_tesis" / "meta.json").read_text(encoding="utf-8"))["w_all"])
    spec = {"pool_ids": list(v.pool_ids), "feature_ids": list(v.feature_ids), "group": group, "columns": n_model, "w": w}
    prev = out / "spec.json"
    if prev.exists() and w is None:
        spec["w"] = json.loads(prev.read_text(encoding="utf-8")).get("w")
    prev.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    print(json.dumps(spec, indent=1), flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ parts of one case


def refuse_stored() -> None:
    """Any later open of a stored score file, the pair table or a stored list stops the run."""
    def hook(event, args):
        if event == "open" and args and isinstance(args[0], (str, os.PathLike)):
            p = os.fspath(args[0]).replace("\\", "/").lower()
            # a method's manifest holds its chosen settings, not its scores: the plain release reads it from the stored folder
            if any(x in p for x in REFUSED) and not p.endswith("/manifest.json"):
                raise PermissionError(f"the whole run opened a stored file: {args[0]}")
    sys.addaudithook(hook)


class Ranker:
    """The saved model, its pool and its columns for one case (``new_cases_fit.NewCaseRanker`` without the pair table)."""

    def __init__(self, sid: str, candidate_texts, folder: Path | None = None):
        folder = folder or WORK / "models" / sid
        self.meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        if sha256(folder / "model.pkl") != self.meta["sha256"]:
            raise RuntimeError("the saved model does not match its hash")
        with (folder / "model.pkl").open("rb") as f:
            self.model = pickle.load(f)
        self.model.set_params(n_jobs=MODEL_THREADS)
        self.spec = json.loads((folder / "spec.json").read_text(encoding="utf-8"))
        self.instrument = np.load(folder / "instrument.npy")
        self.log_len = np.log1p(np.asarray([len(t) for t in candidate_texts], dtype=np.float32))
        self.sources = list(dict.fromkeys(self.spec["pool_ids"] + self.spec["feature_ids"]))

    def pool(self, src: dict) -> np.ndarray:
        from uol.embedding_push import fusion_ltr as fl
        tops = [fl.topk_block(fl.sanitize(np.asarray(src[s], dtype=np.float32)[None, :]))[0] for s in self.spec["pool_ids"]]
        return np.unique(np.concatenate(tops)).astype(np.int64)

    def features(self, text: str, src: dict, pc: np.ndarray) -> np.ndarray:
        from uol.embedding_push import fusion_ltr as fl
        from uol.final import timing
        cols = [timing.full_cols(fl.sanitize(np.asarray(src[s], dtype=np.float32)[None, :])[0], pc, 0) for s in self.spec["feature_ids"]]
        shared = np.stack([self.log_len[pc], self.instrument[pc], np.full(len(pc), np.log1p(np.float32(len(text))))], axis=1)
        return np.hstack(cols + [shared.astype(np.float32)]).astype(np.float32)


class WordVotes:
    """The older-tesis votes of the setup without the encoder: neighbours by word similarity (``m1_memory.build_neighbours``,
    key ``lex``), one case at a time; the vote arithmetic is ``timing.M1Case``'s."""

    def __init__(self, cfg: dict):
        from uol.embedding_push import common as ep
        from uol.facts_to_law.lexical import LexicalIndex
        from uol.round2.b_resolve import compose
        from uol.round3 import common as R3, m1_memory as M1
        self.cfg, self.M1 = cfg, M1
        self.mem = M1.memory()
        want = set(self.mem["record_id"].tolist())
        hist = {}
        with (ROOT / "experiments" / "runs" / "round2_20261004" / "B" / "historical_tesis.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if str(r["record_id"]) in want:
                    hist[str(r["record_id"])] = compose(r)
        self.sel = np.flatnonzero(np.asarray([rid in hist for rid in self.mem["record_id"]]))
        self.index = LexicalIndex([hist[self.mem["record_id"][i]] for i in self.sel], kind="bm25")
        self.m_dates_sel = self.mem["date"][self.sel]
        self.P, self.Pb = M1.cited_matrices(self.mem["cited"], len(ep.release().candidate_ids))
        pd = R3.publication_dates()
        self.dates = {int(r): np.datetime64(pd[int(r)], "D") for r in ep.dev_rows()}

    def neighbours(self, text: str, row: int) -> tuple[np.ndarray, np.ndarray]:
        sims = np.asarray(self.index.score([text]), dtype=np.float32)
        self.M1._date_mask(sims, np.asarray([self.dates[int(row)]], dtype="datetime64[D]"), self.m_dates_sel)
        li, ls = self.M1._topk(sims, self.M1.KMAX)
        return self.sel[li[0]], ls[0]

    def columns(self, text: str, row: int, pc: np.ndarray) -> np.ndarray:
        from uol.final import timing as T
        idx, sim = self.neighbours(text, row)
        v, c = T.M1Case.votes(self, idx, sim, row)
        return self.M1.row_columns(v, c, pc)


class Encoder:
    """Qwen3-Embedding-8B in its own process (the machine's card or its processor), asked one case at a time."""

    def __init__(self, kind: str):
        self.kind = kind
        py = os.environ["UOL_ENCODER_PYTHON"]
        env = {**os.environ, "HF_HUB_OFFLINE": "1"}
        for k in ("PYTHONPATH", "PYTHONHOME", "PYTHONHASHSEED", "OMP_NUM_THREADS"):
            env.pop(k, None)
        env["PATH"] = os.pathsep.join(x for x in env["PATH"].split(os.pathsep) if "pack" + os.sep + "py" not in x)
        script = Path(__file__).parent / "gpu_jobs" / "whole_encoder_server.py"
        self.p = subprocess.Popen([py, str(script), "--device", kind], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8", env=env, bufsize=1)
        ready = json.loads(self._line())
        self.info = ready

    def _line(self) -> str:
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("the encoder process stopped")
            if line.startswith("{"):
                return line

    def __call__(self, text: str) -> tuple[np.ndarray, dict]:
        from uol.embedding_push import qwen_endpoint as qe
        self.p.stdin.write(json.dumps({"text": qe.query_input("c", text)}, ensure_ascii=False) + "\n")
        self.p.stdin.flush()
        r = json.loads(self._line())
        return np.frombuffer(base64.b64decode(r["vector"]), dtype=np.float32).copy(), {k: r[k] for k in ("seconds", "tokens")}

    def memory(self) -> dict:
        self.p.stdin.write(json.dumps({"memory": True}) + "\n")
        self.p.stdin.flush()
        return json.loads(self._line())["memory"]

    def close(self) -> None:
        try:
            self.p.stdin.close()
            self.p.wait(timeout=60)
        except Exception:
            self.p.kill()


class ModelSteps:
    """The model's list of 20 from the facts, and the final reranking of the first 30: a case per call, answers never cached
    from another run (``UOL_LUNA_CACHE`` is this run's own)."""

    def __init__(self, generic: set[int], w: float, tag: str):
        from uol.final import external as X, footing as FT, llm_agents as G, prompt_check as P
        self.lock = threading.Lock()
        self.X, self.FT, self.G, self.P = X, FT, G, P
        self.luna = G.pin_model()
        # codex.bind() points the cache at the profile's shared one, seeded with earlier answers: this run gets its own,
        # new and empty, and keeps it through every later bind
        from uol.final import codex as CX
        own = WORK / "luna" / f"cache_{tag}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.sqlite"
        if own.exists():
            raise RuntimeError(f"{own} exists")
        own.parent.mkdir(parents=True, exist_ok=True)
        bind0 = getattr(CX, "_bind_shared", CX.bind)
        CX._bind_shared = bind0

        def bind_own():
            luna = bind0()
            luna.CACHE = own
            return luna
        CX.bind = bind_own
        self.luna.CACHE = own
        self.cache = own
        self.generic, self.w, self.tag = generic, w, tag
        self.cfg = {**X.SWEEP["gpt6sol"], **FT.MODELS[KIND]}
        self.schema = P.item_schema(20)

    def model_list(self, row: int) -> tuple[list[int], dict]:
        prompt, ids = self.P.build([row], 20)
        # luna.call binds its default model when it is defined: the pinned model is passed by name
        r = self.luna.call(prompt, self.schema, effort="medium", tag=f"whole_list_{self.tag}", model=self.luna.MODEL)
        if r.get("model", self.luna.MODEL) != self.luna.MODEL:
            raise RuntimeError(f"the list came from {r.get('model')}, not {self.luna.MODEL}")
        got = self.luna.checked(r, prompt, self.schema, ids, effort="medium", tag=f"whole_list_{self.tag}")
        return self.G.resolved(got[ids[0]]["articulos"], self.generic), {self.luna.key_of(prompt, self.schema, self.luna.MODEL, "medium", 0): r.get("usage") or {}}

    def rerank(self, row: int, merged: list[int]) -> tuple[list[int], dict]:
        X, FT, G = self.X, self.FT, self.G
        shown = {row: merged[:X.TOP]}
        stats: dict[str, int] = {}
        usage: dict[str, dict] = {}
        ask = G.ask_some(lambda items: X.ask_claude(items, X.codex_api(self.cfg, f"whole_rerank_{self.tag}", usage), stats, self.cfg))
        o = FT.rerank_depth([row], shown, ask, X.TOP, passes=("fwd",))
        return G.fuse_order(shown[row], o["orders"], ["fwd"], row, self.w) + merged[X.TOP:], usage

    def usd(self, usage: dict) -> float:
        return self.X.codex_usage(usage, self.cfg)[1]


def need_test_release() -> dict:
    """Nothing that loads a test row runs before ``TEST_RELEASE.json`` holds the author's word: loading and warm-up included."""
    path = PROG / "TEST_RELEASE.json"
    if not path.exists():
        raise PermissionError(f"{path} does not exist: the test set is closed until the author releases it")
    release = json.loads(path.read_text(encoding="utf-8"))
    if not release.get("word") or not release.get("scope"):
        raise PermissionError(f"{path} must hold the author's word and the scope (TEST_RELEASE_SCOPE.json)")
    return release


class Live:
    """One setup, loaded once, answering one case at a time from its facts."""

    def __init__(self, setup: str, encoder: str, tag: str, models: Path | None = None, new_cases: dict | None = None):
        """``models``: another saved model's folder. ``new_cases``: the extended release, the all-development first stage and
        these rows (``whole_test``); ``None`` is a development case answered by the fold-0 first stage."""
        from uol.embedding_push import common as ep
        if new_cases is not None:
            need_test_release()
        from uol.final import new_cases_fit as NF, timing as T, timing_ranker as TR, timing_sources as TS
        self.setup, self.sid, self.T, self.encoder_kind = setup, SIDS[setup], T, encoder
        self.load: dict[str, float] = {}

        def timed(name, fn):
            t0 = time.perf_counter()
            v = fn()
            self.load[name] = round(time.perf_counter() - t0, 2)
            print(f"{time.strftime('%H:%M:%S')} load {name}: {self.load[name]} s", flush=True)
            return v

        if new_cases is None:
            os.environ["UOL_TIMING_DATA"] = str(TR.STAGE)
        else:
            os.environ.pop("UOL_TIMING_DATA", None)      # the extended release reads the original release-row files
        timed("profile", (lambda: NF.activate(TR.WORK)) if new_cases is None else new_cases["activate"])
        self.d = ep.release()
        self.generic = {int(i) for i in self.d.generic_idx}
        self.rk = timed("ranker", lambda: Ranker(self.sid, self.d.candidate_texts, models))
        m1 = dict(self.rk.meta["m1"])
        self.dense = m1.get("key") == "dense"
        self.votes = timed("votes", lambda: T.M1Case(m1) if self.dense else WordVotes(m1))
        if new_cases is not None:
            self.votes.dates.update(new_cases["dates"])
        self.fs = TS.FirstStage("dev_fold0" if new_cases is None else "lockbox", sources=self.rk.sources, colbert="off")
        if new_cases is not None:
            self.fs.eval_only = np.asarray(new_cases["rows"], dtype=np.int64)
            self.fs.append = dict(new_cases["append"])
        with T.first_stage_labels():
            timed("first_stage", self.fs.load)
        # nothing stored about the query stays reachable: a source reading a stored query vector would read NaN.
        # The past cases of the citation memory keep theirs: the votes compare the query with them.
        self.q_stored = {k: np.array(v[:0]) for k, v in self.fs.stored.items()}
        self.stored_qc = np.array(self.fs.stored["Qc"], copy=True) if "Qc" in self.fs.stored else None
        for k in self.fs.stored:
            if k != "Qc" or encoder != "stored":
                self.fs.stored[k][:] = np.nan
        if "Qc" in self.fs.stored and "qe_fit" in self.fs.S:
            fit = self.fs.S["qe_fit"]
            past = np.union1d(fit.rows, fit.mem_rows)
            if new_cases is not None and np.isin(new_cases["rows"], past).any():
                raise ValueError("a new case is in the citation memory")
            self.fs.stored["Qc"][past] = self.stored_qc[past]
        self.fs.reset_vectors()
        self.encoder = timed("encoder", lambda: Encoder(encoder)) if encoder in ("card", "cpu") else None
        self.model = timed("model_steps", lambda: ModelSteps(self.generic, float(self.rk.spec["w"]), tag)) if setup == "system" else None

    def ranker_list(self, row: int, clock) -> tuple[list[int], dict]:
        text = self.d.queries[row].text
        info: dict = {}
        q = None
        self.fs.reset_vectors([row])
        self.fs.S.pop("qe_sims_key", None)
        if self.dense:
            if self.encoder is not None:
                with clock.stage("encode"):
                    q, enc = self.encoder(text)
                info["encoder"] = enc
            else:
                q = np.array(self.stored_qc[row], dtype=np.float32)
            q = q / max(float(np.linalg.norm(q)), 1e-12)
            s = self.stored_qc[row]
            info["cos_to_stored"] = float(np.dot(q, s) / np.linalg.norm(s))
            self.fs.inject(row, {"Qc": q})
        with self.T.first_stage_labels():
            src = self.fs.case(row, clock=clock)
        self.fs.reset_vectors([row])                         # the next case's memory holds the stored vector, not this fresh one
        with clock.stage("pool"):
            pc = self.rk.pool(src)
        with clock.stage("features"):
            X = self.rk.features(text, src, pc)
        with clock.stage("votes"):
            cols = self.votes.columns(q, row, pc)[0] if self.dense else self.votes.columns(text, row, pc)
        with clock.stage("rank"):
            dec = self.rk.model.predict(np.hstack([X, np.asarray(cols, dtype=np.float32)]).astype(np.float32))
            order = pc[np.argsort(-dec, kind="stable")]
        info["pool"] = int(len(pc))
        return [int(c) for c in order if int(c) not in self.generic][:200], info

    def answer(self, row: int) -> dict:
        clock = self.T.Clock()
        t0, c0 = time.perf_counter(), time.process_time()
        ranked, info = self.ranker_list(row, clock)
        rec = {"row": row, "query_id": self.d.queries[row].query_id, "info": info, "ranker_list": ranked}
        rec["cpu_seconds"] = time.process_time() - c0
        final = ranked
        if self.model is not None:
            with clock.stage("model_list"):
                lst, usage = self.model.model_list(row)
            with clock.stage("merge"):
                merged = self.model.G.rrf([lst, ranked])
            with clock.stage("rerank"):
                final, u2 = self.model.rerank(row, merged)
            rec.update({"model_list": lst, "merged": merged, "usd": self.model.usd({**usage, **u2})})
        rec["seconds"] = time.perf_counter() - t0
        rec["wall"] = {k: round(v, 5) for k, v in clock.wall.items()}
        rec["model_wait"] = sum(clock.wall.get(k, 0.0) for k in ("model_list", "rerank"))
        rec["encode_wait"] = clock.wall.get("encode", 0.0)
        rec["final"] = final
        return rec


# ------------------------------------------------------------------------------------------------ the run


def spread(v: list[float]) -> dict:
    w = sorted(v)
    return {"n": len(v), "median": statistics.median(v), "p95": w[int(np.ceil(0.95 * len(w))) - 1], "max": w[-1]}


class Memory:
    """RAM of this process and its children (the encoder), summed at each sample, every half second; the card's memory in
    use from nvidia-smi every two seconds, and its level before the run. In GiB."""

    def __init__(self):
        import psutil
        self.ps, self.me = psutil, psutil.Process()
        self.card0 = self.card_used()
        self.ram_peak, self.card_peak, self.n = 0.0, self.card0 or 0.0, 0
        self.stop = threading.Event()
        self.t = threading.Thread(target=self.loop, daemon=True)
        self.t.start()

    @staticmethod
    def card_used() -> float | None:
        try:
            s = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
            return round(float(s.split()[0]) / 1024, 2)
        except Exception:
            return None

    def ram(self) -> float:
        tot = 0
        for q in [self.me, *self.me.children(recursive=True)]:
            try:
                tot += q.memory_info().rss
            except Exception:
                pass
        return tot / 2**30

    def loop(self) -> None:
        k = 0
        while not self.stop.wait(0.5):
            self.ram_peak = max(self.ram_peak, self.ram())
            self.n += 1
            k += 1
            if k % 4 == 0:
                c = self.card_used()
                if c is not None:
                    self.card_peak = max(self.card_peak, c)

    def result(self, encoder) -> dict:
        self.stop.set()
        self.t.join(timeout=10)
        mi = self.me.memory_info()
        out = {"unit": "GiB", "samples": self.n, "ram_together_peak_sampled": round(self.ram_peak, 2),
               "ranker_ram_peak": round(getattr(mi, "peak_wset", mi.rss) / 2**30, 2), "ranker_ram_now": round(mi.rss / 2**30, 2),
               "card_before_run": self.card0, "card_used_peak_sampled": round(self.card_peak, 2)}
        if encoder is not None:
            try:
                e = encoder.memory()
            except Exception as exc:          # a failed question costs the figure, not the run
                e = {"error": repr(exc)}
            out.update({f"encoder_{k}": v for k, v in e.items()})
            if "ram_peak_gb" in e:
                out["ram_both_peaks_summed"] = round(out["ranker_ram_peak"] + e["ram_peak_gb"], 2)
        return out


def folder_of(setup: str, encoder: str, tag: str | None) -> Path:
    return WORK / "runs" / f"{setup}_{encoder}{'_' + tag if tag else ''}"


def run(setup: str, encoder: str, cases: int, tag: str | None, sample_from: list[int] | None = None) -> int:
    from uol.final import timing as T, timing_ranker as TR
    refuse_stored()
    t_all = time.perf_counter()
    out = folder_of(setup, encoder, tag)
    out.mkdir(parents=True, exist_ok=True)
    mem = Memory() if os.environ.get("UOL_WHOLE_MEMORY") == "1" else None     # off for the timed runs
    live = Live(setup, encoder, tag or "a")
    if setup == "system":                 # the cases with a stored list from the model, first in fold order
        rows_all = [int(r) for r in json.loads((WORK / "models" / "system_rows.json").read_text(encoding="utf-8"))]
        rows, warm = rows_all[:cases], []
    else:
        rows, warm = TR.case_rows(cases, 3)
    recs = []
    with (out / "cases.jsonl").open("w", encoding="utf-8") as f:
        for r in warm:
            live.answer(r)
        for r in rows:
            rec = live.answer(r)
            f.write(json.dumps(rec, default=float) + "\n")
            f.flush()
            recs.append(rec)
            print(json.dumps({"row": r, "seconds": round(rec["seconds"], 3), "wall": {k: round(v, 2) for k, v in rec["wall"].items() if v >= .05}, "usd": rec.get("usd")}), flush=True)
    memory = mem.result(live.encoder) if mem else None
    if live.encoder is not None:
        live.encoder.close()
    machine = {**T.machine(), "processor": os.environ.get("UOL_WHOLE_PROCESSOR") or os.environ.get("PROCESSOR_IDENTIFIER"), "encoder": live.encoder.info if live.encoder else None}
    outside = [x["seconds"] - x["model_wait"] - x["encode_wait"] for x in recs]
    res = {"setup": setup, "sid": live.sid, "encoder": encoder, "cases": len(recs), "load_seconds": live.load, "machine": machine, "seconds": spread([x["seconds"] for x in recs]),
           "seconds_apart_from_model_and_encoder": spread(outside), "encode": spread([x["encode_wait"] for x in recs]) if live.encoder else None,
           "stages_median": {k: statistics.median(x["wall"].get(k, 0.0) for x in recs) for k in sorted({k for x in recs for k in x["wall"]})},
           "model_wait": spread([x["model_wait"] for x in recs]) if setup == "system" else None, "usd_per_case": float(np.mean([x["usd"] for x in recs])) if setup == "system" else None,
           "whole_seconds": round(time.perf_counter() - t_all, 1), **({"memory": memory} if memory else {})}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res, indent=1, default=float), flush=True)
    return 0


def compare(setup: str, encoder: str, tags: list[str | None]) -> int:
    """nDCG@10, live minus stored, per case; for repeated answers their spread on the same cases (PLAN's threshold)."""
    import pandas as pd
    from uol.final import rank_rerun as R
    G = R._agents()
    R.D.activate(R.STORE)
    from uol.embedding_push import common as ep
    d = ep.release()
    sid = {"ranker": "rt_lm_tesis", "noenc": "rk_noenc", "system": R.SYSTEM}[setup]
    stored = A_per_case(R, sid)
    out: dict = {"setup": setup, "encoder": encoder, "stored": sid}
    per = {}
    for tag in tags:
        recs = [json.loads(x) for x in (folder_of(setup, encoder, tag) / "cases.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        rows = [r["row"] for r in recs]
        if any(d.queries[r["row"]].query_id != r["query_id"] for r in recs):
            raise RuntimeError("row numbers differ between the run and the store: join by query id")
        fr = G.score_ranked(rows, {r["row"]: r["final"] for r in recs})
        joined = fr.merge(stored[["query_id", "ndcg_at_10"]], on="query_id", suffixes=("", "_stored"))
        diff = (joined["ndcg_at_10"] - joined["ndcg_at_10_stored"]).to_numpy()
        per[tag or ""] = fr.set_index("query_id")["ndcg_at_10"]
        out[tag or "run"] = {"cases": len(joined), "live": float(joined["ndcg_at_10"].mean()), "stored": float(joined["ndcg_at_10_stored"].mean()), "mean_paired_difference": float(diff.mean()),
                             "cases_same_score": int((np.abs(diff) < 1e-9).sum()), "abs_difference_p95": float(np.percentile(np.abs(diff), 95))}
    if len(per) == 2:
        a, b = list(per.values())
        j = pd.concat([a, b], axis=1, join="inner")
        rep = (j.iloc[:, 0] - j.iloc[:, 1]).to_numpy()
        rng = np.random.default_rng(0)
        boots = [rep[rng.integers(0, len(rep), len(rep))].mean() for _ in range(2000)]
        out["repeat"] = {"cases": len(j), "mean_difference": float(rep.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "abs_difference_mean": float(np.abs(rep).mean())}
    PROG.mkdir(parents=True, exist_ok=True)
    (PROG / f"compare_{setup}_{encoder}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1), flush=True)
    return 0


def A_per_case(R, sid: str):
    from uol.final import audit_d181_common as A
    return A.per_case(R.STORE, sid)


def system_rows() -> int:
    """Fold 0's cases among the 1,217 of the model's stored list, in the fold's order."""
    from uol.final import new_cases_fit as NF, rank_rerun as R, timing_ranker as TR
    G = R._agents()
    R.D.activate(R.STORE)
    keep = set(int(r) for r in G.all_rows())
    TR_rows = None
    from uol.final import populations
    pop = populations.load("DEV", None)
    f = next(x for x in pop.folds() if x.k == FOLD)
    TR_rows = [int(r) for r in f.score_rows if int(r) in keep]
    (WORK / "models").mkdir(parents=True, exist_ok=True)
    (WORK / "models" / "system_rows.json").write_text(json.dumps(TR_rows), encoding="utf-8")
    print(len(TR_rows), TR_rows[:8], flush=True)
    return 0


OPENED: dict[str, int] = {}


def trace_files() -> None:
    def hook(event, args):
        if event == "open" and args and isinstance(args[0], (str, os.PathLike)):
            p = os.path.abspath(os.fspath(args[0]))
            if p not in OPENED:
                OPENED[p] = os.path.getsize(p) if os.path.isfile(p) else 0
    sys.addaudithook(hook)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fit", "package", "system_rows", "run", "compare"])
    ap.add_argument("--setup", choices=list(SIDS), default="ranker")
    ap.add_argument("--encoder", choices=["none", "stored", "card", "cpu"], default="stored")
    ap.add_argument("--cases", type=int, default=5)
    ap.add_argument("--tag")
    ap.add_argument("--tags")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--trace", action="store_true")
    a = ap.parse_args(argv)
    if not a.go:
        print(f"[dry] {a.step} {a.setup}")
        return 0
    if a.trace:
        trace_files()
    if a.step == "fit":
        code = fit(a.setup)
    elif a.step == "package":
        code = package(a.setup)
    elif a.step == "system_rows":
        code = system_rows()
    elif a.step == "compare":
        code = compare(a.setup, a.encoder, (a.tags.split(",") if a.tags else [a.tag]))
    else:
        code = run(a.setup, a.encoder, a.cases, a.tag)
    if a.trace:
        (WORK / f"opened_{a.setup}_{a.step}.json").write_text(json.dumps({**OPENED, "__modules__": 0}, indent=0), encoding="utf-8")
        (WORK / f"modules_{a.setup}_{a.step}.json").write_text(json.dumps(sorted({m.split(".")[0] for m in sys.modules})), encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
