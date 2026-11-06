"""EP-gamma GPU part: learned case similarity (E5-small full fine-tune, bge-m3 LoRA).

A query encoder is trained so that fact summaries (masked Hechos) citing the same SPECIFIC provisions
are close: positives are training queries of DIFFERENT components sharing at least one specific
article, weighted by the shared article's IDF; the loss is an in-batch multi-positive InfoNCE
(SupCon) with temperature .05; same-component items are masked out; batches are built around
positive pairs. The learned vectors then serve as a memory key (``knn_casesim_*``) or as the basis of
dense article prototypes (``proto_casesim_*``) through ``memory_proto``.

Label discipline. For every scoring group the encoder is fitted on that group's training rows only.
Selection (epoch count, k, key mix, weighting) uses an inner holdout: model A is trained on the
training rows minus one training dev fold h and judged on h with a memory of the rest; model B is then
trained on all training rows for the selected number of epochs and scores the group. The lockbox group
trains on dev rows only. Run with the GPU virtual environment and ``PYTHONPATH=src``;
every GPU job runs inside ``common.gpu_lock``.

    python -B -m uol.embedding_push.case_sim pilot --kind e5 --role dev_fold0   # fold-0 pilot (orchestrator, CPU)
    python -B -m uol.embedding_push.case_sim run --kind e5                       # all six groups (orchestrator, CPU)
    python -B -m uol.embedding_push.case_sim train ...                           # one GPU job, launched by the orchestrator
    python -B -m uol.embedding_push.case_sim adapt                               # CPU adapters on frozen vectors
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uol.embedding_push import common, memory_proto as mp

OWNER = "EP-gamma"
WORK = mp.WORK / "casesim"
TAU = 0.05

MODELS = {
    # name: (model dir getter, prefix, pooling)
    "e5": {"dir": lambda: _e5_dir(), "prefix": "query: ", "pool": "mean", "dim": 384, "lora": False},
    "bge": {"dir": lambda: common.BGE_DIR, "prefix": "", "pool": "cls", "dim": 1024, "lora": True},
}


def _e5_dir() -> Path:
    from uol.facts_to_law import dense

    return Path(dense.MODEL_DIR)       # common.E5_DIR holds the weights only; this directory also has config and tokenizer


# ---------------------------------------------------------------------------------------------
# training set: specific-article positives across components
# ---------------------------------------------------------------------------------------------

@dataclass
class TrainSet:
    rows: np.ndarray                         # training rows that have a partner in another component
    spec: dict[int, tuple[int, ...]]         # row -> specific articles
    comp: dict[int, str]
    idf: dict[int, float]
    art_rows: dict[int, np.ndarray]          # article -> training rows citing it (specific)

    def partner_articles(self, row: int) -> list[int]:
        return [a for a in self.spec[row] if len(set(self.comp[r] for r in self.art_rows[a])) >= 2]


def build_trainset(train_rows: Sequence[int]) -> TrainSet:
    data = common.release()
    comp = mp.components()
    spec = {int(r): tuple(data.specific_idx[int(r)]) for r in train_rows if data.specific_idx[int(r)]}
    art_rows: dict[int, list[int]] = {}
    for r, arts in spec.items():
        for a in arts:
            art_rows.setdefault(a, []).append(r)
    comps_of = {a: {str(comp[r]) for r in rs} for a, rs in art_rows.items()}
    n_comp = len({str(comp[r]) for r in spec})
    idf = {a: math.log(n_comp / len(cs)) for a, cs in comps_of.items() if len(cs) >= 2}
    keep = [r for r, arts in spec.items() if any(a in idf for a in arts)]
    return TrainSet(np.asarray(sorted(keep), dtype=np.int64), spec, {r: str(comp[r]) for r in spec}, idf,
                    {a: np.asarray(rs, dtype=np.int64) for a, rs in art_rows.items() if a in idf})


def sample_partner(ts: TrainSet, row: int, rng: np.random.Generator) -> int | None:
    arts = [a for a in ts.spec[row] if a in ts.idf]
    if not arts:
        return None
    weights = np.asarray([ts.idf[a] * len(ts.art_rows[a]) ** 0.5 for a in arts])
    a = arts[int(rng.choice(len(arts), p=weights / weights.sum()))]
    pool = ts.art_rows[a]
    for _ in range(12):
        r = int(pool[rng.integers(len(pool))])
        if ts.comp[r] != ts.comp[row]:
            return r
    return None


def epoch_batches(ts: TrainSet, batch: int, rng: np.random.Generator) -> list[list[int]]:
    """One epoch: every row with a partner seeds one pair; batches hold about ``batch`` items."""
    seeds = rng.permutation(ts.rows).tolist()
    out, cur = [], []
    for q in seeds:
        if q in cur:
            continue
        p = sample_partner(ts, q, rng)
        cur.append(q)
        if p is not None and p not in cur:
            cur.append(p)
        if len(cur) >= batch:
            out.append(cur)
            cur = []
    if len(cur) >= batch // 2:
        out.append(cur)
    return out


def batch_weights(ts: TrainSet, items: Sequence[int]):
    """Pair weights W[i, j] = sum of the IDFs of the specific articles shared by items i and j (0 within a component)."""
    import torch

    n = len(items)
    w = np.zeros((n, n), dtype=np.float32)
    same = np.zeros((n, n), dtype=bool)
    sets = [{a for a in ts.spec[r] if a in ts.idf} for r in items]
    for i in range(n):
        for j in range(n):
            if i == j or ts.comp[items[i]] == ts.comp[items[j]]:
                same[i, j] = True
                continue
            shared = sets[i] & sets[j]
            if shared:
                w[i, j] = sum(ts.idf[a] for a in shared)
    return torch.from_numpy(w), torch.from_numpy(same)


def supcon_loss(z, weights, masked, tau: float = TAU):
    """In-batch multi-positive InfoNCE: -sum_j w_ij log softmax_j(s_ij / tau) / sum_j w_ij over anchors with positives."""
    import torch

    logits = (z @ z.T) / tau
    logits = logits.masked_fill(masked, -1e9)
    logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    wsum = weights.sum(dim=1)
    valid = wsum > 0
    if not bool(valid.any()):
        return logits.sum() * 0.0
    per = -(weights * logp).sum(dim=1) / wsum.clamp_min(1e-9)
    return per[valid].mean()


# ---------------------------------------------------------------------------------------------
# encoder
# ---------------------------------------------------------------------------------------------

class CaseEncoder:
    def __init__(self, kind: str, *, device: str = "cuda", lora_rank: int = 16, freeze_embeddings: bool = True,
                 gradient_checkpointing: bool = True, max_len: int = 256):
        import torch
        from transformers import AutoModel, AutoTokenizer

        spec = MODELS[kind]
        path = str(spec["dir"]())
        self.kind, self.prefix, self.pool, self.max_len, self.device = kind, spec["prefix"], spec["pool"], max_len, device
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True, dtype=torch.float32)
        if spec["lora"]:
            from peft import LoraConfig, get_peft_model

            cfg = LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.05, bias="none",
                             target_modules=["query", "key", "value", "dense"])
            self.model = get_peft_model(self.model, cfg)
        elif freeze_embeddings:
            self.model.embeddings.word_embeddings.weight.requires_grad_(False)
        if gradient_checkpointing:
            base = self.model.base_model.model if spec["lora"] else self.model
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if spec["lora"]:
                self.model.enable_input_require_grads()
        self.model.to(device)

    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _forward(self, batch_texts: Sequence[str], max_len: int):
        import torch

        enc = self.tokenizer([self.prefix + t for t in batch_texts], truncation=True, max_length=max_len, padding=True,
                             return_tensors="pt").to(self.device)
        hidden = self.model(**enc).last_hidden_state
        if self.pool == "cls":
            vec = hidden[:, 0]
        else:
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            vec = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return torch.nn.functional.normalize(vec, dim=-1)

    def train_forward(self, batch_texts: Sequence[str]):
        return self._forward(batch_texts, self.max_len)

    def encode(self, texts: Sequence[str], batch: int = 64, max_len: int = 512) -> np.ndarray:
        import torch

        self.model.eval()
        lengths = np.asarray([len(t) for t in texts])
        order = np.argsort(-lengths, kind="stable")
        out = np.zeros((len(texts), MODELS[self.kind]["dim"]), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(texts), batch):
                idx = order[s:s + batch]
                out[idx] = self._forward([texts[i] for i in idx], max_len).float().cpu().numpy()
        self.model.train()
        return out


@dataclass
class Recipe:
    kind: str = "e5"
    epochs: int = 7
    lr: float = 3e-5
    batch: int = 64
    tau: float = TAU
    max_len: int = 256
    warmup_steps: int = 20
    weight_decay: float = 0.01
    seed: int = 20261004
    freeze_embeddings: bool = True
    lora_rank: int = 16
    grad_clip: float = 1.0
    device: str = "cuda"
    max_steps: int = 0          # smoke tests only: stop an epoch after this many steps (0 = full epoch)


def train_encoder(recipe: Recipe, train_rows: Sequence[int], encode_rows: Sequence[int], out_dir: Path,
                  save_epochs: Sequence[int] | None = None, log=print) -> dict[str, Any]:
    """Fit the encoder on ``train_rows`` and save the vectors of ``encode_rows`` after the requested epochs.

    The learning rate is constant after warm-up, so the state after epoch e does not depend on the planned
    number of epochs. Vectors are written to ``out_dir/epoch<e>.npy`` ([5980, d], rows not encoded stay 0).
    """
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(recipe.seed)
    np.random.seed(recipe.seed)
    torch.manual_seed(recipe.seed)
    ts = build_trainset(train_rows)
    texts = mp.texts()
    log(f"train set: {len(ts.rows)} rows with partners, {len(ts.idf)} shared articles, kind={recipe.kind}, "
        f"lr={recipe.lr}, batch={recipe.batch}, max_len={recipe.max_len}")
    enc = CaseEncoder(recipe.kind, device=recipe.device, freeze_embeddings=recipe.freeze_embeddings, max_len=recipe.max_len,
                      lora_rank=recipe.lora_rank)
    enc.model.train()
    params = enc.parameters()
    log(f"trainable parameters: {sum(p.numel() for p in params) / 1e6:.2f}M")
    opt = torch.optim.AdamW(params, lr=recipe.lr, weight_decay=recipe.weight_decay)
    rng = np.random.default_rng(recipe.seed)
    encode_rows = np.asarray(encode_rows, dtype=np.int64)
    save = set(save_epochs) if save_epochs is not None else set(range(1, recipe.epochs + 1))
    history, step, started = [], 0, time.perf_counter()
    for epoch in range(1, recipe.epochs + 1):
        batches = epoch_batches(ts, recipe.batch, rng)
        if recipe.max_steps:
            batches = batches[:recipe.max_steps]
        losses = []
        for items in batches:
            w, masked = batch_weights(ts, items)
            z = enc.train_forward([texts[r] for r in items])
            loss = supcon_loss(z, w.to(enc.device), masked.to(enc.device), recipe.tau)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, recipe.grad_clip)
            step += 1
            for g in opt.param_groups:
                g["lr"] = recipe.lr * min(1.0, step / max(1, recipe.warmup_steps))
            opt.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "steps": len(batches),
                        "seconds": time.perf_counter() - started})
        log(f"epoch {epoch}: loss {history[-1]['loss']:.4f} over {len(batches)} steps ({history[-1]['seconds']:.0f}s)")
        if epoch in save:
            vec = np.zeros((len(texts), MODELS[recipe.kind]["dim"]), dtype=np.float32)
            vec[encode_rows] = enc.encode([texts[int(r)] for r in encode_rows])
            np.save(out_dir / f"epoch{epoch}.npy", vec)
    result = {"recipe": asdict(recipe), "history": history, "train_rows": int(len(train_rows)),
              "partner_rows": int(len(ts.rows)), "seconds": time.perf_counter() - started,
              "peak_gpu_mib": int(torch.cuda.max_memory_allocated() / 2**20) if recipe.device == "cuda" and torch.cuda.is_available() else 0}
    (out_dir / "train.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    del enc, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------------------------
# evaluation of learned keys (CPU)
# ---------------------------------------------------------------------------------------------

def neighbour_precision(nb: mp.Neighbours, mem: mp.Mem, eval_rows: Sequence[int], k: int = 10) -> float:
    """Share of the top-k memory entries sharing a specific positive with the query (macro over components)."""
    data = common.release()
    entry = [set(p) for p in mem.positives]
    generic = set(data.generic_idx.tolist())
    entry = [e - generic for e in entry]
    vals, comps = [], []
    for i, r in enumerate(eval_rows):
        spec = set(data.specific_idx[int(r)])
        top = nb.index[i, :k]
        vals.append(float(np.mean([bool(spec & entry[j]) for j in top])))
        comps.append(str(mp.components()[int(r)]))
    return mp.macro(vals, comps)


def eval_keys(eval_rows: np.ndarray, fit_rows: np.ndarray, keys: Sequence[str], cfgs: Sequence[mp.Cfg],
              objective: str = "P_spec") -> dict[str, Any]:
    """Votes-alone objective of every config and neighbour precision@10 of every key for (eval rows, memory of fit rows)."""
    ev = common.population_rows(objective, eval_rows)
    out: dict[str, Any] = {"configs": {}, "precision_at_10": {}, "n_eval": int(len(ev))}
    for pos_mode in sorted({c.pos_mode for c in cfgs}):
        mem = mp.build_mem(fit_rows, pos_mode)
        cache = mp.KeyCache(ev, mem)
        matrix = mem.matrix()
        for cfg in (c for c in cfgs if c.pos_mode == pos_mode):
            vals, comps = mp.per_query_ndcg(mp.dense_block(cache.votes(cfg, matrix)), ev, objective)
            out["configs"][cfg.label()] = mp.macro(vals, comps)
    mem_s = mp.build_mem(fit_rows, "specific")
    ev_s = mp.pspec_rows(eval_rows)
    cache_s = mp.KeyCache(ev_s, mem_s)
    for key in keys:
        out["precision_at_10"][key] = neighbour_precision(cache_s.nb(key), mem_s, ev_s, 10)
    return out


def grid_for(keys: Sequence[str], pos_mode: str = "specific") -> list[mp.Cfg]:
    cfgs = []
    for key in keys:
        if key == "lex":
            cfgs += [mp.Cfg(key, k, "sim", 0.0, pos_mode) for k in mp.KS_X]
        elif key.startswith(("fuse", "mix")):
            cfgs += [mp.Cfg(key, k, w, tau, pos_mode) for k in mp.KS_X for (w, tau) in (("sim", 0.0), ("softmax", 0.5), ("softmax", 1.0), ("softmax", 2.0))]
        else:
            cfgs += [mp.Cfg(key, k, w, tau, pos_mode) for k in mp.KS_X for (w, tau) in (("sim", 0.0), ("softmax", 0.02), ("softmax", 0.05), ("softmax", 0.1))]
    return cfgs


def with_vectors(alias: str, path: Path):
    """Register learned vectors under ``alias`` for ``memory_proto``'s dense keys."""
    mp.register_vectors(alias, path)


# ---------------------------------------------------------------------------------------------
# learned adapters on frozen vectors (CPU): the same case-similarity loss, a linear map instead of an encoder
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AdaptRecipe:
    inputs: str = "bge"                     # bge | e5 | cat (bge and e5 concatenated, scaled to unit norm)
    d_out: int = 256
    lr: float = 3e-4
    tau: float = 0.1
    epochs: tuple[int, ...] = (10, 20, 30)
    batch: int = 256
    weight_decay: float = 1e-2
    seed: int = 0

    def label(self) -> str:
        return f"{self.inputs}-d{self.d_out}-lr{self.lr:g}-tau{self.tau:g}"


def adapter_input(name: str) -> np.ndarray:
    if name in ("lsa", "lsa+clsa"):
        mp.ensure_lsa()
    if name in ("clsa", "lsa+clsa"):
        mp.ensure_clsa()
    if name == "lsa+clsa":                  # NO-EMB: word LSA and character LSA, each unit norm, scaled to unit norm together
        return np.concatenate([mp.qvec("lsa"), mp.qvec("clsa")], axis=1) / np.sqrt(2.0)
    if name == "cat":
        return np.concatenate([mp.qvec("bge"), mp.qvec("e5")], axis=1) / np.sqrt(2.0)
    return mp.qvec(name)


def train_adapter(recipe: AdaptRecipe, fit_rows: Sequence[int]) -> dict[int, np.ndarray]:
    """Linear map trained with the case-similarity loss on ``fit_rows``; unit vectors of ALL release rows per snapshot epoch."""
    import torch

    torch.manual_seed(recipe.seed)
    ts = build_trainset(fit_rows)
    x = torch.from_numpy(adapter_input(recipe.inputs))
    net = torch.nn.Linear(x.shape[1], recipe.d_out, bias=False)
    opt = torch.optim.AdamW(net.parameters(), lr=recipe.lr, weight_decay=recipe.weight_decay)
    rng = np.random.default_rng(recipe.seed)
    out: dict[int, np.ndarray] = {}
    for epoch in range(1, max(recipe.epochs) + 1):
        for items in epoch_batches(ts, recipe.batch, rng):
            w, masked = batch_weights(ts, items)
            z = torch.nn.functional.normalize(net(x[items]), dim=-1)
            loss = supcon_loss(z, w, masked, recipe.tau)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if epoch in recipe.epochs:
            with torch.no_grad():
                out[epoch] = torch.nn.functional.normalize(net(x), dim=-1).numpy().astype(np.float32)
    return out


ADAPT_FAMILIES: dict[str, dict[str, Any]] = {
    # EMB: adapters on frozen bge / bge||e5 vectors. NO-EMB control: the same adapter on LSA vectors, no dense key anywhere.
    "emb": {"family": "EMB", "recipes": (AdaptRecipe(inputs="bge"), AdaptRecipe(inputs="cat")),
            "methods": {"knn_adapt_spec": {"pos_mode": "specific", "objective": "P_spec"},
                        "knn_adapt_all": {"pos_mode": "all", "objective": "P"}},
            "proto": "proto_adapt", "dense_mix": True},
    "lsac": {"family": "NO-EMB", "recipes": (AdaptRecipe(inputs="lsa+clsa"),),
             "methods": {"knn_lsacadapt_spec": {"pos_mode": "specific", "objective": "P_spec"},
                         "knn_lsacadapt_all": {"pos_mode": "all", "objective": "P"}},
             "proto": "proto_lsacadapt", "dense_mix": False},
    "lsa": {"family": "NO-EMB", "recipes": (AdaptRecipe(inputs="lsa"),),
            "methods": {"knn_lsaadapt_spec": {"pos_mode": "specific", "objective": "P_spec"},
                        "knn_lsaadapt_all": {"pos_mode": "all", "objective": "P"}},
            "proto": "proto_lsaadapt", "dense_mix": False},
}
ADAPT_RECIPES = ADAPT_FAMILIES["emb"]["recipes"]
ADAPT_METHODS = ADAPT_FAMILIES["emb"]["methods"]


def adapt_grid(pos_mode: str, alias: str = "cs", dense_mix: bool = True) -> list[mp.Cfg]:
    """Learned-key configurations: the key alone, fused with BM25, and (``dense_mix``) mixed with BM25 and frozen bge."""
    out = [mp.Cfg(alias, k, "softmax", tau, pos_mode) for tau in (0.05, 0.1) for k in (20, 50, 100)]
    for a in mp.ALPHAS:
        out += [mp.Cfg(f"fuse:{alias}:{a:g}", k, "softmax", tau, pos_mode) for tau in (1.0, 2.0) for k in (50, 100)]
    if dense_mix:
        out += [mp.Cfg(f"mix:lex+bge+{alias}:0.4,0.3,0.3", k, "softmax", 1.0, pos_mode) for k in (50, 100)]
    return out


def adapt_group_job(role: str, family: str = "emb") -> dict[str, Any]:
    """Nested selection (recipe, epoch, key, k, weighting) by inner CV, then train on all training rows and score the group."""
    import torch

    torch.set_num_threads(4)
    fam = ADAPT_FAMILIES[family]
    recipes, specs, mix = fam["recipes"], fam["methods"], fam["dense_mix"]
    if family in ("lsa", "lsac"):
        mp.ensure_lsa()
        mp.ensure_clsa() if family == "lsac" else None
    methods = list(specs)
    eval_rows, train_rows = mp.group_by_role(role)
    started = mp.stamp()
    parts: dict[tuple[str, tuple[int, int, int]], list[tuple[np.ndarray, list[str]]]] = {}
    grids = {m: adapt_grid(specs[m]["pos_mode"], dense_mix=mix) for m in methods}
    for name, val_rows, fit_rows in mp.inner_splits(train_rows):
        base: dict[str, tuple[Any, Any, Any, Any]] = {}
        for m in methods:
            ev = common.population_rows(specs[m]["objective"], val_rows)
            mem = mp.build_mem(fit_rows, specs[m]["pos_mode"])
            cache = mp.KeyCache(ev, mem)
            cache.z("lex")
            if mix:
                cache.z("bge")                                    # shared across recipes and epochs
            base[m] = (ev, mem, cache, mem.matrix())
        for ri, recipe in enumerate(recipes):
            for epoch, vec in train_adapter(recipe, fit_rows).items():
                mp.register_array("cs", vec)
                for m in methods:
                    ev, mem, root, matrix = base[m]
                    cache = root.fork("cs")
                    for ci, cfg in enumerate(grids[m]):
                        parts.setdefault((m, (ri, epoch, ci)), []).append(
                            mp.per_query_ndcg(mp.dense_block(cache.votes(cfg, matrix)), ev, specs[m]["objective"]))
        mp.log(f"adapt[{family}] {role}: inner split {name} done ({mp.stamp() - started:.0f}s)")
    finals: dict[str, tuple[AdaptRecipe, int, mp.Cfg, float]] = {}
    for m in methods:
        keys = [k for k in parts if k[0] == m]
        objective = [mp.pooled(parts[k]) for k in keys]
        for k, obj in zip(keys, objective):
            ri, epoch, ci = k[1]
            mp.search_log({"stage": "adapt_inner_cv", "method": m, "role": role, "recipe": recipes[ri].label(), "epoch": epoch,
                           "config": grids[m][ci].label(), "objective": obj, "objective_population": specs[m]["objective"]})
        best = keys[mp.choose(objective)]
        ri, epoch, ci = best[1]
        finals[m] = (recipes[ri], epoch, grids[m][ci], max(objective))
        mp.log(f"adapt[{family}] {role} {m}: selected {recipes[ri].label()} epoch {epoch} {grids[m][ci].label()} "
               f"inner objective {max(objective):.4f}")
    final_vectors: dict[tuple[str, int], np.ndarray] = {}
    summary = {}
    for m in methods:
        recipe, epoch, cfg, obj = finals[m]
        if (recipe.label(), epoch) not in final_vectors:
            final_vectors[(recipe.label(), epoch)] = train_adapter(recipe, train_rows)[epoch]
        mp.register_array("cs", final_vectors[(recipe.label(), epoch)])
        mp.write_rows(m, eval_rows, mp.knn_outer_block(eval_rows, train_rows, cfg))
        info = {"role": role, "recipe": asdict(recipe), "epoch": epoch, "selected": asdict(cfg), "label": cfg.label(),
                "inner_objective": obj, "objective_population": specs[m]["objective"], "train_rows": int(len(train_rows)),
                "eval_rows": int(len(eval_rows)), "seconds": mp.stamp() - started}
        mp.save_group_state(m, role, info)
        summary[m] = info
        if specs[m]["pos_mode"] == "specific":
            mp.write_rows(fam["proto"], eval_rows, mp.proto_block(train_rows, eval_rows, "cs", "specific", True))
            mp.save_group_state(fam["proto"], role, {"role": role, "recipe": asdict(recipe), "epoch": epoch,
                                                      "train_rows": int(len(train_rows)), "eval_rows": int(len(eval_rows))})
    return summary


def run_adapt(workers: int = 3, roles: Sequence[str] | None = None, family: str = "emb") -> None:
    """Create the score files, run the six group jobs (worker processes) and write the manifests."""
    from concurrent.futures import ProcessPoolExecutor

    fam, started = ADAPT_FAMILIES[family], mp.stamp()
    if family in ("lsa", "lsac"):
        mp.ensure_lsa()
        mp.ensure_clsa() if family == "lsac" else None
    for m in list(fam["methods"]) + [fam["proto"]]:
        mp.ensure_scores(m)
    chosen = list(roles) if roles else [r for r, _, _ in common.scoring_groups()]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for fut in [pool.submit(adapt_group_job, role, family) for role in chosen]:
                fut.result()
    else:
        for role in chosen:
            adapt_group_job(role, family)
    mp.merge_logs()
    wall = mp.stamp() - started
    for m in fam["methods"]:
        finish_adapt(m, wall, family)
    finish_proto_adapt(wall, family)


def adapt_inputs(family: str) -> dict[str, str]:
    if family == "lsac":
        return {"lsa_vectors": common.sha256_file(mp.LSA_PATH), "clsa_vectors": common.sha256_file(mp.CLSA_PATH), **mp.text_hashes()}
    if family == "lsa":
        return {"lsa_vectors": common.sha256_file(mp.LSA_PATH), **mp.text_hashes()}
    return mp.input_hashes("bge", "e5")


def finish_adapt(method_id: str, wall: float, family: str = "emb") -> None:
    fam = ADAPT_FAMILIES[family]
    spec, states = fam["methods"][method_id], mp.group_states(method_id)
    manifest = {
        "method": method_id, "family": fam["family"],
        "kind": "kNN citation-memory votes with a learned linear case-similarity key (votes only)"
                + ("" if family == "emb" else "; NO-EMB control: the same adapter on LSA vectors (no pretrained neural model)"),
        "training_data": "per group: adapter and memory from that group's training rows only; selection by inner CV over its training folds",
        "configuration": {"adapter": ("linear map of frozen vectors (bge or bge||e5)" if family == "emb" else "linear map of LSA vectors" if family == "lsa" else "linear map of word-LSA||char-LSA vectors")
                                     + ", unit-normalized output, trained with the IDF-weighted multi-positive InfoNCE over "
                                       "specific-article positives of different components",
                          "recipes": [asdict(r) for r in fam["recipes"]],
                          "grid": [c.label() for c in adapt_grid(spec["pos_mode"], dense_mix=fam["dense_mix"])],
                          "pos_mode": spec["pos_mode"],
                          "selected_per_group": {r: {"recipe": s["recipe"], "epoch": s["epoch"], "config": s["label"]}
                                                 for r, s in states.items()}},
        "selection": {"protocol": "inner CV over the group's training dev folds, adapter retrained on each inner fitting set; pooled "
                                  "component-macro nDCG@10; ties -> lowest grid index", "objective_population": spec["objective"],
                      "inner_objective_per_group": {r: s["inner_objective"] for r, s in states.items()},
                      "search_log": str(mp.WORK / "search_log.jsonl")},
        "seeds": {"adapter_torch_and_batches": 0}, "input_sha256": adapt_inputs(family), "wall_seconds": wall, "device": "cpu",
    }
    manifest["dev_oof"] = mp.dev_metrics(common.open_scores(method_id))
    common.finish_scores(method_id, manifest, [mp.CODE_FILE, Path(__file__)])
    mp.search_log({"stage": "final_dev_oof", "method": method_id, "dev_oof": manifest["dev_oof"]})
    mp.log(f"{method_id}: finished; dev OOF P_spec {manifest['dev_oof']['P_spec']['ndcg_at_10']:.4f} "
           f"P {manifest['dev_oof']['P']['ndcg_at_10']:.4f}")


def finish_proto_adapt(wall: float, family: str = "emb") -> None:
    fam = ADAPT_FAMILIES[family]
    method_id = fam["proto"]
    states = mp.group_states(method_id)
    manifest = {
        "method": method_id, "family": fam["family"],
        "kind": "article prototypes in the learned adapter space" + ("" if family == "emb" else " (NO-EMB control, LSA adapter)"),
        "training_data": "per group: adapter and prototypes from that group's training rows only",
        "configuration": {"prototype": "specific positives, component-normalized (the frozen-vector variant chosen on dev OOF)",
                          "adapter_recipe_per_group": {r: {"recipe": s["recipe"], "epoch": s["epoch"]} for r, s in states.items()},
                          "recipe_source": "the recipe selected by the group's inner CV for the specific-memory kNN method"},
        "selection": {"protocol": "recipe inherited from the specific-memory kNN inner CV (no separate selection)"},
        "seeds": {"adapter_torch_and_batches": 0}, "input_sha256": adapt_inputs(family), "wall_seconds": wall, "device": "cpu",
    }
    manifest["dev_oof"] = mp.dev_metrics(common.open_scores(method_id))
    common.finish_scores(method_id, manifest, [mp.CODE_FILE, Path(__file__)])
    mp.log(f"{method_id}: finished; dev OOF P_spec {manifest['dev_oof']['P_spec']['ndcg_at_10']:.4f} "
           f"P {manifest['dev_oof']['P']['ndcg_at_10']:.4f}")


# ---------------------------------------------------------------------------------------------
# cosine-softmax probe on frozen vectors (CPU): prototypes tuned with cross-entropy
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ProbeRecipe:
    inputs: str = "bge"                      # bge | e5 | cat
    pos_mode: str = "specific"               # targets: uniform over specific (or all) positives of a training query
    scale: float = 20.0
    lr: float = 3e-3
    epochs: tuple[int, ...] = (5, 10, 20, 40)
    batch: int = 256
    seed: int = 0

    def label(self) -> str:
        return f"{self.inputs}-{self.pos_mode}-s{self.scale:g}-lr{self.lr:g}"


PROBE_RECIPES = tuple(ProbeRecipe(inputs=i, pos_mode=p) for i in ("bge", "cat") for p in ("specific", "all"))


def train_probe(recipe: ProbeRecipe, fit_rows: Sequence[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per snapshot epoch: (cited article indices, unit class vectors). Classes start at the weighted prototypes."""
    import torch

    data = common.release()
    torch.manual_seed(recipe.seed)
    source = data.positive_idx if recipe.pos_mode == "all" else data.specific_idx
    rows = np.asarray([int(r) for r in fit_rows if source[int(r)]], dtype=np.int64)
    cited = np.asarray(sorted({a for r in rows for a in source[r]}), dtype=np.int64)
    index = {int(a): i for i, a in enumerate(cited)}
    targets = np.zeros((len(rows), len(cited)), dtype=np.float32)
    for i, r in enumerate(rows):
        for a in source[r]:
            targets[i, index[a]] = 1.0 / len(source[r])
    x = torch.from_numpy(adapter_input(recipe.inputs))
    xt, tt = x[rows], torch.from_numpy(targets)
    classes = torch.nn.Parameter(torch.nn.functional.normalize(tt.T @ xt, dim=-1).clone())
    opt = torch.optim.AdamW([classes], lr=recipe.lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(recipe.seed)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for epoch in range(1, max(recipe.epochs) + 1):
        perm = torch.randperm(len(rows), generator=gen)
        for s in range(0, len(rows), recipe.batch):
            b = perm[s:s + recipe.batch]
            logits = recipe.scale * xt[b] @ torch.nn.functional.normalize(classes, dim=-1).T
            loss = -(tt[b] * torch.log_softmax(logits, dim=1)).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if epoch in recipe.epochs:
            out[epoch] = (cited, torch.nn.functional.normalize(classes, dim=-1).detach().numpy().astype(np.float32))
    return out


def probe_block(recipe: ProbeRecipe, cited: np.ndarray, classes: np.ndarray, eval_rows: Sequence[int]) -> np.ndarray:
    """Cosine of the evaluation queries to the class vectors; articles without a class get the row minimum."""
    x = adapter_input(recipe.inputs)[np.asarray(eval_rows, dtype=np.int64)]
    sims = x @ classes.T
    out = np.empty((len(eval_rows), mp.n_candidates()), dtype=np.float32)
    out[:] = sims.min(axis=1, keepdims=True)
    out[:, cited] = sims
    return out


def probe_group_job(role: str, recipes: Sequence[ProbeRecipe] = PROBE_RECIPES) -> dict[str, Any]:
    """Nested selection (recipe, epoch) by inner CV on P_spec, then train on all training rows and score the group."""
    eval_rows, train_rows = mp.group_by_role(role)
    started = mp.stamp()
    parts: dict[tuple[int, int], list[tuple[np.ndarray, list[str]]]] = {}
    for name, val_rows, fit_rows in mp.inner_splits(train_rows):
        ev = mp.pspec_rows(val_rows)
        for ri, recipe in enumerate(recipes):
            for epoch, (cited, classes) in train_probe(recipe, fit_rows).items():
                parts.setdefault((ri, epoch), []).append(mp.per_query_ndcg(probe_block(recipe, cited, classes, ev), ev, "P_spec"))
    keys = list(parts)
    objective = [mp.pooled(parts[k]) for k in keys]
    for k, obj in zip(keys, objective):
        mp.search_log({"stage": "probe_inner_cv", "method": "proto_probe", "role": role, "recipe": recipes[k[0]].label(), "epoch": k[1],
                       "objective": obj, "objective_population": "P_spec"})
    ri, epoch = keys[mp.choose(objective)]
    recipe = recipes[ri]
    cited, classes = train_probe(recipe, train_rows)[epoch]
    mp.write_rows("proto_probe", eval_rows, probe_block(recipe, cited, classes, eval_rows))
    info = {"role": role, "recipe": asdict(recipe), "epoch": epoch, "label": f"{recipe.label()}|e{epoch}", "inner_objective": max(objective),
            "objective_population": "P_spec", "train_rows": int(len(train_rows)), "eval_rows": int(len(eval_rows)),
            "seconds": mp.stamp() - started}
    mp.save_group_state("proto_probe", role, info)
    mp.log(f"probe {role}: selected {info['label']} inner objective {max(objective):.4f} ({info['seconds']:.0f}s)")
    return info


def run_probe(workers: int = 3, roles: Sequence[str] | None = None) -> None:
    from concurrent.futures import ProcessPoolExecutor

    started = mp.stamp()
    mp.ensure_scores("proto_probe")
    chosen = list(roles) if roles else [r for r, _, _ in common.scoring_groups()]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for fut in [pool.submit(probe_group_job, role) for role in chosen]:
                fut.result()
    else:
        for role in chosen:
            probe_group_job(role)
    mp.merge_logs()
    states = mp.group_states("proto_probe")
    manifest = {
        "method": "proto_probe", "family": "EMB", "kind": "article class vectors tuned with a cosine-softmax cross-entropy (learned prototypes)",
        "training_data": "per group: the group's training rows only (targets uniform over each query's positives)",
        "configuration": {"initialization": "weighted prototypes (1/|positives|)", "scale": 20.0, "optimizer": "AdamW lr 3e-3, batch 256",
                          "recipes": [asdict(r) for r in PROBE_RECIPES], "unseen_articles": "row minimum",
                          "selected_per_group": {r: s["label"] for r, s in states.items()}},
        "selection": {"protocol": "inner CV over the group's training dev folds (probe retrained on each inner fitting set); pooled "
                                  "component-macro P_spec nDCG@10; ties -> lowest grid index",
                      "inner_objective_per_group": {r: s["inner_objective"] for r, s in states.items()},
                      "search_log": str(mp.WORK / "search_log.jsonl")},
        "seeds": {"torch": 0}, "input_sha256": mp.input_hashes("bge", "e5"), "wall_seconds": mp.stamp() - started, "device": "cpu",
    }
    manifest["dev_oof"] = mp.dev_metrics(common.open_scores("proto_probe"))
    common.finish_scores("proto_probe", manifest, [mp.CODE_FILE, Path(__file__)])
    mp.log(f"proto_probe: finished; dev OOF P_spec {manifest['dev_oof']['P_spec']['ndcg_at_10']:.4f} "
           f"P {manifest['dev_oof']['P']['ndcg_at_10']:.4f}")


# ---------------------------------------------------------------------------------------------
# learned encoders: GPU jobs in short-lived processes, inner-holdout selection, group scoring
# ---------------------------------------------------------------------------------------------

FIRST_EPOCH = 3                      # earlier epochs are never competitive (pilot: holdout objective flat from epoch 3 on)
GPU_PYTHON = Path(r"C:\uol_legacy_home\.venv\gpu\Scripts\python.exe")
ENC_METHODS = {
    # suffix: memory positives, objective population (the same split of labour as knn_x_spec / knn_x_all)
    "spec": {"pos_mode": "specific", "objective": "P_spec"},
    "all": {"pos_mode": "all", "objective": "P"},
}


def inner_holdout(train_rows: np.ndarray) -> tuple[str, np.ndarray, np.ndarray]:
    """The lowest-numbered training dev fold is the inner validation fold h; the rest fits model A."""
    name, val, fit = mp.inner_splits(train_rows)[0]
    return name, val, fit


def stage_rows(role: str, stage: str) -> tuple[np.ndarray, np.ndarray]:
    """(fit rows, rows to encode) of a GPU stage. A: training rows minus the holdout fold h; B: all training rows."""
    eval_rows, train_rows = mp.group_by_role(role)
    if stage == "A":
        return inner_holdout(train_rows)[2], train_rows
    if stage == "B":
        return train_rows, np.union1d(train_rows, eval_rows)
    raise ValueError(stage)


def _log_to(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"{mp._utc()} {message}"
        print(line, flush=True)
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
    return log


def train_job(recipe: Recipe, role: str, stage: str, out_dir: Path, save_epochs: Sequence[int] | None = None) -> dict[str, Any]:
    """GPU work in its own short-lived process (``launch_train``), inside ONE acquisition of the study GPU lock.

    Stage ``A`` is the whole group: model A (holdout fit), the inner-holdout selection (CPU, still inside the lock, about
    a minute) and model B for the selected epochs, so a group needs a single lock acquisition. Stage ``B`` trains model B
    alone (kept for restarts).
    """
    if stage == "A":
        return group_gpu_job(recipe, role, out_dir.parent, save_epochs)
    fit, encode = stage_rows(role, stage)
    log = _log_to(out_dir / "train.log")
    with common.gpu_lock(owner=OWNER):
        log(f"GPU lock acquired: {recipe.kind} {role} stage {stage}; fit rows {len(fit)}, encode rows {len(encode)}")
        result = train_encoder(recipe, fit, encode, out_dir, save_epochs, log=log)
    log("GPU lock released")
    return result


def group_gpu_job(recipe: Recipe, role: str, base: Path, a_epochs: Sequence[int] | None = None) -> dict[str, Any]:
    """Model A, holdout selection and model B of one group under a single lock acquisition (about one fold per lock)."""
    epochs = list(a_epochs) if a_epochs else list(range(FIRST_EPOCH, recipe.epochs + 1))
    log = _log_to(base / "A" / "train.log")
    result: dict[str, Any] = {}
    with common.gpu_lock(owner=OWNER):
        log(f"GPU lock acquired: {recipe.kind} {role} model A, selection and model B in one acquisition")
        if not (base / "A" / "train.json").exists():
            fit, encode = stage_rows(role, "A")
            result["A"] = train_encoder(recipe, fit, encode, base / "A", epochs, log=log)
        table = holdout_table(role, base, epochs)
        choice = {m: pick(rows) for m, rows in table.items()}
        needed = sorted({c["epoch"] for c in choice.values()})
        (base / "selection.json").write_text(json.dumps(
            {m: {"epoch": c["epoch"], "config": c["config"].label(), "holdout_objective": c["objective"]} for m, c in choice.items()},
            indent=1), encoding="utf-8")
        log("selected " + "; ".join(f"{m}: epoch {c['epoch']} {c['config'].label()} ({c['objective']:.4f})" for m, c in choice.items()))
        if not all((base / "B" / f"epoch{e}.npy").exists() for e in needed):
            fit, encode = stage_rows(role, "B")
            result["B"] = train_encoder(Recipe(**{**asdict(recipe), "epochs": max(needed)}), fit, encode, base / "B", needed, log=log)
    log("GPU lock released")
    return result


def launch_train(recipe: Recipe, role: str, stage: str, out_dir: Path, save_epochs: Sequence[int] | None = None) -> None:
    """Run ``train_job`` in a fresh process with the GPU environment so the CUDA context never outlives the job."""
    import subprocess

    cmd = [str(GPU_PYTHON), "-B", "-m", "uol.embedding_push.case_sim", "train", "--kind", recipe.kind, "--role", role,
           "--stage", stage, "--epochs", str(recipe.epochs), "--lr", repr(recipe.lr), "--batch", str(recipe.batch),
           "--max-len", str(recipe.max_len), "--tau", repr(recipe.tau), "--seed", str(recipe.seed), "--out", str(out_dir)]
    if save_epochs:
        cmd += ["--save-epochs", *[str(e) for e in save_epochs]]
    env = {**os.environ, "PYTHONPATH": str(common.ROOT / "src"), "TOKENIZERS_PARALLELISM": "false"}
    subprocess.run(cmd, check=True, env=env)


def holdout_table(role: str, base: Path, epochs: Sequence[int], alias: str = "cs") -> dict[str, list[dict[str, Any]]]:
    """Votes-alone objective of every (epoch, configuration) on the inner holdout fold h, memory = the other training folds."""
    eval_rows, train_rows = mp.group_by_role(role)
    h_name, h_rows, fit_a = inner_holdout(train_rows)
    roots = {}
    for m, spec in ENC_METHODS.items():
        ev = common.population_rows(spec["objective"], h_rows)
        mem = mp.build_mem(fit_a, spec["pos_mode"])
        cache = mp.KeyCache(ev, mem)
        cache.z("lex"), cache.z("bge")
        roots[m] = (ev, mem, cache, mem.matrix())
    out: dict[str, list[dict[str, Any]]] = {m: [] for m in ENC_METHODS}
    for e in epochs:
        mp.register_array(alias, np.load(base / "A" / f"epoch{e}.npy"))
        for m, spec in ENC_METHODS.items():
            ev, mem, root, matrix = roots[m]
            cache = root.fork(alias)
            for cfg in adapt_grid(spec["pos_mode"], alias):
                vals, comps = mp.per_query_ndcg(mp.dense_block(cache.votes(cfg, matrix)), ev, spec["objective"])
                out[m].append({"epoch": e, "config": cfg, "objective": mp.macro(vals, comps)})
        mp.log(f"{base.name} {role}: holdout {h_name} epoch {e} evaluated")
    return out


def pick(table: list[dict[str, Any]]) -> dict[str, Any]:
    return table[mp.choose([t["objective"] for t in table])]


def encoder_methods(kind: str) -> dict[str, str]:
    return {"spec": f"knn_casesim_{kind}", "all": f"knn_casesim_{kind}_all", "proto": f"proto_casesim_{kind}"}


def encoder_group_job(recipe: Recipe, role: str) -> dict[str, Any]:
    """Model A (holdout fit) -> choose epoch and configuration on h -> model B (all training rows) -> score the group."""
    kind, started = recipe.kind, mp.stamp()
    ids = encoder_methods(kind)
    base = WORK / f"{kind}_{role}"
    eval_rows, train_rows = mp.group_by_role(role)
    epochs = list(range(FIRST_EPOCH, recipe.epochs + 1))
    if not (base / "A" / "train.json").exists():
        launch_train(recipe, role, "A", base / "A", save_epochs=epochs)
    table = holdout_table(role, base, epochs)
    choice = {m: pick(table[m]) for m in ENC_METHODS}
    for m, rows in table.items():
        for t in rows:
            mp.search_log({"stage": "casesim_holdout", "kind": kind, "role": role, "method": ids[m], "epoch": t["epoch"],
                           "config": t["config"].label(), "objective": t["objective"],
                           "objective_population": ENC_METHODS[m]["objective"]})
    needed = sorted({c["epoch"] for c in choice.values()})
    mp.log(f"{kind} {role}: selected " + "; ".join(f"{ids[m]}: epoch {c['epoch']} {c['config'].label()} ({c['objective']:.4f})"
                                                   for m, c in choice.items()))
    if not all((base / "B" / f"epoch{e}.npy").exists() for e in needed):
        launch_train(Recipe(**{**asdict(recipe), "epochs": max(needed)}), role, "B", base / "B", save_epochs=needed)
    info: dict[str, Any] = {}
    for m, c in choice.items():
        mp.register_array("cs", np.load(base / "B" / f"epoch{c['epoch']}.npy"))
        mp.write_rows(ids[m], eval_rows, mp.knn_outer_block(eval_rows, train_rows, c["config"]))
        info[m] = {"role": role, "epoch": c["epoch"], "selected": asdict(c["config"]), "label": c["config"].label(),
                   "inner_objective": c["objective"], "objective_population": ENC_METHODS[m]["objective"],
                   "train_rows": int(len(train_rows)), "eval_rows": int(len(eval_rows)),
                   "model_A": json.loads((base / "A" / "train.json").read_text(encoding="utf-8"))["seconds"],
                   "model_B": json.loads((base / "B" / "train.json").read_text(encoding="utf-8"))["seconds"],
                   "seconds": mp.stamp() - started}
        mp.save_group_state(ids[m], role, info[m])
    mp.register_array("cs", np.load(base / "B" / f"epoch{choice['spec']['epoch']}.npy"))
    mp.write_rows(ids["proto"], eval_rows, mp.proto_block(train_rows, eval_rows, "cs", "specific", True))
    mp.save_group_state(ids["proto"], role, {"role": role, "epoch": choice["spec"]["epoch"], "train_rows": int(len(train_rows)),
                                             "eval_rows": int(len(eval_rows))})
    return info


def run_encoder(kind: str, roles: Sequence[str] | None = None, recipe: Recipe | None = None, finish: bool = True) -> None:
    """The groups one after the other (the GPU is exclusive); with finish the manifests once all six are done."""
    recipe = recipe or Recipe(kind=kind)
    started = mp.stamp()
    ids = encoder_methods(kind)
    for m in ids.values():
        mp.ensure_scores(m)
    for role in (list(roles) if roles else [r for r, _, _ in common.scoring_groups()]):
        if all((mp.STATE / f"{m}.{role}.json").exists() for m in ids.values()):
            mp.log(f"{kind} {role}: already done")
            continue
        encoder_group_job(recipe, role)
        mp.merge_logs()
    if finish:
        finish_encoder(kind, recipe, mp.stamp() - started)


def finish_encoder(kind: str, recipe: Recipe, wall: float) -> None:
    ids = encoder_methods(kind)
    model_files = sorted(Path(MODELS[kind]["dir"]()).glob("*.safetensors")) + sorted(Path(MODELS[kind]["dir"]()).glob("pytorch_model.bin"))
    weights = {p.name: common.sha256_file(p) for p in model_files[:1]}
    train_info = {}
    for role, _, _ in common.scoring_groups():
        a = WORK / f"{kind}_{role}" / "A" / "train.json"
        b = WORK / f"{kind}_{role}" / "B" / "train.json"
        if a.exists() and b.exists():
            ja, jb = json.loads(a.read_text(encoding="utf-8")), json.loads(b.read_text(encoding="utf-8"))
            train_info[role] = {"A_seconds": ja["seconds"], "B_seconds": jb["seconds"], "A_peak_gpu_mib": ja["peak_gpu_mib"],
                                "B_peak_gpu_mib": jb["peak_gpu_mib"]}
    for m, method_id in ids.items():
        states = mp.group_states(method_id)
        base_conf = {"encoder": kind, "recipe": asdict(recipe), "training": "per group: model A on the training rows minus the holdout "
                     "fold, model B on all training rows; both with the IDF-weighted multi-positive InfoNCE",
                     "gpu_training": train_info, "selected_per_group": {r: {k: v for k, v in s.items() if k in ("epoch", "label")}
                                                                       for r, s in states.items()}}
        manifest = {
            "method": method_id, "family": "EMB",
            "kind": ("dense article prototypes in the learned space" if m == "proto"
                     else "kNN citation-memory votes with the learned case-similarity key (votes only)"),
            "training_data": "per group: encoder, memory and selection from that group's training rows only (lockbox: all dev rows)",
            "configuration": {**base_conf, "pos_mode": ENC_METHODS[m]["pos_mode"] if m in ENC_METHODS else "specific (comp_norm)",
                              "grid": [c.label() for c in adapt_grid(ENC_METHODS[m]["pos_mode"])] if m in ENC_METHODS else "n/a"},
            "selection": {"protocol": "single inner holdout fold h (lowest-numbered training dev fold): model A fitted without h, "
                                      "epoch and configuration chosen by votes-alone nDCG@10 on h; model B refitted on all "
                                      "training rows for the chosen epoch" if m in ENC_METHODS else
                                      "epoch inherited from the knn_casesim spec selection",
                          "objective_population": ENC_METHODS[m]["objective"] if m in ENC_METHODS else "P_spec",
                          "inner_objective_per_group": {r: s.get("inner_objective") for r, s in states.items()},
                          "search_log": str(mp.WORK / "search_log.jsonl")},
            "seeds": {"torch_numpy_random": recipe.seed}, "input_sha256": {**mp.input_hashes("bge"), **{f"weights:{k}": v for k, v in weights.items()}},
            "wall_seconds": wall, "device": "cuda:0 GTX 1070 fp32 (training); cpu (selection, scoring)",
        }
        manifest["dev_oof"] = mp.dev_metrics(common.open_scores(method_id))
        common.finish_scores(method_id, manifest, [mp.CODE_FILE, Path(__file__)])
        mp.search_log({"stage": "final_dev_oof", "method": method_id, "dev_oof": manifest["dev_oof"]})
        mp.log(f"{method_id}: finished; dev OOF P_spec {manifest['dev_oof']['P_spec']['ndcg_at_10']:.4f} "
               f"P {manifest['dev_oof']['P']['ndcg_at_10']:.4f}")
    mp.merge_logs()


def pilot(recipe: Recipe, role: str = "dev_fold0") -> dict[str, Any]:
    """Pilot on one dev group: holdout selection, model B, and the outer-fold P_spec of the selected configuration."""
    kind, log = recipe.kind, None
    log = _log_to(WORK / f"pilot_{kind}_{role}.log")
    base = WORK / f"{kind}_{role}"
    eval_rows, train_rows = mp.group_by_role(role)
    if not (base / "A" / "train.json").exists():
        launch_train(recipe, role, "A", base / "A")
    epochs = [e for e in range(1, recipe.epochs + 1) if (base / "A" / f"epoch{e}.npy").exists()]
    table = holdout_table(role, base, epochs)
    summary: dict[str, Any] = {"recipe": asdict(recipe), "holdout": {}}
    for m, rows in table.items():
        by_epoch = {}
        for t in rows:
            by_epoch.setdefault(t["epoch"], []).append(t)
        summary["holdout"][m] = {e: {"best": max(x["objective"] for x in v),
                                     "best_config": max(v, key=lambda x: x["objective"])["config"].label()} for e, v in by_epoch.items()}
        for t in rows:
            mp.search_log({"stage": "casesim_holdout", "kind": kind, "role": role, "method": f"pilot_{kind}_{m}", "epoch": t["epoch"],
                           "config": t["config"].label(), "objective": t["objective"], "objective_population": ENC_METHODS[m]["objective"]})
        log(f"holdout {m}: " + " ".join(f"e{e}={v['best']:.4f}" for e, v in summary["holdout"][m].items()))
    choice = {m: pick(table[m]) for m in ENC_METHODS}
    needed = sorted({c["epoch"] for c in choice.values()})
    if not all((base / "B" / f"epoch{e}.npy").exists() for e in needed):
        launch_train(Recipe(**{**asdict(recipe), "epochs": max(needed)}), role, "B", base / "B", save_epochs=needed)
    summary["outer"] = {}
    for m, c in choice.items():
        mp.register_array("cs", np.load(base / "B" / f"epoch{c['epoch']}.npy"))
        spec = ENC_METHODS[m]
        ev = common.population_rows(spec["objective"], eval_rows)
        mem = mp.build_mem(train_rows, spec["pos_mode"])
        cache = mp.KeyCache(ev, mem)
        matrix = mem.matrix()
        vals, comps = mp.per_query_ndcg(mp.dense_block(cache.votes(c["config"], matrix)), ev, spec["objective"])
        oracle = {}
        for cfg in adapt_grid(spec["pos_mode"]):
            v, cs = mp.per_query_ndcg(mp.dense_block(cache.votes(cfg, matrix)), ev, spec["objective"])
            oracle[cfg.label()] = mp.macro(v, cs)
        best_label = max(oracle, key=oracle.get)
        summary["outer"][m] = {"epoch": c["epoch"], "config": c["config"].label(), "holdout_objective": c["objective"],
                               "outer_fold_objective": mp.macro(vals, comps), "oracle_best_on_outer_fold": [best_label, oracle[best_label]],
                               "p_at_10": neighbour_precision(cache.nb("cs"), mem, ev, 10)}
        log(f"pilot {kind} {role} {m}: epoch {c['epoch']} {c['config'].label()} holdout {c['objective']:.4f} -> outer fold "
            f"{spec['objective']} {summary['outer'][m]['outer_fold_objective']:.4f} (oracle {best_label} {oracle[best_label]:.4f})")
    (WORK / f"pilot_{kind}_{role}.json").write_text(json.dumps(summary, indent=1, default=mp._jsonable), encoding="utf-8")
    mp.merge_logs()
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -B -m uol.embedding_push.case_sim")
    sub = parser.add_subparsers(dest="command", required=True)

    def recipe_args(p):
        p.add_argument("--kind", default="e5")
        p.add_argument("--epochs", type=int, default=7)
        p.add_argument("--lr", type=float, default=3e-5)
        p.add_argument("--batch", type=int, default=64)
        p.add_argument("--max-len", type=int, default=256)
        p.add_argument("--tau", type=float, default=TAU)
        p.add_argument("--seed", type=int, default=20261004)

    p = sub.add_parser("train", help="one GPU job (run in its own process)")
    recipe_args(p)
    p.add_argument("--role", required=True)
    p.add_argument("--stage", required=True, choices=("A", "B"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--save-epochs", type=int, nargs="*", default=None)
    p = sub.add_parser("pilot")
    recipe_args(p)
    p.add_argument("--role", default="dev_fold0")
    p = sub.add_parser("run")
    recipe_args(p)
    p.add_argument("--roles", nargs="*", default=None)
    p.add_argument("--no-finish", action="store_true")
    p = sub.add_parser("adapt")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--roles", nargs="*", default=None)
    p.add_argument("--family", default="emb", choices=sorted(ADAPT_FAMILIES))
    p = sub.add_parser("probe")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--roles", nargs="*", default=None)
    args = parser.parse_args(argv)
    if args.command == "adapt":
        run_adapt(args.workers, args.roles, args.family)
        return 0
    if args.command == "probe":
        run_probe(args.workers, args.roles)
        return 0
    recipe = Recipe(kind=args.kind, epochs=args.epochs, lr=args.lr, batch=args.batch, max_len=args.max_len, tau=args.tau, seed=args.seed)
    if args.command == "train":
        train_job(recipe, args.role, args.stage, args.out, args.save_epochs)
    elif args.command == "pilot":
        pilot(recipe, args.role)
    elif args.command == "run":
        run_encoder(args.kind, args.roles, recipe, finish=not args.no_finish)
    return 0


if __name__ == "__main__":
    sys.exit(main())
