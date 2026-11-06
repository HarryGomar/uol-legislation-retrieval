"""EP-beta: supervised fine-tuning of dense query towers against frozen article chunk vectors.

Recipe (embedding push, agent EP-beta):

- The article side is frozen: the facts-to-law v2 chunk vectors (bge-m3: CLS + L2; E5-small: attention-mask mean
  + L2).  An article's score is MaxP over its chunks of ``cos(q, v_c)``, scaled by ``1/tau`` in the loss.
- Only the query tower is trained: E5-small by full fine-tuning (``'query: '`` prefix), bge-m3 by LoRA.
- Loss: full-catalogue multi-positive softmax ``-log sum_{p in pos} softmax(s)_p`` over all candidate articles.
  Optional popularity logit adjustment (training logits only): ``logits += lam * log1p(component count)`` with
  fold-local counts (``lam > 0`` is the logit-adjusted loss of Menon et al.: the learned score then does not have
  to carry the prior; ``lam < 0`` is the literal "subtract" reading and is kept as a control).  Generic positives
  are either dropped (their columns leave the softmax altogether, matching the condensed P_spec lists) or kept.
- One model per scoring group (``common.scoring_groups()``): the five dev folds on their training folds, the
  lockbox on all dev rows.  Training rows are exactly ``common.training_rows(role)``; the lockbox never trains.
- Scores written to ``scores/<method_id>`` are raw MaxP cosines (float32).

Command line (GPU Python with ``PYTHONPATH`` pointing at ``W/src``)::

    python -B -m uol.embedding_push.finetune_dense {parity,baseline,pilot,run,report}
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from uol.embedding_push import common
from uol.facts_to_law import dense as f2l_dense
from uol.facts_to_law import fusion, prior

OWNER = "EP-beta"
OUT = common.RUN / "finetune_dense"
CACHE = OUT / "cache"
CKPT = OUT / "checkpoints"
LOGS = OUT / "logs"
SEARCH_LOG = OUT / "search_log.jsonl"
PROGRESS = OUT / "progress"
CODE_FILES = (Path(__file__), Path(common.__file__))
SEED = 20261004

E5_MODEL_DIR = f2l_dense.MODEL_DIR  # hash-bound by facts_to_law.dense (common.E5_DIR holds only the weights)
BGE_MODEL_DIR = common.BGE_DIR

FROZEN_FILES = {
    "e5": {"chunks": common.DERIVED / "e5_small_chunks_v1/chunk_vectors.npy",
           "owner": common.DERIVED / "e5_small_chunks_v1/article_index.npy",
           "ids": common.DERIVED / "e5_small_chunks_v1/ids.json",
           "queries": common.DERIVED / "e5_small_queries_v11/query_vectors.npy"},
    "bge": {"chunks": common.DERIVED / "bge_m3_chunks_v1/chunk_vectors.npy",
            "units": common.DERIVED / "bge_m3_chunks_v1/units.jsonl",
            "queries": common.DERIVED / "bge_m3_chunks_v1/query_vectors.npy"},
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """One fine-tuning configuration; ``name`` is a stable hash used for cache and checkpoint folders."""

    kind: str  # 'e5' (full fine-tune) or 'bge' (LoRA)
    tau: float = 0.05
    lam: float = 0.0  # training logits += lam * log1p(count); 0 disables the adjustment
    generic: str = "drop"  # 'drop' | 'keep'
    lr: float = 2e-5
    epochs: int = 3
    batch: int = 32
    warmup: float = 0.1
    weight_decay: float = 0.01
    clip: float = 1.0
    max_len: int = 512
    token_budget: int = 4096  # padded tokens per micro-batch (rows are independent, so splitting is exact)
    seed: int = SEED
    lora_r: int = 0  # > 0 enables LoRA on query/key/value/dense (bge-m3)
    lora_alpha: float = 0.0  # 0 means 2 * r
    lora_dropout: float = 0.05
    grad_ckpt: bool = False
    train_embeddings: bool = True  # E5 only: whether the word-embedding matrix is trained
    shuffle_seed: int = 0  # > 0: shuffled-label control (positive sets permuted across training queries)

    def __post_init__(self) -> None:
        if self.kind not in ("e5", "bge"):
            raise ValueError("kind must be 'e5' or 'bge'")
        if self.generic not in ("drop", "keep"):
            raise ValueError("generic must be 'drop' or 'keep'")
        if self.tau <= 0 or self.epochs < 1 or self.batch < 1:
            raise ValueError("tau, epochs and batch must be positive")

    @property
    def name(self) -> str:
        fields = dataclasses.asdict(self)
        if not fields["shuffle_seed"]:
            del fields["shuffle_seed"]  # keeps the names of configurations that predate the control
        blob = json.dumps(fields, sort_keys=True)
        return f"{self.kind}-{hashlib.sha1(blob.encode()).hexdigest()[:8]}"

    def asdict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------------------------------------------
# Frozen article side
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class Frozen:
    """Frozen chunk vectors sorted by article, with the article each chunk belongs to (release candidate index)."""

    kind: str
    vecs: np.ndarray  # float32 [n_chunks, d], sorted by owner (stable)
    owner: np.ndarray  # int64 [n_chunks], non-decreasing
    queries: np.ndarray  # frozen query vectors in release row order [n_queries, d]
    n_articles: int

    @property
    def dim(self) -> int:
        return int(self.vecs.shape[1])


def _ascii_key(ids: Sequence[str]) -> list[str]:
    return [s.encode("ascii", "replace").decode("ascii") for s in ids]


@lru_cache(maxsize=2)
def load_frozen(kind: str) -> Frozen:
    data = common.release()
    n = len(data.candidate_ids)
    files = FROZEN_FILES[kind]
    vecs = np.load(files["chunks"])
    queries = np.load(files["queries"])
    if kind == "e5":
        owner = np.load(files["owner"]).astype(np.int64)
        ids = json.loads(files["ids"].read_text(encoding="utf-8"))["candidate_ids"]
        # article_index is a catalogue *position*; five ids differ from the release ids only in one mis-encoded
        # character, so positions, not strings, are authoritative.  Verify that positions agree up to that.
        if len(ids) != n or _ascii_key(ids) != _ascii_key(data.candidate_ids):
            raise ValueError("E5 chunk catalogue order disagrees with the release candidate order")
    else:
        owner = common.chunk_article_index(files["units"])
    if len(vecs) != len(owner) or queries.shape[0] != len(data.queries):
        raise ValueError("frozen vector files disagree with the release")
    order = np.argsort(owner, kind="stable")
    owner, vecs = owner[order], np.ascontiguousarray(vecs[order], dtype=np.float32)
    if len(np.unique(owner)) != n:
        raise ValueError("every article needs at least one chunk")
    return Frozen(kind, vecs, owner, np.asarray(queries, dtype=np.float32), n)


def frozen_maxp(frozen: Frozen, qvecs: np.ndarray) -> np.ndarray:
    """float32 [len(qvecs), n_articles] MaxP scores through ``common.maxp_scores`` (CPU)."""
    out = np.empty((len(qvecs), frozen.n_articles), dtype=np.float32)
    for start, block in common.maxp_scores(qvecs, frozen.vecs, frozen.owner, frozen.n_articles):
        out[start:start + len(block)] = block
    return out


class MaxPTorch:
    """Differentiable MaxP over frozen chunk vectors kept on the GPU.

    Single-chunk articles are gathered directly; multi-chunk articles use a padded index and ``amax``.
    ``__call__(q)`` returns ``[B, n_articles]`` cosine MaxP scores for L2-normalised ``q``.
    """

    def __init__(self, vecs: np.ndarray, owner: np.ndarray, n_articles: int, device: Any) -> None:
        import torch
        counts = np.bincount(owner, minlength=n_articles)
        if (counts == 0).any() or (np.diff(owner) < 0).any():
            raise ValueError("chunks must be sorted by article and cover every article")
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        single = counts == 1
        multi = np.flatnonzero(~single)
        n_chunks = len(owner)
        self.n_chunks, self.n_articles = n_chunks, n_articles
        self.V = torch.from_numpy(np.ascontiguousarray(vecs, dtype=np.float32)).to(device)
        self.single_chunk = torch.from_numpy(starts[single]).to(device)
        if len(multi):
            width = int(counts[multi].max())
            pad = np.full((len(multi), width), n_chunks, dtype=np.int64)  # n_chunks = the -inf dummy column
            for j, a in enumerate(multi):
                pad[j, :counts[a]] = starts[a] + np.arange(counts[a])
            self.multi_idx = torch.from_numpy(pad).to(device)
        else:
            self.multi_idx = None
        order = np.concatenate([np.flatnonzero(single), multi])  # article id of each concatenated column
        inverse = np.empty(n_articles, dtype=np.int64)
        inverse[order] = np.arange(n_articles)
        self.inverse = torch.from_numpy(inverse).to(device)
        self.device = device

    def __call__(self, q: Any) -> Any:
        import torch
        sims = q @ self.V.T
        parts = [sims[:, self.single_chunk]]
        if self.multi_idx is not None:
            ext = torch.cat([sims, sims.new_full((sims.shape[0], 1), -float("inf"))], dim=1)
            parts.append(ext[:, self.multi_idx].amax(dim=-1))
        return torch.cat(parts, dim=1)[:, self.inverse]


# ---------------------------------------------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------------------------------------------


def multi_positive_loss(logits: Any, pos_mask: Any, col_keep: Any | None = None) -> Any:
    """Per-row ``-log sum_{p in pos} softmax(logits)_p`` over the kept columns (``[B]``).

    ``pos_mask`` is a boolean ``[B, N]`` positive indicator; ``col_keep`` a boolean ``[N]`` that removes columns
    (the generic articles when they are dropped) from the softmax.  Positives must lie in kept columns.
    """
    import torch
    if col_keep is not None:
        logits = logits.masked_fill(~col_keep, -float("inf"))
    log_den = torch.logsumexp(logits, dim=1)
    log_num = torch.logsumexp(logits.masked_fill(~pos_mask, -float("inf")), dim=1)
    return log_den - log_num


def adjustment_vector(counts: np.ndarray, lam: float) -> np.ndarray:
    """Training-only logit offset ``lam * log1p(count)`` (float32 ``[N]``)."""
    return (float(lam) * np.log1p(np.asarray(counts, dtype=np.float64))).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------
# Tokenization and the query tower
# ---------------------------------------------------------------------------------------------------------------


def _verify_e5_files() -> None:
    for name, (size, digest) in f2l_dense.MODEL_FILES.items():
        if name in ("model.safetensors", "sentencepiece.bpe.model"):
            continue  # large; tokenizer.json and config.json pin the rest (weights verified once below)
        if common.sha256_file(E5_MODEL_DIR / name) != digest:
            raise ValueError(f"installed E5 file changed: {name}")


@lru_cache(maxsize=2)
def get_tokenizer(kind: str) -> Any:
    from transformers import AutoTokenizer
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    path = E5_MODEL_DIR if kind == "e5" else BGE_MODEL_DIR
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


@lru_cache(maxsize=4)
def tokenize_queries(kind: str, max_len: int) -> list[np.ndarray]:
    """Token ids of every release query, as the frozen vectors were made (E5: 'query: ' prefix, both: truncation)."""
    prefix = "query: " if kind == "e5" else ""
    texts = [prefix + q.text for q in common.release().queries]
    ids = get_tokenizer(kind)(texts, truncation=True, max_length=max_len)["input_ids"]
    return [np.asarray(x, dtype=np.int64) for x in ids]


def collate(tokens: Sequence[np.ndarray], pad_id: int, device: Any) -> tuple[Any, Any]:
    import torch
    width = max(len(t) for t in tokens)
    ids = torch.full((len(tokens), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(tokens), width), dtype=torch.long)
    for i, t in enumerate(tokens):
        ids[i, :len(t)] = torch.from_numpy(t)
        mask[i, :len(t)] = 1
    return ids.to(device), mask.to(device)


def pool(kind: str, hidden: Any, mask: Any) -> Any:
    """E5: attention-mask mean; bge-m3: CLS; both L2-normalised in float32."""
    import torch
    if kind == "e5":
        m = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * m).sum(1) / m.sum(1).clamp_min(1)
    else:
        pooled = hidden[:, 0]
    return torch.nn.functional.normalize(pooled.float(), dim=-1)


def build_tower(cfg: Config, device: Any) -> tuple[Any, list[Any]]:
    """Load the pretrained encoder (and LoRA for bge); returns (model, trainable parameters)."""
    import torch
    from transformers import AutoModel
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    path = E5_MODEL_DIR if cfg.kind == "e5" else BGE_MODEL_DIR
    model = AutoModel.from_pretrained(str(path), local_files_only=True, add_pooling_layer=False,
                                      attn_implementation="sdpa", dtype=torch.float32)
    if cfg.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if cfg.lora_r > 0:
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(cfg.seed)
        lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha or 2 * cfg.lora_r, lora_dropout=cfg.lora_dropout,
                          target_modules=["query", "key", "value", "dense"], bias="none")
        model = get_peft_model(model, lcfg)
    else:
        if not cfg.train_embeddings:
            model.embeddings.word_embeddings.weight.requires_grad_(False)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    return model, params


@contextlib.contextmanager
def _eval_mode(model: Any):
    was = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was)


def encode_rows(model: Any, kind: str, tokens: Sequence[np.ndarray], rows: Sequence[int], device: Any,
                token_budget: int = 16384) -> np.ndarray:
    """L2-normalised float32 vectors for ``rows`` (input order); length-sorted dynamic batches, no gradients."""
    import torch
    pad = get_tokenizer(kind).pad_token_id
    rows = np.asarray(rows, dtype=np.int64)
    lengths = np.asarray([len(tokens[int(r)]) for r in rows])
    order = np.argsort(-lengths, kind="stable")
    out = None
    i = 0
    with _eval_mode(model), torch.inference_mode():
        while i < len(order):
            size = max(1, token_budget // int(lengths[order[i]]))
            sel = order[i:i + size]
            ids, mask = collate([tokens[int(rows[j])] for j in sel], pad, device)
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
            vec = pool(kind, hidden, mask).cpu().numpy()
            if out is None:
                out = np.zeros((len(rows), vec.shape[1]), dtype=np.float32)
            out[sel] = vec
            i += size
    return out if out is not None else np.zeros((0, 0), dtype=np.float32)


def micro_batches(lengths: Sequence[int], batch: Sequence[int], token_budget: int) -> list[list[int]]:
    """Split a batch (positions into ``lengths``) into length-sorted micro-batches under a padded-token budget."""
    order = sorted(batch, key=lambda j: -lengths[j])
    groups: list[list[int]] = []
    current: list[int] = []
    for j in order:
        if current and (len(current) + 1) * lengths[current[0]] > token_budget:
            groups.append(current)
            current = []
        current.append(j)
    if current:
        groups.append(current)
    return groups


# ---------------------------------------------------------------------------------------------------------------
# Group data
# ---------------------------------------------------------------------------------------------------------------


def group_rows(role: str) -> tuple[np.ndarray, np.ndarray]:
    """(rows to score, rows to train on) for a role, from ``common.scoring_groups()`` semantics."""
    train = common.training_rows(role)
    score = common.lockbox_rows() if role == "lockbox" else common.fold_rows(int(role.removeprefix("dev_fold")))
    if np.intersect1d(train, score).size or (role != "lockbox" and np.isin(train, common.lockbox_rows()).any()):
        raise AssertionError("training rows overlap scored rows or the lockbox")
    return score, train


def training_targets(cfg: Config, train_rows: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """(positives per used row, used rows, fold-local component counts) for the training rows.

    Rows without a positive after generic handling are skipped.  Counts use every training row's references.
    """
    data = common.release()
    n = len(data.candidate_ids)
    source = data.specific_idx if cfg.generic == "drop" else data.positive_idx
    used, positives = [], []
    for r in train_rows:
        p = sorted(set(source[int(r)]))
        if p:
            used.append(int(r))
            positives.append(np.asarray(p, dtype=np.int64))
    counts = prior.component_counts(data.references, data.component_by_query,
                                    [data.queries[int(r)].query_id for r in train_rows], n, data.candidate_index)
    if cfg.shuffle_seed:
        # control: each training query gets another training query's positives; marginal popularity is unchanged
        order = np.random.default_rng(cfg.shuffle_seed).permutation(len(positives))
        positives = [positives[i] for i in order]
    return positives, np.asarray(used, dtype=np.int64), counts


# ---------------------------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------------------------


class StopTraining(RuntimeError):
    """Raised after the checkpoint of the requested epoch (tests and time-boxed runs)."""


def _gpu_mem(device: Any = None) -> dict[str, float]:
    import torch
    if device is not None and device.type != "cuda":
        return {"peak_allocated_mib": 0.0, "peak_reserved_mib": 0.0}
    return {"peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1)}


def _lr_lambda(total: int, warmup: float) -> Callable[[int], float]:
    warm = max(1, int(round(total * warmup)))

    def f(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return max(0.0, (total - step) / max(1, total - warm))
    return f


def train_and_encode(cfg: Config, role: str, *, eval_rows: np.ndarray | None = None, max_steps: int = 0,
                     resume: bool = True, device_name: str = "cuda",
                     stop_after_epoch: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
    """Train the query tower for ``role`` and encode the rows of that role (GPU; call inside ``gpu_lock``).

    Returns (query vectors for the role's rows in ``group_rows`` order, statistics).  With ``eval_rows`` (dev only)
    the dev P_spec nDCG@10 of the current tower is logged at the end of each epoch.
    """
    import torch
    t_start = time.time()
    data = common.release()
    n_art = len(data.candidate_ids)
    score_rows, train_rows = group_rows(role)
    positives, used, counts = training_targets(cfg, train_rows)
    frozen = load_frozen(cfg.kind)
    tokens = tokenize_queries(cfg.kind, cfg.max_len)
    lengths = [len(t) for t in tokens]
    pad_id = get_tokenizer(cfg.kind).pad_token_id
    device = torch.device(device_name)
    on_gpu = device.type == "cuda"
    if on_gpu:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model, params = build_tower(cfg, device)
    n_trainable = sum(p.numel() for p in params)
    maxp = MaxPTorch(frozen.vecs, frozen.owner, n_art, device)
    adj = torch.from_numpy(adjustment_vector(counts, cfg.lam)).to(device)
    col_keep = None
    if cfg.generic == "drop" and len(data.generic_idx):
        col_keep = torch.ones(n_art, dtype=torch.bool, device=device)
        col_keep[torch.from_numpy(data.generic_idx).to(device)] = False

    decay = [p for n_, p in model.named_parameters() if p.requires_grad and p.ndim >= 2 and cfg.lora_r == 0]
    no_decay = [p for n_, p in model.named_parameters() if p.requires_grad and not (p.ndim >= 2 and cfg.lora_r == 0)]
    groups = [{"params": g, "weight_decay": wd} for g, wd in ((decay, cfg.weight_decay), (no_decay, 0.0)) if g]
    opt = torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8)
    n_used = len(used)
    used_lengths = [lengths[int(r)] for r in used]
    steps_per_epoch = math.ceil(n_used / cfg.batch)
    total_steps = cfg.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(total_steps, cfg.warmup))

    ckpt_dir = CKPT / cfg.name / role
    ckpt_path = ckpt_dir / "last.pt"
    start_epoch, epoch_losses, curve = 0, [], []
    if resume and cfg.lora_r > 0 and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        named = dict(model.named_parameters())
        for k, v in state["trainable"].items():
            named[k].data.copy_(v.to(device))
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_epoch, epoch_losses, curve = state["epoch"], state["losses"], state["curve"]
        log(f"[{cfg.name}/{role}] resumed from epoch {start_epoch}")
    resumed_from = start_epoch
    model.train()
    log(f"[{cfg.name}/{role}] train rows {len(train_rows)} used {n_used} positives {sum(map(len, positives))} "
        f"trainable params {n_trainable / 1e6:.2f}M steps {total_steps} cfg {json.dumps(cfg.asdict())}")
    t_train = time.time()
    seen_tokens = 0
    step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, cfg.epochs):
        rng = np.random.default_rng(cfg.seed + 1000 * epoch)
        perm = rng.permutation(n_used)
        running, count = 0.0, 0
        for b in range(steps_per_epoch):
            batch = perm[b * cfg.batch:(b + 1) * cfg.batch]
            if len(batch) == 0:
                continue
            batch_loss = 0.0
            for mb in micro_batches(used_lengths, batch.tolist(), cfg.token_budget):
                rows_mb = [int(used[j]) for j in mb]
                ids, mask = collate([tokens[r] for r in rows_mb], pad_id, device)
                hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
                q = pool(cfg.kind, hidden, mask)
                logits = maxp(q) / cfg.tau + adj
                pos_mask = torch.zeros_like(logits, dtype=torch.bool)
                r_idx = np.concatenate([np.full(len(positives[j]), i) for i, j in enumerate(mb)])
                c_idx = np.concatenate([positives[j] for j in mb])
                pos_mask[torch.from_numpy(r_idx).to(device), torch.from_numpy(c_idx).to(device)] = True
                loss = multi_positive_loss(logits, pos_mask, col_keep).sum() / len(batch)
                loss.backward()
                batch_loss += float(loss.detach())
                seen_tokens += int(mask.sum())
            torch.nn.utils.clip_grad_norm_(params, cfg.clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            running += batch_loss
            count += 1
            step += 1
            if step % 25 == 0 or step == 1:
                el = time.time() - t_train
                log(f"[{cfg.name}/{role}] ep {epoch + 1}/{cfg.epochs} step {step}/{total_steps} loss {batch_loss:.4f} "
                    f"avg {running / count:.4f} lr {sched.get_last_lr()[0]:.2e} {seen_tokens / max(el, 1e-9):.0f} tok/s "
                    f"{el:.0f}s peak {_gpu_mem(device)['peak_allocated_mib']:.0f} MiB")
            if max_steps and step >= max_steps:
                break
        epoch_losses.append(running / max(count, 1))
        if eval_rows is not None:
            qv = encode_rows(model, cfg.kind, tokens, eval_rows, device)
            with torch.inference_mode():
                scores = torch.cat([maxp(torch.from_numpy(qv[s:s + 256]).to(device)).cpu()
                                    for s in range(0, len(qv), 256)]).numpy()
            ev = common.evaluate(scores, eval_rows, "P_spec")
            curve.append({"epoch": epoch + 1, "train_loss": round(epoch_losses[-1], 5),
                          "P_spec_ndcg10": round(ev["macro"]["ndcg_at_10"], 5)})
            log(f"[{cfg.name}/{role}] epoch {epoch + 1} train loss {epoch_losses[-1]:.4f} dev P_spec nDCG@10 "
                f"{ev['macro']['ndcg_at_10']:.4f}")
            model.train()
        else:
            log(f"[{cfg.name}/{role}] epoch {epoch + 1} train loss {epoch_losses[-1]:.4f}")
        if cfg.lora_r > 0 and not max_steps:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            state = {"epoch": epoch + 1, "losses": epoch_losses, "curve": curve,
                     "trainable": {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad},
                     "opt": opt.state_dict(), "sched": sched.state_dict()}
            torch.save(state, ckpt_path.with_suffix(".tmp"))
            os.replace(ckpt_path.with_suffix(".tmp"), ckpt_path)
        if stop_after_epoch and epoch + 1 >= stop_after_epoch:
            raise StopTraining(f"stopped after epoch {epoch + 1} as requested")
        if max_steps and step >= max_steps:
            break
    train_seconds = time.time() - t_train
    t_enc = time.time()
    qvecs = encode_rows(model, cfg.kind, tokens, score_rows, device)
    stats = {"role": role, "config": cfg.asdict(), "name": cfg.name, "train_rows": int(len(train_rows)),
             "train_rows_used": int(n_used), "positives": int(sum(map(len, positives))), "trainable_params": n_trainable,
             "resumed_from_epoch": resumed_from, "steps": step, "epoch_losses": [round(x, 5) for x in epoch_losses], "epoch_curve": curve,
             "train_seconds": round(train_seconds, 1), "encode_seconds": round(time.time() - t_enc, 1),
             "wall_seconds": round(time.time() - t_start, 1), "tokens_seen": seen_tokens,
             "gpu": torch.cuda.get_device_name(0) if on_gpu else "cpu", **_gpu_mem(device), "utc": _now()}
    if cfg.lora_r > 0 and not max_steps:
        CKPT.joinpath(cfg.name, role).mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad},
                   CKPT / cfg.name / role / "final_adapter.pt")
        if ckpt_path.exists():
            ckpt_path.unlink()
    del model, opt, maxp
    if on_gpu:
        torch.cuda.empty_cache()
    return qvecs, stats


GPU_PYTHON = Path(r"C:\uol_legacy_home\.venv\gpu\Scripts\python.exe")


def job_cache_path(cfg: Config, role: str, smoke: bool = False) -> Path:
    return CACHE / cfg.name / (f"{role}.smoke.npz" if smoke else f"{role}.npz")


def run_job(cfg: Config, role: str, *, eval_rows: np.ndarray | None = None, max_steps: int = 0,
            use_cache: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    """Cached group job: query vectors for ``role`` under ``cfg``.

    A cache miss runs the job in its own short-lived GPU process (``job`` subcommand), so the GPU lock and all
    CUDA memory are released as soon as the job ends and this (CPU) process never holds the GPU.
    """
    path = job_cache_path(cfg, role, smoke=bool(max_steps))
    if use_cache and not max_steps and path.exists():
        z = np.load(path, allow_pickle=False)
        log(f"[{cfg.name}/{role}] cached query vectors")
        return z["qvecs"], json.loads(str(z["stats"]))
    cmd = [str(GPU_PYTHON), "-B", "-m", "uol.embedding_push.finetune_dense", "job", "--cfg", json.dumps(cfg.asdict()),
           "--role", role]
    if eval_rows is not None:
        cmd.append("--eval")
    if max_steps:
        cmd += ["--max-steps", str(max_steps)]
    env = {**os.environ, "PYTHONPATH": str(common.ROOT / "src")}
    subprocess.run(cmd, check=True, env=env)
    z = np.load(path, allow_pickle=False)
    return z["qvecs"], json.loads(str(z["stats"]))


def cmd_job(cfg: Config, roles: Sequence[str], evaluate: bool = False, max_steps: int = 0) -> None:
    """GPU job(s) in this process under one lock hold: train for each role, save its query vectors, exit."""
    import gc
    import torch
    roles = [r for r in roles if max_steps or not job_cache_path(cfg, r).exists()]
    if any(r == "lockbox" for r in roles) and evaluate:
        raise PermissionError("no evaluation of lockbox rows")
    if not roles:
        return
    with common.gpu_lock(owner=OWNER):
        for role in roles:
            score_rows, _ = group_rows(role)
            path = job_cache_path(cfg, role, smoke=bool(max_steps))
            if not max_steps and path.exists():
                continue
            qvecs, stats = train_and_encode(cfg, role, eval_rows=score_rows if evaluate else None, max_steps=max_steps)
            gc.collect()
            torch.cuda.empty_cache()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.stem + ".tmp.npz")
            np.savez(tmp, qvecs=qvecs, rows=score_rows, stats=json.dumps(stats))
            os.replace(tmp, path)
            log(f"[{cfg.name}/{role}] job done: {stats['wall_seconds']}s, peak {stats['peak_allocated_mib']} MiB -> {path.name}")


# ---------------------------------------------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------------------------------------------


def evaluate_both(scores: np.ndarray, rows: np.ndarray) -> dict[str, Any]:
    out = {}
    for pop in ("P_spec", "P"):
        ev = common.evaluate(scores, rows, pop)
        out[pop] = {**{k: round(float(v), 5) for k, v in ev["macro"].items()}, "n_queries": ev["n_queries"],
                    "n_components": ev["n_components"]}
    return out


def band_recall20(scores: np.ndarray, rows: np.ndarray, role: str) -> dict[str, Any]:
    """Pair-level Recall@20 of the specific positives of P_spec rows by fold-local popularity band.

    Bands follow ``prior.band_of`` on the role's training rows (unseen 0, rare 1-4, frequent 5+ components).
    """
    data = common.release()
    train = common.training_rows(role)
    counts = prior.component_counts(data.references, data.component_by_query,
                                    [data.queries[int(r)].query_id for r in train], len(data.candidate_ids),
                                    data.candidate_index)
    per_query = common.evaluate(scores, rows, "P_spec")["per_query"]
    hits: dict[str, list[int]] = {"unseen": [], "rare": [], "frequent": []}
    for rec in per_query:
        for article, rank in zip(rec["positive_idx"], rec["positive_ranks"]):
            hits[prior.band_of(int(counts[article]))].append(int(rank <= 20))
    return {band: {"pairs": len(v), "hits": int(sum(v)), "recall_at_20": round(float(np.mean(v)), 4) if v else None}
            for band, v in hits.items()}


def top10_pop_share(scores: np.ndarray, rows: np.ndarray, role: str, top_n: int = 50, specific: bool = True,
                    population: str = "P_spec") -> float:
    """Share of top-10 slots (dev rows of ``population``) held by the ``top_n`` most-cited training articles.

    Citation counts are fold-local component counts of the role's training rows.  With ``specific`` the generic
    articles are excluded from both the ranking and the popular set (condensed lists, as in P_spec).
    """
    data = common.release()
    n = len(data.candidate_ids)
    train = common.training_rows(role)
    counts = prior.component_counts(data.references, data.component_by_query,
                                    [data.queries[int(r)].query_id for r in train], n, data.candidate_index).astype(np.float64)
    if specific and len(data.generic_idx):
        counts[data.generic_idx] = -1
    popular = np.argsort(-counts, kind="stable")[:top_n]
    keep = np.asarray(common.population_rows(population, rows))
    local = np.flatnonzero(np.isin(rows, keep))
    block = np.array(scores[local], dtype=np.float32, copy=True)
    if specific and len(data.generic_idx):
        block[:, data.generic_idx] = -np.inf
    top = np.argpartition(-block, 9, axis=1)[:, :10]
    return float(np.isin(top, popular).mean())


# ---------------------------------------------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------------------------------------------


def append_search_log(record: dict[str, Any]) -> None:
    SEARCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SEARCH_LOG.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"utc": _now(), **record}, ensure_ascii=False) + "\n")


def read_search_log() -> list[dict[str, Any]]:
    if not SEARCH_LOG.exists():
        return []
    return [json.loads(x) for x in SEARCH_LOG.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------------------------------


def cmd_parity(kind: str, n: int = 64) -> dict[str, Any]:
    """The untrained query tower must reproduce the frozen query vectors (same space)."""
    import torch
    frozen = load_frozen(kind)
    cfg = Config(kind=kind, lora_r=16 if kind == "bge" else 0)
    tokens = tokenize_queries(kind, 512)
    rows = np.random.default_rng(0).choice(len(tokens), size=n, replace=False)
    with common.gpu_lock(owner=OWNER):
        device = torch.device("cuda")
        model, _ = build_tower(cfg, device)
        qv = encode_rows(model, kind, tokens, rows, device)
    cos = (qv * frozen.queries[rows]).sum(1)
    res = {"kind": kind, "n": n, "min_cos": float(cos.min()), "mean_cos": float(cos.mean())}
    log(f"parity {json.dumps(res)}")
    return res


def cmd_baseline() -> dict[str, Any]:
    """Frozen MaxP reference per dev fold and pooled (CPU); also the top-10 popularity share."""
    out = {}
    for kind in ("e5", "bge"):
        frozen = load_frozen(kind)
        dev = common.dev_rows()
        per_fold = {}
        pooled = np.empty((len(dev), frozen.n_articles), dtype=np.float32)
        pos = {int(r): i for i, r in enumerate(dev)}
        for k in range(5):
            rows = common.fold_rows(k)
            sc = frozen_maxp(frozen, frozen.queries[rows])
            pooled[[pos[int(r)] for r in rows]] = sc
            per_fold[f"dev_fold{k}"] = {**evaluate_both(sc, rows),
                                        "pop50_top10_share_Pspec": round(top10_pop_share(sc, rows, f"dev_fold{k}"), 4)}
        out[kind] = {"per_fold": per_fold, "pooled": evaluate_both(pooled, dev)}
        log(f"frozen {kind}: pooled dev P_spec nDCG@10 {out[kind]['pooled']['P_spec']['ndcg_at_10']} "
            f"P {out[kind]['pooled']['P']['ndcg_at_10']}; fold0 P_spec {per_fold['dev_fold0']['P_spec']['ndcg_at_10']}")
    append_search_log({"stage": "reference_frozen_maxp", "metrics": out})
    return out


def cmd_table(kind: str = "", stage: str = "") -> list[dict[str, Any]]:
    """Print every logged pilot configuration, best dev fold-0 P_spec nDCG@10 first."""
    recs = [r for r in read_search_log() if "config" in r and r["config"]["kind"] == (kind or r["config"]["kind"])
            and (not stage or r["stage"].startswith(stage))]
    recs.sort(key=lambda r: -r["metrics"]["P_spec"]["ndcg_at_10"])
    for r in recs:
        c, m = r["config"], r["metrics"]
        extra = {k: v for k, v in c.items() if k not in ("kind", "tau", "lam", "generic", "lr", "epochs", "batch")
                 and v != dataclasses.asdict(Config(kind=c["kind"])).get(k)}
        print(f"{r['key']} {r['stage']:<14} tau {c['tau']:<5} lam {c['lam']:<5} gen {c['generic']:<4} lr {c['lr']:<7} ep {c['epochs']} "
              f"bs {c['batch']:<3} | Pspec nDCG {m['P_spec']['ndcg_at_10']:.4f} MRR {m['P_spec']['mrr']:.4f} R@20 "
              f"{m['P_spec']['recall_at_20']:.4f} | P nDCG {m['P']['ndcg_at_10']:.4f} | pop50 {r['pop50_top10_share_Pspec']} "
              f"| {r['train']['wall_seconds']}s {extra if extra else ''}")
    return recs


def e5_pilot_grid() -> list[Config]:
    """tau in {.02,.05} x lam in {0,1} x generic {drop,keep}; lr 2e-5, 3 epochs, batch 32."""
    return [Config(kind="e5", tau=t, lam=l, generic=g) for t in (0.02, 0.05) for l in (0.0, 1.0) for g in ("drop", "keep")]


def run_pilot(configs: Sequence[Config], stage: str, fold: int = 0, force: bool = False) -> list[dict[str, Any]]:
    """Train on the other dev folds, evaluate dev fold ``fold`` (P_spec and P); log every configuration."""
    role = f"dev_fold{fold}"
    rows = common.fold_rows(fold)
    done = {r["key"]: r for r in read_search_log() if r.get("stage") == stage and r.get("role") == role}
    results = []
    for cfg in configs:
        if cfg.name in done and not force:
            log(f"[{cfg.name}] already logged for {stage}; skipping")
            results.append(done[cfg.name])
            continue
        t0 = time.time()
        qvecs, stats = run_job(cfg, role, eval_rows=rows, use_cache=not force)
        frozen = load_frozen(cfg.kind)
        scores = frozen_maxp(frozen, qvecs)
        metrics = evaluate_both(scores, rows)
        share = top10_pop_share(scores, rows, role)
        rec = {"stage": stage, "role": role, "key": cfg.name, "config": cfg.asdict(), "metrics": metrics,
               "pop50_top10_share_Pspec": round(share, 4), "band_recall20": band_recall20(scores, rows, role), "train": {k: stats[k] for k in (
                   "train_rows_used", "positives", "steps", "epoch_losses", "epoch_curve", "train_seconds", "encode_seconds",
                   "wall_seconds", "peak_allocated_mib", "peak_reserved_mib", "gpu")}, "seconds_total": round(time.time() - t0, 1)}
        append_search_log(rec)
        results.append(rec)
        log(f"[{cfg.name}] fold {fold}: P_spec nDCG@10 {metrics['P_spec']['ndcg_at_10']:.4f} MRR {metrics['P_spec']['mrr']:.4f} "
            f"R@20 {metrics['P_spec']['recall_at_20']:.4f} | P nDCG@10 {metrics['P']['ndcg_at_10']:.4f} | pop50 share {share:.3f}")
    return results


FUSION_WEIGHTS = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)


def _paired(c: dict[str, Any]) -> dict[str, Any]:
    """Compact view of a ``stats.contrast`` record (dev, unadjusted, descriptive)."""
    return {"effect": round(float(c["effect"]), 5), "ci95": [round(float(x), 5) for x in c["ci95"]],
            "p_sign_flip": round(float(c["p_sign_flip"]), 4), "n_components": c["components"]}


def fusion_check(dense_scores: np.ndarray, lstar_id: str = "L_star", weights: Sequence[float] = FUSION_WEIGHTS,
                 label: str = "") -> dict[str, Any]:
    """Quick check ``z(L*) + w * z(method)`` with robust-z rows; ``w`` chosen per fold on the training folds' OOF rows.

    The weight is chosen separately for P_spec and P (nDCG@10 on the other dev folds' rows, whose scores are
    themselves out-of-fold) and applied to the held-out fold; pooled dev numbers are reported.  Dev rows only.
    """
    manifest = common.SCORES / lstar_id / "manifest.json"
    if not manifest.exists():
        return {"status": f"{lstar_id} manifest not found"}
    base = common.open_scores(lstar_id)
    pooled: dict[str, dict[str, Any]] = {pop: {"rows": [], "fused": []} for pop in ("P_spec", "P")}
    chosen: dict[str, dict[str, float]] = {}
    for k in range(5):
        role, rows, train = f"dev_fold{k}", common.fold_rows(k), common.training_rows(f"dev_fold{k}")
        train_p = common.population_rows("P", train)
        spec_flag = np.isin(train_p, common.population_rows("P_spec", train))
        zb = fusion.robust_z(np.asarray(base[train_p]))
        zd = fusion.robust_z(np.asarray(dense_scores[train_p]))
        curves: dict[str, list[float]] = {"P_spec": [], "P": []}
        for w in weights:
            fused = zb + np.float32(w) * zd
            curves["P"].append(common.evaluate(fused, train_p, "P")["macro"]["ndcg_at_10"])
            curves["P_spec"].append(common.evaluate(fused[spec_flag], train_p[spec_flag], "P_spec")["macro"]["ndcg_at_10"])
        del zb, zd
        best = {pop: float(weights[int(np.argmax(c))]) for pop, c in curves.items()}
        chosen[role] = {**{f"w_{pop}": w for pop, w in best.items()},
                        **{f"train_{pop}_curve": [round(x, 4) for x in c] for pop, c in curves.items()}}
        for pop in ("P_spec", "P"):
            sel = common.population_rows(pop, rows)
            fused = fusion.robust_z(np.asarray(base[sel])) + np.float32(best[pop]) * fusion.robust_z(np.asarray(dense_scores[sel]))
            pooled[pop]["rows"].append(sel)
            pooled[pop]["fused"].append(fused)
        log(f"[fusion {label}] {role}: w P_spec {best['P_spec']} w P {best['P']}")
    out: dict[str, Any] = {"lstar": lstar_id, "weights_grid": list(weights), "chosen": chosen}
    for pop in ("P_spec", "P"):
        rows = np.concatenate(pooled[pop]["rows"])
        fused = np.concatenate(pooled[pop]["fused"])
        ev = common.evaluate(fused, rows, pop)
        base_ev = common.evaluate(np.asarray(base[rows]), rows, pop)
        dense_ev = common.evaluate(np.asarray(dense_scores[rows]), rows, pop)
        out[pop] = {"fused": {k: round(float(v), 5) for k, v in ev["macro"].items()},
                    "L_star_alone": {k: round(float(v), 5) for k, v in base_ev["macro"].items()},
                    "method_alone": {k: round(float(v), 5) for k, v in dense_ev["macro"].items()},
                    "n_queries": ev["n_queries"], "n_components": ev["n_components"],
                    "paired_vs_L_star_ndcg10": _paired(common.contrast(ev, base_ev))}
    return out


def cmd_report(method_id: str, lstar_id: str = "L_star", skip_fusion: bool = False) -> dict[str, Any]:
    """Dev OOF metrics of a delivered method, popularity shift versus the frozen MaxP, and the L* fusion check."""
    scores = common.open_scores(method_id)
    dev = common.dev_rows()
    kind = "bge" if "bge" in method_id else "e5"
    frozen = load_frozen(kind)
    block = np.asarray(scores[dev])
    out: dict[str, Any] = {"method": method_id, "pooled_dev_oof": evaluate_both(block, dev), "per_fold": {}}
    frozen_block = np.empty_like(block)
    pos = {int(r): i for i, r in enumerate(dev)}
    for k in range(5):
        rows = common.fold_rows(k)
        idx = [pos[int(r)] for r in rows]
        frozen_block[idx] = frozen_maxp(frozen, frozen.queries[rows])
        out["per_fold"][f"dev_fold{k}"] = {
            **evaluate_both(block[idx], rows),
            "pop50_top10_share_Pspec": round(top10_pop_share(block[idx], rows, f"dev_fold{k}"), 4),
            "pop50_top10_share_Pspec_frozen": round(top10_pop_share(frozen_block[idx], rows, f"dev_fold{k}"), 4)}
    out["frozen_pooled_dev"] = evaluate_both(frozen_block, dev)
    bands = {"method": {}, "frozen": {}}
    for k in range(5):
        rows = common.fold_rows(k)
        idx = [pos[int(r)] for r in rows]
        for label, matrix in (("method", block), ("frozen", frozen_block)):
            for band, v in band_recall20(matrix[idx], rows, f"dev_fold{k}").items():
                acc = bands[label].setdefault(band, {"pairs": 0, "hits": 0})
                acc["pairs"] += v["pairs"]
                acc["hits"] += v["hits"]
    out["band_recall20_pooled_dev"] = {label: {band: {**acc, "recall_at_20": round(acc["hits"] / max(acc["pairs"], 1), 4)}
                                              for band, acc in d.items()} for label, d in bands.items()}
    out["pop50_top10_share_Pspec_mean_over_folds"] = round(float(np.mean(
        [v["pop50_top10_share_Pspec"] for v in out["per_fold"].values()])), 4)
    out["pop50_top10_share_Pspec_frozen_mean_over_folds"] = round(float(np.mean(
        [v["pop50_top10_share_Pspec_frozen"] for v in out["per_fold"].values()])), 4)
    # share of the true specific positives that fall into the 50 most-cited training articles (the ceiling for context)
    data = common.release()
    shares = []
    for k in range(5):
        train = common.training_rows(f"dev_fold{k}")
        counts = prior.component_counts(data.references, data.component_by_query,
                                        [data.queries[int(r)].query_id for r in train], len(data.candidate_ids),
                                        data.candidate_index).astype(np.float64)
        counts[data.generic_idx] = -1
        popular = set(np.argsort(-counts, kind="stable")[:50].tolist())
        pos_rows = common.population_rows("P_spec", common.fold_rows(k))
        shares.append(float(np.mean([p in popular for r in pos_rows for p in data.specific_idx[int(r)]])))
    out["pop50_share_of_true_specific_positives_mean"] = round(float(np.mean(shares)), 4)
    del block, frozen_block
    if not skip_fusion:
        out["fusion_with_L_star"] = fusion_check(scores, lstar_id, label=method_id)
    (OUT / f"report_{method_id}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"[report {method_id}] pooled dev P_spec {out['pooled_dev_oof']['P_spec']['ndcg_at_10']} "
        f"P {out['pooled_dev_oof']['P']['ndcg_at_10']} (frozen {out['frozen_pooled_dev']['P_spec']['ndcg_at_10']} / "
        f"{out['frozen_pooled_dev']['P']['ndcg_at_10']}); pop50 share {out['pop50_top10_share_Pspec_mean_over_folds']} "
        f"(frozen {out['pop50_top10_share_Pspec_frozen_mean_over_folds']})")
    return out


def method_dir(method_id: str) -> Path:
    return common.score_path(method_id).parent


def selection_evidence(cfg: Config) -> dict[str, Any]:
    """Dev fold-0 results of every logged pilot configuration of this kind, and where ``cfg`` stands."""
    recs = [r for r in read_search_log() if "config" in r and r["config"]["kind"] == cfg.kind and r.get("role") == "dev_fold0"
            and not r["config"].get("shuffle_seed")]
    table = sorted(({"key": r["key"], "stage": r["stage"], "tau": r["config"]["tau"], "lam": r["config"]["lam"],
                    "generic": r["config"]["generic"], "lr": r["config"]["lr"], "epochs": r["config"]["epochs"],
                    "P_spec_ndcg10": r["metrics"]["P_spec"]["ndcg_at_10"], "P_ndcg10": r["metrics"]["P"]["ndcg_at_10"]}
                    for r in recs), key=lambda x: -x["P_spec_ndcg10"])
    return {"rule": "highest dev fold-0 P_spec nDCG@10 among all logged configurations (trained on dev folds 1-4)",
            "chosen_key": cfg.name, "chosen_rank_in_log": next((i + 1 for i, t in enumerate(table) if t["key"] == cfg.name), None),
            "n_logged_configurations": len(table), "fold0_table": table}


def run_method(method_id: str, cfg: Config, selection: dict[str, Any]) -> dict[str, Any]:
    """All six scoring groups for ``cfg`` into ``scores/<method_id>`` (resumable by group), then the manifest."""
    path = common.score_path(method_id)
    progress_path = PROGRESS / f"{method_id}.json"
    PROGRESS.mkdir(parents=True, exist_ok=True)
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {"groups": {}}
    if path.exists():
        matrix = np.load(path, mmap_mode="r+")
    else:
        matrix = common.create_scores(method_id)
        progress = {"groups": {}}
    frozen = load_frozen(cfg.kind)
    t0 = time.time()
    for role, rows, _train in common.scoring_groups():
        if role in progress["groups"]:
            log(f"[{method_id}] {role} already written")
            continue
        qvecs, stats = run_job(cfg, role)
        for start, block in common.maxp_scores(qvecs, frozen.vecs, frozen.owner, frozen.n_articles):
            matrix[rows[start:start + len(block)]] = block
        matrix.flush()
        progress["groups"][role] = {k: stats[k] for k in ("train_rows_used", "positives", "steps", "epoch_losses",
                                                         "train_seconds", "encode_seconds", "wall_seconds",
                                                         "peak_allocated_mib", "utc")}
        progress_path.write_text(json.dumps(progress, indent=1), encoding="utf-8")
        log(f"[{method_id}] wrote {role}: {len(rows)} rows")
    del matrix
    selection = {**selection, **selection_evidence(cfg)}
    wall = sum(g["wall_seconds"] for g in progress["groups"].values())
    manifest = {
        "method": method_id, "family": "EMB: fine-tuned dense retriever (query tower only, frozen article chunk vectors)",
        "configuration": cfg.asdict(),
        "recipe": {"article_score": "MaxP over frozen chunk vectors, cosine (scores written are raw cosines)",
                   "loss": "full-catalogue multi-positive softmax over all 39,178 articles, scaled by 1/tau",
                   "logit_adjustment": "training logits += lam*log1p(component count of the group's training rows); none at inference",
                   "generic": cfg.generic, "train_rows": "all strata with at least one positive, common.training_rows(role)",
                   "groups": "5 dev folds (trained on the other four dev folds) + lockbox (trained on all dev rows)"},
        "selection": selection,
        "seeds": {"torch_numpy": cfg.seed, "batch_order": "default_rng(seed + 1000*epoch)"},
        "inputs_sha256": {**{f"frozen_{k}": common.sha256_file(v) for k, v in FROZEN_FILES[cfg.kind].items() if k != "ids"}},
        "model_dir": str(E5_MODEL_DIR if cfg.kind == "e5" else BGE_MODEL_DIR),
        "groups": progress["groups"], "wall_seconds_gpu_jobs": round(wall, 1),
        "wall_seconds_total_incl_waiting": round(time.time() - t0, 1),
        "device": _device_string(), "versions": _versions(),
    }
    return common.finish_scores(method_id, manifest, CODE_FILES)


def _device_string() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True).stdout.strip().splitlines()[0] + ", fp32"
    except Exception:
        return "unknown"


def _versions() -> dict[str, str]:
    import torch
    import transformers
    out = {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__}
    try:
        import peft
        out["peft"] = peft.__version__
    except ImportError:
        pass
    return out


def cfg_from_args(args: argparse.Namespace) -> Config:
    fields = {f.name for f in dataclasses.fields(Config)}
    given = json.loads(args.cfg) if getattr(args, "cfg", None) else {}
    bad = set(given) - fields
    if bad:
        raise ValueError(f"unknown config fields {bad}")
    return Config(**given)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parity"); p.add_argument("--kind", choices=["e5", "bge"], required=True)
    sub.add_parser("baseline")
    p = sub.add_parser("table"); p.add_argument("--kind", default=""); p.add_argument("--stage", default="")
    p = sub.add_parser("pilot"); p.add_argument("--stage", required=True)
    p.add_argument("--grid", choices=["e5_default"], help="named grid")
    p.add_argument("--cfg", action="append", default=[], help="JSON Config fields; repeatable")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("run"); p.add_argument("--method", required=True); p.add_argument("--cfg", required=True)
    p.add_argument("--selection-note", default="")
    p = sub.add_parser("job"); p.add_argument("--cfg", required=True); p.add_argument("--role", required=True, action="append")
    p.add_argument("--eval", action="store_true"); p.add_argument("--max-steps", type=int, default=0)
    p = sub.add_parser("report"); p.add_argument("--method", required=True); p.add_argument("--lstar", default="L_star")
    p.add_argument("--skip-fusion", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "parity":
        cmd_parity(args.kind)
    elif args.cmd == "baseline":
        cmd_baseline()
    elif args.cmd == "table":
        cmd_table(args.kind, args.stage)
    elif args.cmd == "pilot":
        configs = e5_pilot_grid() if args.grid == "e5_default" else []
        configs += [Config(**json.loads(c)) for c in args.cfg]
        run_pilot(configs, args.stage, force=args.force)
    elif args.cmd == "run":
        cfg = Config(**json.loads(args.cfg))
        run_method(args.method, cfg, {"note": args.selection_note, "search_log": str(SEARCH_LOG)})
    elif args.cmd == "job":
        cmd_job(Config(**json.loads(args.cfg)), args.role, args.eval, args.max_steps)
    elif args.cmd == "report":
        cmd_report(args.method, args.lstar, args.skip_fusion)


if __name__ == "__main__":
    main()
