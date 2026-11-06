"""Remote job (R2): fine-tune a cross-encoder reranker on training-fold citation groups, then score held-out pairs.

Inputs:
* ``train.jsonl``: {"q": query text, "docs": [positive, neg_1..neg_15]} (positive first; training folds only);
* ``queries.jsonl`` / ``docs.jsonl`` / ``pairs.npy``: scoring pairs (held-out fold, no labels);
* ``task.json``: {"model", "lora_r", "lr", "epochs", "group_size", "batch_groups", "max_query_tokens", "max_doc_tokens",
  "seed", "loss" ("lce"|"bce"), "instruction", "key", "shuffle_labels" (bool), "blank_query" (bool)}.
Backbones: ``qwen3_rr_0.6b`` / ``qwen3_rr_4b`` (causal LM yes/no head: score = logit(yes) - logit(no), LoRA via peft) or
``bge_rr_v2_m3`` (sequence classification, full fine-tuning). bf16 autocast, gradient checkpointing. Loss: listwise
softmax cross-entropy over each group (LCE) or pointwise BCE. The trained adapter stays in ``UOL_WORKER_SCRATCH/r2/<key>``.
Output: ``scores.npy`` (float32, one per scoring pair; MaxP over document chunks of max_doc_tokens, ≤ 4) and ``train.json``.

    python r2_train.py --inputs inputs --out outputs
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rerank_bench as RB  # noqa: E402

PEFT_PINS = ["peft==0.21.0"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    scratch = Path(os.environ.get("UOL_WORKER_SCRATCH", a.out.parent / "scratch"))
    os.environ.setdefault("HF_HOME", str(scratch / "hf"))
    RB.deps(scratch, RB.PINS)
    task = json.loads((a.inputs / "task.json").read_text(encoding="utf-8"))
    kind = RB.MODELS[task["model"]][2]
    if kind == "qwen":
        RB.deps(scratch, PEFT_PINS + ["accelerate==1.15.0"])
    import torch
    from huggingface_hub import snapshot_download
    RB.no_symlinks()
    torch.manual_seed(task["seed"])
    random.seed(task["seed"])
    np.random.seed(task["seed"])
    repo, rev, _ = RB.MODELS[task["model"]]
    path = snapshot_download(repo, revision=rev, cache_dir=str(scratch / "hf"),
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"])
    a.out.mkdir(parents=True, exist_ok=True)
    work = scratch / "r2" / task["key"]
    work.mkdir(parents=True, exist_ok=True)

    if kind == "qwen":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        tok = AutoTokenizer.from_pretrained(path, padding_side="left")
        base = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda()
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
        model = get_peft_model(base, LoraConfig(r=task["lora_r"], lora_alpha=2 * task["lora_r"], lora_dropout=0.05,
                                                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                                "up_proj", "down_proj"], task_type="CAUSAL_LM"))
        yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
        pre = tok(RB.PREFIX, add_special_tokens=False)["input_ids"]
        suf = tok(RB.SUFFIX, add_special_tokens=False)["input_ids"]

        def forward(qs, ds):
            bodies = [f"<Instruct>: {task['instruction']}\n<Query>: {q}\n<Document>: {d}" for q, d in zip(qs, ds)]
            enc = tok(bodies, add_special_tokens=False)["input_ids"]
            seqs = [pre + e + suf for e in enc]
            n = max(map(len, seqs))
            ids = torch.tensor([[tok.pad_token_id] * (n - len(s)) + s for s in seqs]).cuda()
            att = torch.tensor([[0] * (n - len(s)) + [1] * len(s) for s in seqs]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(input_ids=ids, attention_mask=att, logits_to_keep=1).logits[:, -1, :].float()
            return lg[:, yes] - lg[:, no]
    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32).cuda()
        model.gradient_checkpointing_enable()

        def forward(qs, ds):
            enc = tok(list(qs), list(ds), padding=True, truncation=True, max_length=task["max_query_tokens"] + task["max_doc_tokens"],
                      return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model(**enc).logits.view(-1).float()

    def cut(t, n):
        return RB.truncate(tok, t, n)

    groups = [json.loads(line) for line in (a.inputs / "train.jsonl").read_text(encoding="utf-8").splitlines() if line]
    G = task["group_size"]
    groups = [g for g in groups if len(g["docs"]) >= 2]
    rng = random.Random(task["seed"])
    if task.get("shuffle_labels"):            # control: the "positive" is a random member of the group
        for g in groups:
            j = rng.randrange(len(g["docs"]))
            g["docs"][0], g["docs"][j] = g["docs"][j], g["docs"][0]
    prepared = [([("" if task.get("blank_query") else cut(g["q"], task["max_query_tokens"]))] * min(G, len(g["docs"])),
                 [cut(d, task["max_doc_tokens"]) for d in g["docs"][:G]]) for g in groups]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=task["lr"], weight_decay=0.01)
    steps = task["epochs"] * int(np.ceil(len(prepared) / task["batch_groups"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps // 10)) * max(0.0, 1 - s / steps))
    model.train()
    t0, step, losses = time.perf_counter(), 0, []
    for ep in range(task["epochs"]):
        order = list(range(len(prepared)))
        rng.shuffle(order)
        for s in range(0, len(order), task["batch_groups"]):
            loss = 0.0
            for gi in order[s:s + task["batch_groups"]]:
                qs, ds = prepared[gi]
                sc = forward(qs, ds)
                if task["loss"] == "lce":
                    l = torch.nn.functional.cross_entropy(sc[None, :], torch.zeros(1, dtype=torch.long, device=sc.device))
                else:
                    y = torch.zeros_like(sc)
                    y[0] = 1.0
                    l = torch.nn.functional.binary_cross_entropy_with_logits(sc, y)
                (l / task["batch_groups"]).backward()
                loss += float(l) / task["batch_groups"]
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss)
            step += 1
            if step % 25 == 0:
                RB.log(f"epoch {ep} step {step}/{steps} loss {np.mean(losses[-25:]):.4f}")
    train_s = time.perf_counter() - t0
    try:
        model.save_pretrained(str(work / "adapter"))
    except Exception as exc:                  # noqa: BLE001 (saving is a convenience; scores are the deliverable)
        RB.log(f"save failed: {exc}")
    model.eval()
    queries = {int(r["qid"]): r["text"] for r in map(json.loads, (a.inputs / "queries.jsonl").read_text(encoding="utf-8").splitlines()) if r}
    docs = {int(r["did"]): r["text"] for r in map(json.loads, (a.inputs / "docs.jsonl").read_text(encoding="utf-8").splitlines()) if r}
    pairs = np.load(a.inputs / "pairs.npy")
    qtxt = {q: ("" if task.get("blank_query") else cut(t, task["max_query_tokens"])) for q, t in queries.items()}
    qtxt[-1] = ""
    chunks = {}
    for d, t in docs.items():
        ids = tok(t, add_special_tokens=False)["input_ids"]
        n = task["max_doc_tokens"]
        chunks[d] = [tok.decode(ids[i:i + n]) for i in range(0, max(len(ids), 1), n)][:4]
    flat = [(i, c) for i, (q, d) in enumerate(pairs) for c in range(len(chunks[int(d)]))]
    best = np.full(len(pairs), -np.inf, dtype=np.float32)
    t1 = time.perf_counter()
    with torch.no_grad():
        for s in range(0, len(flat), 32):
            b = flat[s:s + 32]
            sc = forward([qtxt[int(pairs[i][0])] for i, _ in b], [chunks[int(pairs[i][1])][c] for i, c in b]).cpu().numpy()
            for (i, _), v in zip(b, sc):
                best[i] = max(best[i], v)
    np.save(a.out / "scores.npy", best)
    (a.out / "train.json").write_text(json.dumps({"task": task, "groups": len(prepared), "steps": steps,
                                                  "loss_first25": float(np.mean(losses[:25])) if losses else None,
                                                  "loss_last25": float(np.mean(losses[-25:])) if losses else None,
                                                  "train_seconds": round(train_s, 1),
                                                  "score_seconds": round(time.perf_counter() - t1, 1),
                                                  "pairs": int(len(pairs))}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
