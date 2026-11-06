"""Remote job (Phase 0 step 9): load each reranker on the worker, score fixed dev pairs, measure throughput.

Inputs: ``pairs.jsonl`` ({"pid", "query", "doc"}: masked dev Hechos and public article text; no labels) and
``task.json`` ({"models": [...], "instruction": str, "max_query_tokens", "max_doc_tokens", "batch_sizes"}).
Models are downloaded on the worker into ``UOL_WORKER_SCRATCH/hf`` at pinned revisions. Pinned packages go to a
private target dir, never the shared venv.

Qwen3-Reranker scoring follows the model card: the yes/no chat template, score = P("yes") from the last-token logits
over {"yes", "no"}, left padding. bge-reranker-v2-m3: sequence-classification logit. Outputs per model:
``scores_<name>.json`` (pid -> score) and ``bench.json`` (load time, peak memory, pairs/s per batch size, errors).

    python rerank_bench.py --inputs inputs --out outputs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

PINS = ["transformers==5.17.0", "tokenizers==0.23.2", "huggingface_hub==1.32.0"]
BNB_PINS = ["bitsandbytes==0.50.2", "accelerate==1.15.0"]
# Runtime packages of the overlay above that ``--no-deps`` leaves out (hub 1.32 / transformers 5.17 metadata) and the
# GPU jobs' PEFT; the run computer added exactly these by hand on 21 Oct 2026. Installed into the same overlay folder
# (its name still hashes the pins only), so both run computers get the same overlay.
OVERLAY_RUNTIME = ["httpx==0.28.1", "httpcore==1.0.9", "anyio==4.15.1", "hf-xet==1.7.0", "typer==0.27.3",
                   "shellingham==1.5.4", "rich==15.0.0", "markdown-it-py==4.2.0", "mdurl==0.1.2", "click==8.5.0",
                   "safetensors==0.8.0", "peft==0.21.0"]
MODELS = {
    "qwen3_rr_0.6b": ("Qwen/Qwen3-Reranker-0.6B", "e61197ed45024b0ed8a2d74b80b4d909f1255473", "qwen"),
    "qwen3_rr_4b": ("Qwen/Qwen3-Reranker-4B", "22e683669bc0f0bd69640a1354a6d0aebcfeede5", "qwen"),
    "qwen3_rr_8b": ("Qwen/Qwen3-Reranker-8B", "77d193c791ed757ca307ee72715aa132723da912", "qwen"),
    "bge_rr_v2_m3": ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", "bge"),
}
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
          "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def no_symlinks() -> None:
    """The model cache holds real files: a worker's Windows account may not be allowed to create symbolic links."""
    from huggingface_hub import file_download
    file_download.are_symlinks_supported = lambda *_a, **_k: False


def deps(scratch: Path, pins: list[str], wait_s: float = 1800.0) -> None:
    """Install the pins once per machine. Jobs that start together (RUN-B, 23 Oct: two GPU smokes) take a lock file;
    pip writes into a private folder that is renamed into place only when complete."""
    target = scratch / ("pydeps_" + hashlib.sha1(" ".join(pins).encode()).hexdigest()[:10])
    scratch.mkdir(parents=True, exist_ok=True)       # pip --target made it before the lock (RUN-B smokes, 23 Oct)
    lock = target.with_name(target.name + ".lock")
    t0 = time.time()
    while not (target / "OK").exists():
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() - lock.stat().st_mtime > wait_s:        # a holder that died
                lock.unlink(missing_ok=True)
            elif time.time() - t0 > wait_s:
                raise RuntimeError(f"waited {wait_s:.0f} s for {lock}")
            time.sleep(2)
            continue
        os.close(fd)
        try:
            if (target / "OK").exists():
                break
            log(f"installing {pins} into {target}")
            extra = OVERLAY_RUNTIME if any(p.startswith("huggingface_hub==1.") for p in pins) else []
            tmp = target.with_name(f"{target.name}.tmp{os.getpid()}")
            shutil.rmtree(tmp, ignore_errors=True)
            r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(tmp), *pins, *extra],
                               capture_output=True, text=True)
            (scratch / f"pip_install_{target.name}.log").write_text(r.stdout + r.stderr, encoding="utf-8")
            if r.returncode != 0:
                shutil.rmtree(tmp, ignore_errors=True)
                raise RuntimeError(f"pip install {pins} failed")
            (tmp / "OK").write_text(json.dumps({"pins": pins, "runtime": extra, "utc": time.time()}))
            shutil.rmtree(target, ignore_errors=True)               # a partial folder from before this lock
            os.replace(tmp, target)
        finally:
            lock.unlink(missing_ok=True)
    sys.path.insert(0, str(target))


def truncate(tok, text: str, n: int) -> str:
    ids = tok(text, add_special_tokens=False)["input_ids"][:n]
    return tok.decode(ids)


class Qwen:
    def __init__(self, path: str, quant: str | None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path, padding_side="left")
        kw = {"dtype": torch.bfloat16}
        if quant:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = (BitsAndBytesConfig(load_in_8bit=True) if quant == "8bit" else
                                         BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                            bnb_4bit_compute_dtype=torch.bfloat16))
            kw["device_map"] = {"": 0}
        self.model = AutoModelForCausalLM.from_pretrained(path, **kw).eval()
        if not quant:
            self.model = self.model.cuda()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.pre = self.tok(PREFIX, add_special_tokens=False)["input_ids"]
        self.suf = self.tok(SUFFIX, add_special_tokens=False)["input_ids"]

    def score(self, instruction: str, pairs: list[tuple[str, str]]) -> list[float]:
        torch = self.torch
        bodies = [f"<Instruct>: {instruction}\n<Query>: {q}\n<Document>: {d}" for q, d in pairs]
        enc = self.tok(bodies, add_special_tokens=False, truncation=False)["input_ids"]
        seqs = [self.pre + e + self.suf for e in enc]
        n = max(map(len, seqs))
        pad = self.tok.pad_token_id
        ids = torch.tensor([[pad] * (n - len(s)) + s for s in seqs]).cuda()
        att = torch.tensor([[0] * (n - len(s)) + [1] * len(s) for s in seqs]).cuda()
        with torch.no_grad():
            logits = self.model(input_ids=ids, attention_mask=att, logits_to_keep=1).logits[:, -1, :]   # last position only
            two = torch.stack([logits[:, self.no], logits[:, self.yes]], 1).float()
            return torch.softmax(two, 1)[:, 1].cpu().tolist()


class Bge:
    def __init__(self, path: str, quant: str | None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.bfloat16).eval().cuda()

    def score(self, instruction: str, pairs: list[tuple[str, str]]) -> list[float]:
        torch = self.torch
        enc = self.tok([q for q, _ in pairs], [d for _, d in pairs], padding=True, truncation=True, max_length=1024,
                       return_tensors="pt").to("cuda")
        with torch.no_grad():
            return self.model(**enc).logits.view(-1).float().cpu().tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    scratch = Path(os.environ.get("UOL_WORKER_SCRATCH", a.out.parent / "scratch"))
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(scratch / "hf"))
    deps(scratch, PINS)
    task = json.loads((a.inputs / "task.json").read_text(encoding="utf-8"))
    pairs = [json.loads(line) for line in (a.inputs / "pairs.jsonl").read_text(encoding="utf-8").splitlines() if line]
    import torch
    from huggingface_hub import snapshot_download
    no_symlinks()
    a.out.mkdir(parents=True, exist_ok=True)
    bench = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0), "n_pairs": len(pairs),
             "instruction": task["instruction"], "models": {}}
    for spec in task["models"]:
        name, quant = spec["name"], spec.get("quant")
        repo, rev, kind = MODELS[name]
        key = name + (f"_{quant}" if quant else "")
        rec = {"repo": repo, "revision": rev, "quant": quant}
        try:
            if quant:
                deps(scratch, BNB_PINS)
            t = time.perf_counter()
            path = snapshot_download(repo, revision=rev, cache_dir=str(scratch / "hf"),
                                     allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"])
            rec["download_seconds"] = round(time.perf_counter() - t, 1)
            rec["bytes_on_disk"] = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            m = (Qwen if kind == "qwen" else Bge)(path, quant)
            rec["load_seconds"] = round(time.perf_counter() - t, 1)
            tok = m.tok
            prepared = [(truncate(tok, p["query"], task["max_query_tokens"]), truncate(tok, p["doc"], task["max_doc_tokens"]))
                        for p in pairs]
            m.score(task["instruction"], prepared[:4])                       # warm-up
            rec["throughput"] = {}
            scores = None
            for bs in task["batch_sizes"]:
                torch.cuda.synchronize()
                t = time.perf_counter()
                out = []
                for i in range(0, len(prepared), bs):
                    out += m.score(task["instruction"], prepared[i:i + bs])
                torch.cuda.synchronize()
                dt = time.perf_counter() - t
                rec["throughput"][str(bs)] = {"pairs_per_second": round(len(prepared) / dt, 2), "seconds": round(dt, 2)}
                scores = scores or out
                log(f"{key} bs={bs}: {len(prepared) / dt:.1f} pairs/s")
            rec["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
            (a.out / f"scores_{key}.json").write_text(json.dumps({p["pid"]: s for p, s in zip(pairs, scores)}),
                                                       encoding="utf-8")
            del m
        except Exception as exc:                                   # record and continue with the next model
            rec["error"] = f"{type(exc).__name__}: {exc}"[:800]
            rec["traceback"] = traceback.format_exc()[-2000:]
            log(f"{key} failed: {rec['error']}")
        torch.cuda.empty_cache()
        bench["models"][key] = rec
        (a.out / "bench.json").write_text(json.dumps(bench, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
