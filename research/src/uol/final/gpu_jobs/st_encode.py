"""GPU job: ``round3/gpu_jobs/legal_encode.py`` for models that need remote code or task-specific encode arguments
(gte-multilingual-base, jina-embeddings-v3), same inputs and outputs.

``task.json`` adds ``trust_remote_code`` (bool) and ``encode_kwargs_by_prompt`` ({prompt text or "": {kwarg: value}},
e.g. jina's ``task="retrieval.query"`` / ``"retrieval.passage"`` with ``prompt_name``). Everything else (windows, the
resumable store, budget, run.json) is legal_encode's code, unchanged; only its loader and the model's ``encode`` are
wrapped. Remote-code models run on the venv's own transformers 4.x stack (``job_stack``), other models on
legal_encode's pinned overlay; run.json records the stack (``runtime.stack``, ``runtime.versions``).

    python st_encode.py --inputs inputs --out outputs
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for cand in (HERE.parents[1] / "round3" / "gpu_jobs",):
    if cand.exists():
        sys.path.insert(0, str(cand))
import legal_encode as LE  # noqa: E402

# Remote code that is imported by a trust_remote_code model but not shipped in its own repository needs packages the
# run venv does not have (jina's xlm-roberta-flash-implementation imports einops); the only private install of a
# remote-code job (job_stack).
REMOTE_CODE_PINS = ["einops==0.8.1"]


def remote_code_repos(model_dir: Path) -> list[str]:
    """Hub repositories that hold the remote code of a snapshot: ``auto_map`` entries of the form
    ``"<org>/<repo>--<module>.<Class>"`` in its config files (gte-multilingual-base -> Alibaba-NLP/new-impl,
    jina-embeddings-v3 -> jinaai/xlm-roberta-flash-implementation). Code inside the snapshot itself has no ``--``."""
    repos: list[str] = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "auto_map" and isinstance(v, dict):
                    for ref in v.values():
                        for r in (ref if isinstance(ref, (list, tuple)) else [ref]):
                            if isinstance(r, str) and "--" in r:
                                repo = r.split("--", 1)[0]
                                if repo not in repos:
                                    repos.append(repo)
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    for p in sorted(Path(model_dir).rglob("*.json")):
        if p.stat().st_size > 2 << 20:                    # tokenizer.json and other large data files hold no auto_map
            continue
        try:
            walk(json.loads(p.read_text(encoding="utf-8")))
        except (ValueError, UnicodeDecodeError):
            continue
    return repos


def fetch_remote_code(model_dir: Path, download=None) -> dict[str, str]:
    """Download the remote-code repositories into the hub cache that transformers reads (``local_files_only=True``
    then finds them); returns {repo: snapshot commit} for run provenance (21 Oct 2026: gte_multilingual, jina_v3)."""
    if download is None:
        from huggingface_hub import snapshot_download as download
    out = {}
    for repo in remote_code_repos(model_dir):
        path = download(repo, allow_patterns=["*.py", "*.json", "*.txt", "*.md"])
        if remove_broken_entries(Path(path)):                 # pointer files left by an earlier attempt: fetch again
            path = download(repo, allow_patterns=["*.py", "*.json", "*.txt", "*.md"])
            if broken_entries(Path(path)):
                raise RuntimeError(f"{repo}: snapshot {path} still holds unreadable entries after a repair")
        out[repo] = Path(path).name
        SNAPSHOTS[repo] = Path(path)
    return out


SNAPSHOTS: dict[str, Path] = {}                                  # remote-code repo -> its hub snapshot (this process)


def sync_dynamic_modules(snapshots: dict[str, Path] | None = None, modules_root: Path | None = None) -> list[str]:
    """Complete transformers' copy of the remote code (``<HF_MODULES_CACHE>/transformers_modules/<org>/<repo>/<commit>``)
    from the hub snapshot. transformers copies a module's relative imports only when the module itself is first
    copied, so an attempt that stopped half-way (e.g. on a missing package) leaves a folder that later loads never
    complete ("No such file ... embedding.py"; seen in the CPU replica, 22 Oct 2026). Only missing ``.py`` files are
    copied, byte for byte as transformers itself would; an existing file is never touched. Returns the files added."""
    snapshots = SNAPSHOTS if snapshots is None else snapshots
    if modules_root is None:
        from transformers.utils import hub as tu
        modules_root = Path(tu.HF_MODULES_CACHE)
    try:
        from transformers.dynamic_module_utils import _sanitize_module_name as sanitize
    except ImportError:                                           # pragma: no cover - older layouts
        def sanitize(name: str) -> str:
            return name.replace(".", "_dot_").replace("-", "_hyphen_")
    import shutil
    added = []
    for repo, snap in snapshots.items():
        target = Path(modules_root) / "transformers_modules"
        for part in repo.split("/"):
            target = target / sanitize(part)
        target = target / Path(snap).name
        if not target.exists():
            continue                                              # never created: transformers builds it whole
        for f in sorted(Path(snap).rglob("*.py")):
            dst = target / f.relative_to(snap)
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, dst)
                added.append(str(dst))
    if added:
        LE.RUNTIME.setdefault("dynamic_modules_completed", []).extend(added)
    return added


def copy_only_hub_cache(module=None) -> bool:
    """Make huggingface_hub store real copies, never symlinks (22 Oct 2026, gpu_smoke__jina_v3 on hub 0.36.2:
    ``WinError 1314`` from ``file_download._create_symlink``). ``are_symlinks_supported`` marks a cache folder as
    supporting symlinks *before* it has tested it, so the parallel download threads of ``snapshot_download`` can see
    True and try ``os.symlink`` on a Windows account without that privilege. With the probe answering False every
    file is copied into ``snapshots/<commit>/`` (hub's documented Windows fallback), which transformers' offline
    lookup of the remote code reads like any snapshot. Called before the first download."""
    if module is None:
        from huggingface_hub import file_download as module
    if not hasattr(module, "are_symlinks_supported"):
        return False
    module.are_symlinks_supported = lambda *a, **k: False
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    LE.RUNTIME["hub_cache"] = "copies (are_symlinks_supported -> False)"
    return True


def broken_entries(snapshot: Path) -> list[Path]:
    """Entries of a snapshot folder that cannot be read: links whose blob is missing (a failed symlink attempt leaves
    no file at all, which the next ``snapshot_download`` fills in by itself)."""
    return [p for p in sorted(Path(snapshot).rglob("*")) if os.path.islink(p) and not os.path.exists(p)]


def remove_broken_entries(snapshot: Path) -> int:
    bad = broken_entries(snapshot)
    for p in bad:
        os.remove(p)
    if bad:
        LE.RUNTIME.setdefault("hub_cache_repaired", []).extend(str(p) for p in bad)
    return len(bad)


def main() -> int:
    args = sys.argv[1:]
    inputs = Path(args[args.index("--inputs") + 1])
    task = json.loads((inputs / "task.json").read_text(encoding="utf-8"))
    by_prompt = task.get("encode_kwargs_by_prompt") or {}
    out = Path(args[args.index("--out") + 1])
    scratch = Path(os.environ.get("UOL_WORKER_SCRATCH", out.parent / "scratch"))
    os.environ.setdefault("HF_HOME", str(scratch / "hf"))      # one hub cache for the model and its remote code
    pins, stack = job_stack(task)
    if task.get("trust_remote_code"):
        copy_only_hub_cache()                                  # the venv's hub 0.36: before any download
    orig_load = LE.load

    def load(task_, scratch_, device=None):
        if not task_.get("trust_remote_code"):
            return orig_load(task_, scratch_, device)
        import torch
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer
        path = snapshot_download(task_["model"], revision=task_["revision"], cache_dir=str(scratch_ / "hf"))
        code = fetch_remote_code(Path(path))
        sync_dynamic_modules()                                   # complete a half-copied module folder of an earlier attempt
        LE.RUNTIME["remote_code"] = code
        LE.RB.log(f"remote code repositories: {code or 'none'}")
        model = SentenceTransformer(path, device="cpu" if device == "cpu" else "cuda", trust_remote_code=True,
                                    model_kwargs=remote_model_kwargs(torch), local_files_only=True)
        model.max_seq_length = int(task_["max_tokens"])
        return model, path

    LE.load = load
    if by_prompt:
        if pins:
            LE.RB.deps(scratch, pins)                         # the job's private installs, before the import
        from sentence_transformers import SentenceTransformer
        orig_encode = SentenceTransformer.encode

        def encode(self, sentences, *a, prompt=None, **kw):
            extra = by_prompt.get(prompt or "", {})
            if extra:
                kw.update(extra)
                prompt = None
            return orig_encode(self, sentences, *a, prompt=prompt, **kw)
        SentenceTransformer.encode = encode
    return LE.main(pins=pins, stack=stack)


def job_stack(task: dict) -> tuple[list[str], str]:
    """Remote-code models (gte-multilingual-base, jina-embeddings-v3) were written for transformers 4.x: under the
    transformers 5.17 overlay jina fails at load (``all_tied_weights_keys``) and gte hits a device-side assert in its
    embeddings (22 Oct 2026). They run on the run venv's own pinned stack (requirements.lock: transformers 4.57.6,
    sentence-transformers 5.1.2, huggingface-hub 0.36.2, peft 0.17.1) plus only what it lacks (einops). Other models
    keep legal_encode's overlay."""
    if task.get("trust_remote_code"):
        return list(REMOTE_CODE_PINS), ("venv stack (requirements.lock: transformers 4.57.6, sentence-transformers "
                                        "5.1.2, huggingface-hub 0.36.2) + " + " ".join(REMOTE_CODE_PINS))
    return list(LE.ST_PINS), "pinned overlay " + " ".join(LE.ST_PINS)


def remote_model_kwargs(torch) -> dict:
    """fp32 weights, under the keyword this transformers takes (``dtype`` from 4.56, ``torch_dtype`` before)."""
    try:
        import transformers
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    except Exception:                                          # pragma: no cover
        return {"dtype": torch.float32}
    return {"dtype": torch.float32} if (major, minor) >= (4, 56) else {"torch_dtype": torch.float32}


if __name__ == "__main__":
    raise SystemExit(main())
