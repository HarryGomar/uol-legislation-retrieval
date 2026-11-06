"""The agents' embedding tool through the query server on the remote worker (``gpu_jobs/query_server_job``, PLAN
addendum 13). ``patch_endpoint`` swaps ``qwen_endpoint.Client`` and ``encode_items`` as ``qwen_local.patch_endpoint``
does, so ``e6_agent`` is unchanged. Before the first search, 50 development queries are encoded and held to the
fidelity gate against their stored vectors (Gate E0: mean cosine >= .99, minimum >= .97).

    py -3.12 -m uol.final.query_server post --go     # send the job
    py -3.12 -m uol.final.query_server stop --go     # the agents are done
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

FOLDER = "agent_queries"
JOB = os.environ.get("UOL_QUERY_SERVER_JOB", "agent_query_server_j3")   # a later posting needs a new name (gpu_remote.push)
IDLE_S, TOTAL_S = 90 * 60, 6 * 3600
WAIT_S = 8 * 3600
CONTROLS = 50
REWRITE_CHUNK = 256
_STATE = {"checked": False}


def exchange() -> Path:
    return Path(os.environ.get("UOL_GPU_EXCHANGE", r"exchange\UOL_GPU_EXCHANGE"))


def ask(texts: list[str], folder: Path | None = None, wait_s: float = WAIT_S) -> np.ndarray:
    folder = folder or exchange() / FOLDER
    rid = f"{os.getpid()}_{time.time_ns()}"
    tmp = folder / f"tmp_{rid}.json"
    tmp.write_text(json.dumps({"texts": texts}, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, folder / f"req_{rid}.json")
    rep, t0 = folder / f"rep_{rid}.npy", time.time()
    while time.time() - t0 < wait_s:
        if rep.exists():
            try:
                v = np.load(rep)
            except (OSError, ValueError, EOFError):        # still syncing
                time.sleep(2)
                continue
            if len(v) != len(texts):
                raise RuntimeError(f"{rep.name}: {len(v)} vectors for {len(texts)} queries")
            return v.astype(np.float32)
        time.sleep(2)
    raise TimeoutError(f"no answer to {rid} in {wait_s} s")


def fidelity() -> dict:
    from uol.embedding_push import common as ep, qwen_endpoint as qe
    d = ep.release()
    dev = [int(r) for r in ep.dev_rows()]
    rows = dev[:: max(1, len(dev) // CONTROLS)][:CONTROLS]
    v = ask([qe.query_input("c", d.queries[r].text) for r in rows])
    stored = qe.query_vectors("c")[rows]
    cos = np.sum(qe.unit_rows(v) * qe.unit_rows(stored), axis=1)
    res = {"controls": len(rows), "cos_mean": float(cos.mean()), "cos_min": float(cos.min())}
    print(f"query server fidelity {res}", flush=True)
    if res["cos_mean"] < .99 or res["cos_min"] < .97:
        raise RuntimeError(f"the query server fails the fidelity gate: {res}")
    return res


def patch_endpoint() -> None:
    from uol.embedding_push import qwen_endpoint as qe
    from uol.final import context

    class _NoClient:
        def __init__(self, *a, **k):
            pass

    def encode_items(client, store, items, **kw):
        if not _STATE["checked"]:
            _STATE["fidelity"] = fidelity()
            _STATE["checked"] = True
        uniq = list(dict((k, t) for k, t, _ in items if k not in store).items())
        if uniq:
            store.add([k for k, _ in uniq], ask([t for _, t in uniq]))
        return {"encoded_remotely": len(uniq)}

    qe.Client, qe.encode_items = _NoClient, encode_items
    context.active().patches["uol.embedding_push.qwen_endpoint.encode_items"] = "Qwen3-8B int8 query server (uol.final.query_server)"


def rewrite(kind: str) -> int:
    """The rewritten queries (``gpu_prep.qwen_texts``) through the server, stored as a query-only vector set of the 8B."""
    from uol.final import audit_d181_dev as D, context, footing as FT, gpu_prep
    D.activate(FT.STORE)
    texts = gpu_prep.qwen_texts(kind)
    rows = sorted(int(r) for r in texts)
    fidelity()
    vecs = [ask([texts[r] for r in rows[s:s + REWRITE_CHUNK]]) for s in range(0, len(rows), REWRITE_CHUNK)]
    folder = context.active().path("new_vectors") / kind
    folder.mkdir(parents=True, exist_ok=True)
    variant = "a" if kind == "hyde" else "c"
    np.save(folder / f"query_vectors_{variant}.npy", np.concatenate(vecs))
    (folder / "query_rows.json").write_text(json.dumps({"release_rows": rows}), encoding="utf-8")
    (folder / "articles_from.json").write_text(json.dumps({"key": "qwen8b"}), encoding="utf-8")
    print(f"{kind}: {len(rows)} query vectors in {folder}", flush=True)
    return 0


def post() -> int:
    os.environ.setdefault("UOL_GPU_TRANSPORT", "drive")
    os.environ.setdefault("UOL_GPU_EXCHANGE", str(exchange()))
    from uol.final import footing as FT
    from uol.round3 import gpu_remote as G
    from uol.scale_curve import legal
    here = Path(__file__).resolve().parent / "gpu_jobs"
    jobs = Path(G.__file__).parent / "gpu_jobs"
    d = FT.A.WORK / "agents" / "server"
    d.mkdir(parents=True, exist_ok=True)
    (d / "task.json").write_text(json.dumps({"exchange": str(exchange()), "folder": FOLDER, "idle_s": IDLE_S, "total_s": TOTAL_S,
                                             "model": {**legal.GATE["qwen3_8b_gate"], "key": "qwen3_8b_server"}}, indent=1), encoding="utf-8")
    (exchange() / FOLDER).mkdir(exist_ok=True)
    job = G.pack(JOB, here / "query_server_job.py", {"task.json": d / "task.json"},
                 helpers=[jobs / "legal_encode.py", jobs / "rerank_bench.py", Path(__file__).resolve().parents[1] / "scale_curve" / "store.py"], expected_outputs=["served.json"])
    G.push(job)
    print(f"posted {JOB}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", choices=["post", "stop", "query2doc", "hyde"])
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)
    if not a.go:
        print(f"[dry] {a.run} {JOB}")
        return 0
    if a.run == "post":
        return post()
    if a.run in ("query2doc", "hyde"):
        return rewrite(a.run)
    (exchange() / FOLDER / "STOP").write_text("agents done\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
