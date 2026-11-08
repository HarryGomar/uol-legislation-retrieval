"""Command line: ``python -m uol <command>``.

    fit      build the out-of-fold table of a ranker, fit its six group models and score the development lists
    answer   answer cases one at a time from their facts (the whole run), for a setup and a group
    score    nDCG@10 of a file of answers
    compare  paired difference between two files of answers, with its 95% interval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from uol import data, evaluate, llm, run, train
from uol.data import ALL, N_FOLDS
from uol.dense import Vectors
from uol.history import CitationMemory


def paths(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", default=None, help="benchmark folder (default $UOL_DATA or data/benchmark)")
    p.add_argument("--vectors", default=None, help="Qwen3 vectors folder (default: vectors/ of the benchmark)")
    p.add_argument("--models", default="models", help="fitted models folder")
    p.add_argument("--work", default="work", help="tables and answers")


def fit(a: argparse.Namespace) -> None:
    bench = data.load(a.data)
    vectors = Vectors.load(a.vectors or bench.root / "vectors") if a.setup == "ranker" else None
    table_path = Path(a.work) / f"table_{a.setup}.npz"
    if table_path.exists():
        table = train.Table.load(table_path)
    else:
        table = train.build_table(bench, a.setup, vectors)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table.save(table_path)
    memory, lists = CitationMemory(bench), {}
    for group in a.groups:
        model = train.fit(bench, table, a.setup, group, memory, threads=a.threads)
        path = run.model_path(a.models, a.setup, group)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(path))
        if group < N_FOLDS:
            lists |= train.fold_lists(bench, table, a.setup, group, model, memory)
    if lists:
        out = Path(a.work) / a.setup / "dev_lists.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for r, lst in sorted(lists.items()):
                f.write(json.dumps({"query_id": bench.cases[r]["query_id"], "ranked": lst}) + "\n")
        print(f"{a.setup}: development nDCG@10 {evaluate.score(bench, lists):.4f} over the folds fitted ({out})")


def answer(a: argparse.Namespace) -> None:
    bench = data.load(a.data)
    model = None
    if a.setup == "system":
        route = llm.OpenAIResponses if a.route == "openai" else llm.CodexCLI
        model = route(cache=Path(a.work) / "llm_cache.jsonl")
    rows = bench.evaluation_rows(a.population)
    if a.group < N_FOLDS:
        rows = rows[bench.fold[rows] == a.group]
    rows = rows[: a.cases] if a.cases else rows
    r = run.Run(bench, a.setup, a.group, a.models, a.vectors or bench.root / "vectors", a.encoder, model, a.windows)
    out = r.answer_all(rows, a.out or Path(a.work) / a.setup / f"{a.population}_group{a.group}.jsonl")
    if model is not None:
        print(f"model cost so far in this run: {model.cost()} USD")
    print(out)


def score(a: argparse.Namespace) -> None:
    bench = data.load(a.data)
    print(f"{evaluate.score(bench, run.read_answers(bench, a.answers, a.key), a.population):.4f}")


def compare(a: argparse.Namespace) -> None:
    bench = data.load(a.data)
    out = evaluate.compare(bench, run.read_answers(bench, a.first, a.key), run.read_answers(bench, a.second, a.key),
                           a.population)
    print(json.dumps(out, indent=1))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="uol", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fit")
    paths(p)
    p.add_argument("--setup", choices=["noenc", "ranker"], required=True)
    p.add_argument("--groups", type=lambda s: [int(x) for x in s.split(",")], default=list(range(ALL + 1)))
    p.add_argument("--threads", type=int, default=4)
    p.set_defaults(func=fit)
    p = sub.add_parser("answer")
    paths(p)
    p.add_argument("--setup", choices=["noenc", "ranker", "system"], required=True)
    p.add_argument("--population", default="DEV")
    p.add_argument("--group", type=int, default=0, help="fold 0-4 answers that fold's cases; 5 answers any case")
    p.add_argument("--encoder", choices=["stored", "cuda", "cpu"], default="stored")
    p.add_argument("--route", choices=["openai", "codex"], default="openai")
    p.add_argument("--cases", type=int, default=0, help="first n cases only")
    p.add_argument("--windows", action="store_true", help="the earlier two-window final step behind the thesis's test scores")
    p.add_argument("--out", default=None)
    p.set_defaults(func=answer)
    for name, func in (("score", score), ("compare", compare)):
        p = sub.add_parser(name)
        p.add_argument("--data", default=None)
        p.add_argument("--population", default="DEV")
        p.add_argument("--key", choices=["final", "ranked", "merged", "model_list"], default="final")
        if name == "score":
            p.add_argument("answers")
        else:
            p.add_argument("first")
            p.add_argument("second")
        p.set_defaults(func=func)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
