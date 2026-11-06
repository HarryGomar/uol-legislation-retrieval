# Research code, as it ran

The code that produced every experiment of the thesis, copied from the working tree without changes except where
`EXPORT.json` says how (server addresses replaced by settings, build-machine folders by neutral names). It is kept
for reproduction and inspection; the maintained implementation of the final system is the `uol` package at the root
of this repository (see its Reproduction section for how closely it matches these runs).

This folder mirrors the thesis's working tree, so the code finds its own root from each file's location:

| Path | What it holds |
|---|---|
| `src/uol/final/` | the experiment commands: ranker fits and reruns (`ranker_tests`, `rank_rerun`, `lean_rerun`, `footing`), language-model runs (`llm_agents`, `why_models`, `prompt_check`, `overnight`), the live path and timing (`whole_run`, `rank_timing`), the test set (`test_final`, `whole_test`), and the first full run (`__main__`: `plan`, `overnight`, `run`, `merge-store`) |
| `src/uol/embedding_push/`, `src/uol/facts_to_law/`, `src/uol/round3/` | the first-stage sources, the learned ranker of the first run, the dense encoders, the language-model transport and its prompts |
| `src/uol/thesis_figures/` | the Chapter 5 and 6 figures |
| `scripts/` | the generators of the thesis's numbers and tables (`final_results.py` writes one record of every result) |
| `research/programs/` | the read-outs whose numbers the thesis prints |
| `tests/test_whole_test_guard.py` | the guards of the live path: the test set stays closed, each model step names its model, each live run keeps its own model cache |

## Running it

Python 3.12 on Windows 11 was the tested platform. Use a separate environment from the `uol` package, since both
are named `uol`:

```bash
cd research
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
PYTHONPATH=src .venv/Scripts/python -m uol.final.rank_rerun ranker --id rk_noenc --go
```

The commands read the thesis's run data from the same relative places as in the working tree:
`data/releases/uol_c01_statutes_20261004_v11_stratified/` (the benchmark), `data/derived/` (the stored vectors) and
`experiments/runs/` (the stored source scores and records). Experiments that call a language model need a Codex
sign-in; GPU experiments need a 16 GB graphics card. `../experiments/README.md` gives the command of each experiment.
