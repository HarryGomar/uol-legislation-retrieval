"""Final benchmark harness (research/programs/final_benchmark_20261021/PLAN_DRAFT.md, D66).

One release, one set of populations, one fold scheme, one harness, one metric set, one cost record and one results
store. The package wraps the existing scorers and evaluators (``uol.embedding_push``, ``uol.round3``,
``uol.scale_curve``, ``uol.gap_tests``, ``uol.facts_to_law``); it does not copy them.

Development rows only. The held-out test set (the SPLIT.json lockbox: tesis whose component starts at record
2030658 or later, published from 27 Jun 2025) is refused by ``uol.final.guard`` wherever rows meet scores or labels.

    py -3.12 -m uol.final {populations,run,report,g1,bundle,plan,overnight,systems}
"""
STORE_NAME = "final_20261021"
