"""Thesis figure and table pack (staging).

Each module regenerates one figure (F1-F12) or one LaTeX table (T1-T7) from saved result files and writes it to
``research/thesis_direction_20261015/figures_staging/``. Nothing here trains, encodes, calls a model or reads lockbox
labels or rankings; lockbox values come from reports already written or, once it exists, from the phase-7 batch's
metric summaries (``common.phase7_value``).

    py -3.12 -m uol.thesis_figures.build_all      # every item, serially, then INDEX.md and manifest.json
    py -3.12 -m uol.thesis_figures.f04_ladder     # one item
"""
