"""Matching the model's answers to the catalogue, and the merge and rerank around it (with a scripted model)."""
from types import SimpleNamespace

import numpy as np

from uol import llm

BENCH = SimpleNamespace(
    article_ids=["https://x/pdf/CPEUM.pdf#articulo=14", "https://x/pdf/CPEUM.pdf#articulo=16",
                 "https://x/pdf/LFT.pdf#articulo=153-F-Bis", "https://x/pdf/LFT.pdf#articulo=1"],
    article_texts=["CONSTITUCIÓN Política de los Estados Unidos Mexicanos\nArtículo 14.- Nadie podrá ser privado...",
                   "CONSTITUCIÓN Política de los Estados Unidos Mexicanos\nArtículo 16.- Nadie puede ser molestado...",
                   "LEY Federal del Trabajo\nArtículo 153-F Bis.- " + "x" * 700,
                   "LEY Federal del Trabajo\nArtículo 1o.- La presente Ley es de observancia general..."],
    generic=np.array([], dtype=int))


def test_article_numbers_take_the_catalogue_form():
    assert llm.article_key("Artículo 1o.") == "1"
    assert llm.article_key("31, fracción IV") == "31"
    assert llm.article_key("153-F Bis") == "153-f-bis"
    assert llm.article_key("123 apartado B") == "123"


def test_laws_match_by_alias_code_or_title():
    cat = llm.Catalogue(BENCH)
    assert cat.article("Constitución Política de los Estados Unidos Mexicanos", "16") == 1
    assert cat.article("LFT", "153-F Bis") == 2
    assert cat.article("Ley Federal del Trabajo", "Artículo 1o.") == 3
    assert cat.article("Ley de Navegación", "1") is None


def test_reranker_sees_a_short_excerpt():
    line = llm.Catalogue(BENCH).line(BENCH, 2)
    assert line.startswith("LEY Federal del Trabajo › Art. 153-F-Bis: ") and line.endswith("…")


def test_rrf_ties_keep_the_model_list_first():
    assert llm.rrf([[7, 8], [8, 7]]) == [7, 8]
    assert llm.rrf([[1], [2, 1]]) == [1, 2]


class Scripted:
    """Reverses every window it is shown."""

    def ask(self, prompt, schema, sample=0):
        n = sum(line.split(".")[0].isdigit() for line in prompt.split("Candidatos:\n")[1].splitlines())
        return {"results": [{"id": "w", "orden": list(range(n, 0, -1))}]}


def test_rerank_windows_orders_two_windows_from_the_bottom_up(monkeypatch):
    monkeypatch.setattr(llm.Catalogue, "line", lambda self, bench, i: f"article {i}")
    merged = list(range(40))
    out = llm.rerank_windows(Scripted(), BENCH, llm.Catalogue(BENCH), "hechos", merged)
    lower = list(range(29, 9, -1))                           # positions 11-30 reversed
    upper = (list(range(10)) + lower[:10])[::-1]             # positions 1-10 with the best ten below, reversed
    reordered = upper + lower[10:]
    assert out[30:] == merged[30:] and sorted(out[:30]) == merged[:30]
    assert out[:30] == llm.fuse(merged[:30], reordered)


def test_fuse_with_no_model_order_keeps_the_merged_order():
    assert llm.fuse([3, 1, 2], []) == [3, 1, 2]


def test_rerank_orders_all_thirty_in_one_request(monkeypatch):
    monkeypatch.setattr(llm.Catalogue, "line", lambda self, bench, i: f"article {i}")
    merged = list(range(40))
    out = llm.rerank(Scripted(), BENCH, llm.Catalogue(BENCH), "hechos", merged)
    assert out[30:] == merged[30:]
    assert out[:30] == llm.fuse(merged[:30], merged[:30][::-1])


class Incomplete:
    """Leaves one candidate out on the first attempt and answers in full on the second."""

    def __init__(self):
        self.calls = 0

    def ask(self, prompt, schema, sample=0):
        self.calls += 1
        n = sum(line.split(".")[0].isdigit() for line in prompt.split("Candidatos:\n")[1].splitlines())
        order = list(range(n, 0, -1))
        return {"results": [{"id": "w", "orden": order[:-1] if sample == 0 else order}]}


def test_rerank_rejects_an_incomplete_order_and_asks_again(monkeypatch):
    monkeypatch.setattr(llm.Catalogue, "line", lambda self, bench, i: f"article {i}")
    model = Incomplete()
    out = llm.rerank(model, BENCH, llm.Catalogue(BENCH), "hechos", list(range(40)))
    assert model.calls == 2 and out[:30] == llm.fuse(list(range(30)), list(range(29, -1, -1)))
