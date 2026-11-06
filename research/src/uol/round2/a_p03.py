"""Round 2 / A: P03 analysis pack (b) complementarity, (c) vocabulary mismatch, (d) encoder scaling, (e) strata.

EXPLORATORY and descriptive. Dev numbers are cross-fitted out-of-fold scores of frozen EP-alpha systems; lockbox numbers
only for hash-pinned frozen systems (``a_common.FROZEN_LOCKBOX``). Run: ``py -3.12 -m uol.round2.a_p03 [b c d e]``.
"""
from __future__ import annotations

import math
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

from uol.embedding_push import common
from uol.facts_to_law import stats
from uol.round2 import a_common as A

EMB, NOEMB = "LTR_emb_hgb", "LTR_noemb_hgb"


# ----------------------------------------------------------------------------- (b) complementarity

def task_b() -> dict:
    out = {}
    for split in ("dev", "lockbox"):
        a, b = A.evaluate_cached(EMB, split), A.evaluate_cached(NOEMB, split)
        assert a["rows"] == b["rows"]
        d = common.release()
        cat = {k: {band: Counter() for band in A.BANDS + ("all",)} for k in (10, 20)}
        # per query, per band nDCG contribution difference
        contrib = {band: np.zeros(len(a["rows"])) for band in A.BANDS}
        for j, row in enumerate(a["rows"]):
            qa, qb = a["per_query"][j], b["per_query"][j]
            ra = dict(zip(qa["positive_idx"], qa["positive_ranks"]))
            rb = dict(zip(qb["positive_idx"], qb["positive_ranks"]))
            labels = A.band_labels(row)
            n_rel = qa["n_relevant"]
            for p in qa["positive_idx"]:
                band = str(labels[p])
                for k in (10, 20):
                    ha, hb = ra[p] <= k, rb[p] <= k
                    key = "both" if ha and hb else "EMB_only" if ha else "NOEMB_only" if hb else "neither"
                    cat[k][band][key] += 1
                    cat[k]["all"][key] += 1
                contrib[band][j] += A.ndcg_contrib(ra[p], n_rel) - A.ndcg_contrib(rb[p], n_rel)
        total_ids, total = A.component_values(a, [qa["ndcg_at_10"] - qb["ndcg_at_10"] for qa, qb in zip(a["per_query"], b["per_query"])])
        dec = {}
        for band in A.BANDS:
            _, cv = A.component_values(a, contrib[band])
            boot = stats.paired_cluster_bootstrap(cv)
            dec[band] = {"delta": boot["effect"], "ci95": [boot["lower"], boot["upper"]], "share": boot["effect"] / total.mean()}
        dec["total"] = float(total.mean())
        out[split] = {"categories": {k: {band: dict(v) for band, v in cat[k].items()} for k in cat}, "decomposition": dec,
                      "n_queries": len(a["rows"]), "n_components": len(total_ids)}
        A.log(f"(b) {split}: total delta {total.mean():+.4f}; " + ", ".join(f"{bd} {dec[bd]['delta']:+.4f}" for bd in A.BANDS))
    A.save_json("p03b_complementarity.json", out)
    return out


def md_b(res: dict) -> str:
    lines = []
    for split in ("dev", "lockbox"):
        r = res[split]
        lines.append(f"**{split}** (P_spec, {r['n_queries']} queries / {r['n_components']} components; bands "
                     f"{'fold-local' if split == 'dev' else 'from all dev rows'}). Target pairs recovered in the top 20 by "
                     f"`{EMB}` (EMB) and `{NOEMB}` (NO-EMB):\n")
        lines.append("| band | pairs | both | EMB only | NO-EMB only | neither | net EMB | share of Δ nDCG@10 | band Δ nDCG@10 [95% CI] |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for band in A.BANDS + ("all",):
            c = r["categories"][20][band] if "20" not in r["categories"] else r["categories"]["20"][band]
            n = sum(c.values())
            if band == "all":
                dd = r["decomposition"]["total"]
                share, ci_s = "100%", f"{dd:+.4f}"
            else:
                dd = r["decomposition"][band]
                share, ci_s = f"{100 * dd['share']:.0f}%", f"{dd['delta']:+.4f} {A.ci(dd['ci95'], 4)}"
            lines.append(f"| {band} | {n} | {c.get('both', 0)} | {c.get('EMB_only', 0)} | {c.get('NOEMB_only', 0)} | {c.get('neither', 0)} | "
                         f"{c.get('EMB_only', 0) - c.get('NOEMB_only', 0):+d} | {share} | {ci_s} |")
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- (c) vocabulary mismatch

C_SYSTEMS = ["L_star", "K_lex", "dense_qwen8b", NOEMB, EMB, "prior"]


def overlap_scores() -> np.ndarray:
    """Per release row: mean over specific positives of IDF-weighted share of query content tokens found in the article text."""
    path = A.CACHE / "overlap.npy"
    if path.exists():
        return np.load(path)
    d = common.release()
    art_tok = [A.content_tokens(t) for t in d.candidate_texts]
    df = Counter()
    for s in art_tok:
        df.update(s)
    n = len(art_tok)
    idf = {t: math.log((n + 1) / (c + 0.5)) for t, c in df.items()}
    out = np.full(len(d.queries), np.nan)
    for i, q in enumerate(d.queries):
        pos = d.specific_idx[i]
        if not pos:
            continue
        qt = A.content_tokens(q.text.replace("[CITA_OCULTA]", " ")) - {"cita", "oculta"}
        denom = sum(idf.get(t, math.log(n + 1)) for t in qt)
        if denom <= 0:
            out[i] = 0.0
            continue
        out[i] = float(np.mean([sum(idf.get(t, 0.0) for t in qt & art_tok[p]) / denom for p in pos]))
    A.CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out


def task_c(n_bins: int = 10) -> dict:
    ov = overlap_scores()
    res = {s: {split: A.evaluate_cached(s, split) for split in ("dev", "lockbox")} for s in C_SYSTEMS}
    dev_rows = np.asarray(res[EMB]["dev"]["rows"])
    edges = np.quantile(ov[dev_rows], np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    out = {"edges": edges[1:-1].tolist(), "bins": {}}
    for split in ("dev", "lockbox"):
        rows = np.asarray(res[EMB][split]["rows"])
        bins = np.clip(np.searchsorted(edges, ov[rows], side="right") - 1, 0, n_bins - 1)
        table = []
        for k in range(n_bins):
            keep = bins == k
            entry = {"bin": k, "n_queries": int(keep.sum()), "overlap_mean": float(ov[rows][keep].mean()) if keep.any() else None}
            for s in C_SYSTEMS:
                entry[s] = A.subset_mean(res[s][split], keep)
            c = A.subset_contrast(res[EMB][split], res[NOEMB][split], keep)
            entry["gain"] = c["effect"]
            entry["gain_ci95"] = c["ci95"]
            entry["n_components"] = c["n_components"]
            table.append(entry)
        # trend: correlation between overlap and per-query gain (component-level)
        ga = np.asarray([qa["ndcg_at_10"] - qb["ndcg_at_10"] for qa, qb in zip(res[EMB][split]["per_query"], res[NOEMB][split]["per_query"])])
        rho = float(np.corrcoef(np.argsort(np.argsort(ov[rows])), np.argsort(np.argsort(ga)))[0, 1])
        out["bins"][split] = {"table": table, "spearman_overlap_vs_gain": rho}
        A.log(f"(c) {split}: spearman(overlap, gain) {rho:+.3f}")
    A.save_json("p03c_vocab_mismatch.json", out)
    plot_c(out, n_bins)
    return out


def plot_c(out: dict, n_bins: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    t = out["bins"]["dev"]["table"]
    x = np.arange(1, n_bins + 1)
    labels = {"L_star": "tuned lexical (L*)", "K_lex": "K_lex", "dense_qwen8b": "dense Qwen3-8B", NOEMB: "LTR NO-EMB",
              EMB: "LTR EMB", "prior": "popularity only"}
    styles = {"prior": dict(color="grey", ls="--"), "L_star": dict(color="#8c6d31"), "K_lex": dict(color="#e6550d"),
              "dense_qwen8b": dict(color="#3182bd"), NOEMB: dict(color="#31a354"), EMB: dict(color="#756bb1", lw=2.5)}
    for s in C_SYSTEMS:
        axes[0].plot(x, [r[s] for r in t], marker="o", ms=3, label=labels[s], **styles[s])
    axes[0].set_xlabel("decile of facts-to-target lexical overlap (1 = least overlap)")
    axes[0].set_ylabel("dev P_spec nDCG@10")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Systems by vocabulary overlap (dev, cross-fitted)")
    g = np.asarray([r["gain"] for r in t])
    lo = np.asarray([r["gain_ci95"][0] for r in t])
    hi = np.asarray([r["gain_ci95"][1] for r in t])
    axes[1].bar(x, g, color="#756bb1", alpha=.7, label="dev")
    axes[1].errorbar(x, g, yerr=[g - lo, hi - g], fmt="none", ecolor="k", capsize=2)
    tl = out["bins"]["lockbox"]["table"]
    axes[1].plot(x, [r["gain"] if r["gain"] is not None else np.nan for r in tl], "o", color="#d62728", label="lockbox (frozen, descriptive)")
    axes[1].axhline(0, color="k", lw=.8)
    axes[1].set_xlabel("decile of lexical overlap (dev cut points)")
    axes[1].set_ylabel("LTR EMB - LTR NO-EMB nDCG@10")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Embedding gain by overlap decile (95% CI, dev)")
    fig.tight_layout()
    fig.savefig(A.OUT / "p03c_vocab_mismatch.png", dpi=130)
    plt.close(fig)


def md_c(out: dict) -> str:
    lines = ["| decile | overlap | n q (dev) | L* | K_lex | dense Qwen8B | LTR NO-EMB | LTR EMB | popularity | gain dev [95% CI] | n q (lockbox) | gain lockbox |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    for r, rl in zip(out["bins"]["dev"]["table"], out["bins"]["lockbox"]["table"]):
        lines.append(f"| {r['bin'] + 1} | {r['overlap_mean']:.2f} | {r['n_queries']} | {r['L_star']:.3f} | {r['K_lex']:.3f} | {r['dense_qwen8b']:.3f} | "
                     f"{r[NOEMB]:.3f} | {r[EMB]:.3f} | {r['prior']:.3f} | {A.fmt(r['gain'], 3, True)} {A.ci(r['gain_ci95'])} | {rl['n_queries']} | "
                     f"{A.fmt(rl['gain'], 3, True)} |")
    lines.append("")
    lines.append(f"Spearman correlation between overlap and per-query gain: dev {out['bins']['dev']['spearman_overlap_vs_gain']:+.3f}, "
                 f"lockbox {out['bins']['lockbox']['spearman_overlap_vs_gain']:+.3f}.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- (d) encoder scaling

ENCODERS = {"e5-small (118M)": 0.118, "bge-m3 (568M)": 0.568, "Qwen3-Emb-8B (7.6B)": 7.57}
D_ROWS = {  # family -> encoder -> system id
    "dense alone (MaxP)": {"e5-small (118M)": "dense_e5_maxp", "bge-m3 (568M)": "dense_bge_maxp", "Qwen3-Emb-8B (7.6B)": "dense_qwen8b"},
    "K_lex + dense + prior": {"e5-small (118M)": "KP_dense_e5_maxp", "bge-m3 (568M)": "KP_dense_bge_maxp", "Qwen3-Emb-8B (7.6B)": "KP_dense_qwen8b"},
    "kNN citation memory": {"e5-small (118M)": "knn_e5", "bge-m3 (568M)": "knn_bge", "Qwen3-Emb-8B (7.6B)": "knn_qwen8b"},
    "article prototypes": {"e5-small (118M)": "proto_e5", "bge-m3 (568M)": "proto_bge", "Qwen3-Emb-8B (7.6B)": "proto_qwen8b"},
}
D_EXTRA = {"dense_bge_long": "bge-m3 long (8k, no chunking)", "dense_bge_firstp": "bge-m3 first chunk", "ft_e5_q": "e5-small fine-tuned query tower",
           "L_star": "tuned lexical (reference)", "K_lex": "K_lex (reference)", "prior": "popularity only (reference)"}


def unseen_recall(res: dict) -> tuple[float, int]:
    hit = tot = 0
    for row, q in zip(res["rows"], res["per_query"]):
        lab = A.band_labels(row)
        for p, r in zip(q["positive_idx"], q["positive_ranks"]):
            if lab[p] == "unseen":
                tot += 1
                hit += r <= 20
    return hit / max(tot, 1), tot


def task_d() -> dict:
    out = {"families": {}, "extra": {}}
    for fam, enc in D_ROWS.items():
        out["families"][fam] = {}
        for e, sid in enc.items():
            r = A.evaluate_cached(sid, "dev")
            rp = A.evaluate_cached(sid, "dev", "P")
            ur, nu = unseen_recall(r)
            out["families"][fam][e] = {"system": sid, "P_spec": r["macro"]["ndcg_at_10"], "P": rp["macro"]["ndcg_at_10"],
                                       "R20_unseen_pairs": ur, "unseen_pairs": nu}
    for sid, label in D_EXTRA.items():
        r = A.evaluate_cached(sid, "dev")
        rp = A.evaluate_cached(sid, "dev", "P")
        ur, nu = unseen_recall(r)
        out["extra"][sid] = {"label": label, "P_spec": r["macro"]["ndcg_at_10"], "P": rp["macro"]["ndcg_at_10"], "R20_unseen_pairs": ur}
    A.save_json("p03d_encoder_scaling.json", out)
    plot_d(out)
    return out


def plot_d(out: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = list(ENCODERS.values())
    for metric, ax, ylab in (("P_spec", axes[0], "dev P_spec nDCG@10"), ("R20_unseen_pairs", axes[1], "dev Recall@20, unseen-band pairs")):
        for fam, rows in out["families"].items():
            ax.plot(x, [rows[e][metric] for e in ENCODERS], marker="o", label=fam)
        ft = out["extra"]["ft_e5_q"][metric]
        ax.plot([ENCODERS["e5-small (118M)"]], [ft], marker="*", ms=14, color="k", ls="none", label="e5-small fine-tuned (dense)")
        for sid, col in (("K_lex", "#e6550d"), ("prior", "grey")):
            ax.axhline(out["extra"][sid][metric], color=col, ls="--", lw=1, label=out["extra"][sid]["label"])
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(list(ENCODERS))
        ax.set_ylabel(ylab)
        ax.set_xlabel("frozen encoder (as released; recipes differ, not a clean scaling law)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(A.OUT / "p03d_encoder_scaling.png", dpi=130)
    plt.close(fig)


def md_d(out: dict) -> str:
    lines = ["| family | " + " | ".join(ENCODERS) + " |", "|---|" + "---:|" * len(ENCODERS)]
    for fam, rows in out["families"].items():
        lines.append(f"| {fam} | " + " | ".join(f"{rows[e]['P_spec']:.3f} / {rows[e]['R20_unseen_pairs']:.2f}" for e in ENCODERS) + " |")
    lines.append("")
    lines.append("Cells: dev P_spec nDCG@10 / Recall@20 on fold-local unseen target pairs. Other score files:")
    lines.append("")
    lines.append("| system | P_spec nDCG@10 | P nDCG@10 | R@20 unseen |")
    lines.append("|---|---:|---:|---:|")
    for sid, r in out["extra"].items():
        lines.append(f"| {r['label']} (`{sid}`) | {r['P_spec']:.3f} | {r['P']:.3f} | {r['R20_unseen_pairs']:.2f} |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- (e) strata

CONTRA = re.compile(r"contendient|contradicci[oó]n de (criterios|tesis)|criterios? (discrepantes|contradictorios|divergentes|distintos|opuestos)|"
                    r"(llegaron|arribaron) a (conclusiones|soluciones|criterios|posicionamientos) (distint|diferent|contrari|opuest|discrepant)|"
                    r"posturas? (contrarias|distintas|divergentes)", re.I)
STATES = (r"AGUASCALIENTES|BAJA CALIFORNIA|CAMPECHE|COAHUILA|COLIMA|CHIAPAS|CHIHUAHUA|DURANGO|GUANAJUATO|GUERRERO|HIDALGO|JALISCO|"
          r"M[EÉ]XICO|MICHOAC[AÁ]N|MORELOS|NAYARIT|NUEVO LE[OÓ]N|OAXACA|PUEBLA|QUER[EÉ]TARO|QUINTANA ROO|SAN LUIS POTOS[IÍ]|SINALOA|SONORA|"
          r"TABASCO|TAMAULIPAS|TLAXCALA|VERACRUZ|YUCAT[AÁ]N|ZACATECAS")
STATE_NARROW = re.compile(r"LEGISLACI[OÓ]N (DEL ESTADO|DE LA CIUDAD|APLICABLE PARA LA CIUDAD|DEL DISTRITO|DE (" + STATES + "))|"
                          r"C[OÓ]DIGO [^.]{0,60}(DEL ESTADO|PARA EL ESTADO|DE LA CIUDAD DE M|DEL DISTRITO FEDERAL)", re.I)
STATE_BROAD = re.compile(r"(DEL|PARA EL) ESTADO DE (" + STATES + r")|CIUDAD DE M[EÉ]XICO|DISTRITO FEDERAL|\bESTATAL(ES)?\b|"
                         r"\bLOCAL(ES)?\b|ENTIDAD(ES)? FEDERATIVA", re.I)
VERSION_RUBRO = re.compile(r"LEGISLACI[OÓ]N (VIGENTE|ANTERIOR|APLICABLE (HASTA|A PARTIR))|ANTES DE LA REFORMA|ABROGAD", re.I)


def strata_of(row: int, src: dict) -> dict[str, list[str]]:
    d = common.release()
    q = d.queries[int(row)]
    s = src.get(q.query_id, {})
    rubro = s.get("source_rubro") or ""
    return {"materia": list(q.materias) or ["(none)"],
            "vehicle": ["contradicción-type facts" if CONTRA.search(q.text) else "case facts"],
            "state_law_rubro": ["state (narrow: legislación/código estatal)" if STATE_NARROW.search(rubro)
                                else "state/local mention (broad only)" if STATE_BROAD.search(rubro) else "no state marker"],
            "version_marker_in_rubro": ["yes" if VERSION_RUBRO.search(rubro) else "no"],
            "instancia": [q.instancia or "?"], "tipo": [q.tipo or "?"], "epoca": [s.get("source_epoca") or "?"]}


def task_e() -> dict:
    src = A.load_sources()
    out = {}
    for split in ("dev", "lockbox"):
        a, b = A.evaluate_cached(EMB, split), A.evaluate_cached(NOEMB, split)
        st = [strata_of(r, src) for r in a["rows"]]
        out[split] = {}
        for dim in st[0]:
            levels = sorted({lv for s in st for lv in s[dim]})
            out[split][dim] = {}
            for lv in levels:
                keep = np.asarray([lv in s[dim] for s in st])
                if keep.sum() < 5:
                    continue
                out[split][dim][lv] = A.subset_contrast(a, b, keep)
        A.log(f"(e) {split} done")
    A.save_json("p03e_strata.json", out)
    return out


def md_e(out: dict) -> str:
    lines = ["| dimension | stratum | dev q/comp | dev EMB | dev NO-EMB | dev Δ [95% CI] | lockbox q | lockbox EMB | lockbox NO-EMB | lockbox Δ [95% CI] |",
             "|---|---|---:|---:|---:|---|---:|---:|---:|---|"]
    for dim, levels in out["dev"].items():
        for lv, r in levels.items():
            l = out["lockbox"].get(dim, {}).get(lv)
            lt = (f"{l['n_queries']} | {A.fmt(l['mean_a'])} | {A.fmt(l['mean_b'])} | {A.fmt(l['effect'], 3, True)} {A.ci(l['ci95'])}"
                  if l else "– | – | – | –")
            lines.append(f"| {dim} | {lv} | {r['n_queries']}/{r['n_components']} | {A.fmt(r['mean_a'])} | {A.fmt(r['mean_b'])} | "
                         f"{A.fmt(r['effect'], 3, True)} {A.ci(r['ci95'])} | {lt} |")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    tasks = argv or ["b", "c", "d", "e"]
    for t in tasks:
        t0 = time.perf_counter()
        res = {"b": task_b, "c": task_c, "d": task_d, "e": task_e}[t]()
        md = {"b": md_b, "c": md_c, "d": md_d, "e": md_e}[t](res)
        (A.OUT / f"p03{t}.md").write_text(md, encoding="utf-8")
        A.log(f"P03({t}) finished in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
