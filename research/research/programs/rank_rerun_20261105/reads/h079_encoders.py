"""H079: every encoder tested, its stored parameters, alone score and the score of a ranker with it. Dimensions come from
the job specs (uol.final.adapters.gpu.EXTRA_SPECS) and the figure module's PARAMS_M / DIMS; where neither holds a value the
cell says "not on record". Ranker scores: S0+x in the 25 Oct record (older ranker) and lf_noenc+x / rk_noenc+x on the
footing, where stored. Writes out/h079.json."""
from common import *

P = {"e5_small": 118, "e5_base": 278, "e5_large": 560, "e5_large_instruct": 560, "gte_multilingual": 305, "jina_v3": 570,
     "arctic_l_v2": 568, "bge_m3": 568, "qwen3_0p6b": 600, "qwen3_4b": 4000, "qwen3_8b": 8000, "e5_large_legal_en": 560,
     "bge_m3_legal": 568, "gemma_legal": 300, "euler_legal": 8000, "alia_mrbert": 150}     # ch6_figs.PARAMS_M
from uol.final.adapters import gpu
ES = gpu.EXTRA_SPECS
ROWS = [  # id, printed name, kind, trained on the task?
    ("e5_small", "E5-small", "general", "no"), ("e5_base", "E5-base", "general", "no"), ("e5_large", "E5-large", "general", "no"),
    ("e5_large_instruct", "E5-large-instruct", "general", "no"), ("gte_multilingual", "GTE-multilingual", "general", "no"),
    ("jina_v3", "Jina v3", "general", "no"), ("arctic_l_v2", "Arctic-L v2", "general", "no"), ("bge_m3", "BGE-M3 (frozen vectors)", "general", "no"),
    ("bge_m3_base", "BGE-M3 (re-encoded)", "general", "no"), ("qwen3_0p6b", "Qwen3-Embedding-0.6B", "general", "no"),
    ("qwen3_4b", "Qwen3-Embedding-4B", "general", "no"), ("qwen3_8b", "Qwen3-Embedding-8B", "general", "no"),
    ("qwen3_8b_int8_base", "Qwen3-8B, eight-bit path", "general", "no"),
    ("e5_large_legal_en", "E5-large-legal (English)", "legal", "by its makers, not on this task"), ("bge_m3_legal", "BGE-M3 legal", "legal", "by its makers, not on this task"),
    ("gemma_legal", "EmbeddingGemma legal", "legal", "by its makers, not on this task"), ("euler_legal", "Euler-Legal (Qwen3-8B)", "legal", "by its makers, not on this task"),
    ("alia_mrbert", "ALIA MrBERT", "legal", "by its makers, not on this task"),
    ("ft_e5_q", "E5-small, query tower fine-tuned", "task", "yes, query tower, per fold"), ("bge_m3_ft", "BGE-M3 fine-tuned", "task", "yes, whole model, per fold"),
    ("qwen3_0p6b_ft", "Qwen3-0.6B fine-tuned", "task", "yes, whole model, per fold"), ("qwen8b_adapter", "Qwen3-8B + adapter", "task", "adapter only, per fold; encoder frozen"),
]
DIM = {"e5_small": 384, "bge_m3": 1024, "bge_m3_base": 1024, "qwen3_0p6b": 1024, "qwen3_4b": 2560, "qwen3_8b": 4096, "qwen3_8b_int8_base": 4096,
       "ft_e5_q": 384, "bge_m3_ft": 1024, "qwen3_0p6b_ft": 1024,
       "alia_mrbert": 768, "gemma_legal": 768, "bge_m3_legal": 1024, "euler_legal": 4096, "e5_large_legal_en": 1024}   # uol.scale_curve.legal specs; e5_large_legal_q
out = []
base_s0 = summary("S0")["ndcg_at_10"]
for sid, name, kind, tuned in ROWS:
    try:
        j = summary(sid)
    except FileNotFoundError:
        print("missing", sid); continue
    spec = ES.get(sid) or ES.get(sid.replace("_en", "")) or {}
    dim = DIM.get(sid) or spec.get("dim")
    par = P.get(sid.replace("_int8_base", "").replace("_base", "") if sid not in P else sid)
    if sid.startswith("ft_e5"):
        par = 118
    if sid == "bge_m3_ft":
        par = 568
    if sid == "qwen3_0p6b_ft":
        par = 600
    if sid == "qwen8b_adapter":
        par = 8000
    r = {"id": sid, "name": name, "kind": kind, "tuned": tuned, "dim": dim, "params_m": par, "max_tokens": spec.get("max_tokens"),
         "alone": j["ndcg_at_10"], "alone_R20": j["recall_at_20"], "alone_never": j["unseen_recall_at_20_pooled"]}
    for lab, key in (("S0+", f"S0+{sid}"), ("lf+", f"lf_noenc+{sid}"), ("rk+", f"rk_noenc+{sid}")):
        try:
            r[lab] = summary(key)["ndcg_at_10"]
            r[lab + "never"] = summary(key)["unseen_recall_at_20_pooled"]
        except FileNotFoundError:
            r[lab] = None
    out.append(r)
df = pd.DataFrame(out)
pd.set_option("display.width", 240, "display.max_columns", 30, "display.max_colwidth", 40)
print(df[["id", "kind", "tuned", "dim", "params_m", "alone", "alone_never", "S0+", "lf+", "rk+"]].round(3).to_string())
print("S0", base_s0, "ft_lean_noenc", summary(NOENC)["ndcg_at_10"], "ft_lean", summary(RANKER)["ndcg_at_10"], "L*", summary(LEX)["ndcg_at_10"])
for b in ("rk_noenc",):
    try:
        print(b, summary(b)["ndcg_at_10"], summary(b)["unseen_recall_at_20_pooled"], "rt_lm_tesis", summary(NEWRANKER)["ndcg_at_10"])
    except FileNotFoundError:
        print(b, "not stored")
g = df[df["kind"] == "general"]
print("Spearman params vs alone, general, all:", round(g[["params_m", "alone"]].corr(method="spearman").iloc[0, 1], 3), "| without Qwen3:", round(g[~g["id"].str.startswith("qwen")][["params_m", "alone"]].corr(method="spearman").iloc[0, 1], 3))
print("Spearman alone vs S0+:", round(df.dropna(subset=["S0+"])[["alone", "S0+"]].corr(method="spearman").iloc[0, 1], 3))
print("\nQwen3-8B truncated or quantised")
for s in ("qwen3_8b", "qwen3_8b_mrl1024", "qwen3_8b_mrl512", "qwen3_8b_mrl256", "qwen3_8b_int8", "qwen3_8b_binary", "qwen3_8b_headings"):
    a = summary(s)["ndcg_at_10"]
    try:
        b = summary("S0+" + s)["ndcg_at_10"]
    except FileNotFoundError:
        b = None
    print(s, round(a, 3), None if b is None else round(b, 3))
save("h079", out)

print("\nspec dims:", {k: (v.get("dim"), v.get("params"), v.get("max_tokens")) for k, v in ES.items()})
ten = ["e5_small", "e5_base", "e5_large", "gte_multilingual", "jina_v3", "arctic_l_v2", "bge_m3", "qwen3_0p6b", "qwen3_4b", "qwen3_8b"]
g10 = df[df["id"].isin(ten)]
print("the ten general encoders of F6_3: Spearman size/alone", round(g10[["params_m", "alone"]].corr(method="spearman").iloc[0, 1], 3), "| the seven at 600M or below, without Qwen3:", round(g10[~g10["id"].str.startswith("qwen")][["params_m", "alone"]].corr(method="spearman").iloc[0, 1], 3))
print("of the three above 600M all are Qwen3 or its legal form; size and model line cannot be separated above 600M")
print("rk ladder stored:", {x.name: round(summary(x.name)["ndcg_at_10"], 3) for x in FOOT.iterdir() if x.name.startswith("rk_noenc") and (x / "DEV/summary.json").exists()})
