# Experiments

Every development result in the thesis can be recomputed from the stored scores in `results/`, and every experiment
can be rerun: the final setups with the `uol` package, the others with the research code as it ran (`../research`).

| File | What it holds |
|---|---|
| `results/dev_per_case.parquet` | nDCG@10, MRR, recall@20/100 and success@1/10 of each of the 416 stored systems on each of the 1,217 development evaluation cases (`record`, `system`, `query_id`, `component_id`, measures) |
| `results/dev_lists/*.parquet` | the ranked lists (first 200) of the seven main systems |
| `results/MAIN.json` | which stored system each main row is |
| `summary.py` | means over groups and paired comparisons with a 95% bootstrap interval |

```bash
python experiments/summary.py
python experiments/summary.py --compare footing/gpt61sol_rk_system footing/rt_lm_tesis
```

The two records are `oct25` (the first full run of every baseline, encoder and ranker, 25 Oct 2026) and `footing`
(the final rankers, the language-model runs and the system, from 1 Nov 2026). Test-set results are not in this
folder.

## Rerunning

**Final setups** (`uol` package, see the main README): `python -m uol fit --setup {noenc,ranker}` refits the ranker
of every group from the benchmark and scores the development lists; `python -m uol answer --setup system` answers
cases one at a time from their facts.

**Everything else** runs from `research/` (Python 3.12, the pinned versions in `research/requirements.txt`, and the
thesis's run data in place of `data/benchmark`; see `research/README.md`):

```bash
cd research
PYTHONPATH=src python -m uol.final.rank_rerun ranker --id rk_noenc --go
```

## The experiments, by thesis section

Systems are named as in `summary.py` (`record/id`). Commands are run from `research/` unless they start with
`python -m uol`.

### Chapter 6.1 The overall result

| Question | Systems | Rerun |
|---|---|---|
| How do the main systems compare? | `oct25/lstar_headings`, `oct25/qwen3_8b`, `footing/rk_noenc`, `footing/rt_lm_tesis`, `footing/gpt61sol_rerank_rt_lm_tesis`, `footing/gpt61sol_alone`, `footing/gpt61sol_rk_system` | `python -m uol fit` (both rankers); the rows below for the others |
| What does each setup hold beyond the first ten? | lists in `results/dev_lists/` | `research/programs/rank_rerun_20261105/reads/framing_numbers.py` |

### 6.2 Lexical matching and the collection

| Question | Systems | Rerun |
|---|---|---|
| How far do lexical, topic, document-vector and dense models go alone? | `oct25/{lstar_headings,bm25_stem,lsa,lda_dm,lda_sim,doc2vec,qwen3_8b,...}` | `python -m uol.final plan --out jobs/plan.json`, then `python -m uol.final run --system <id> --population DEV` per system |
| Why does LSA fall below lexical matching? | `oct25/lsa`, `oct25/lstar_headings`, `oct25/bm25_stem` | `research/programs/rank_rerun_20261105/reads/h073_lsa.py` |
| How much does each family depend on shared words and on past citations? | `oct25/lstar_headings`, `oct25/qwen3_8b`, `footing/rt_lm_tesis`, `footing/gpt61sol_alone`, `footing/gpt61sol_rk_system` | `python -m uol.thesis_figures.v21_figs`; `research/programs/rank_rerun_20261105/reads/h075_bands.py` |

### 6.3 Learning from past citations

| Question | Systems | Rerun |
|---|---|---|
| Do citations help as labels, as inputs, or both, and does the encoder add to each? | `footing/{rk_neither,rk_nocit,rk_noenc,rt_lm_tesis}` | `python -m uol.final.rank_rerun ranker --id <id> --go` |
| What do the older *tesis*' votes add? | `footing/rk_noenc_novotes` vs `footing/rk_noenc`; `footing/rt_check` vs `footing/rt_lm_tesis` | `python -m uol.final.rank_rerun ranker --id rk_noenc_novotes --go`; `python -m uol.final.ranker_tests fit --id rt_check --go` |
| How are the sources best combined (listwise or pointwise trees, LightGBM or XGBoost, per-article classifiers, more sources, topic scores)? | `footing/{rt_lm_tesis,rt_xgb_tesis,rt_lm_wide_tesis,lf_plain,ft_lean,S_full_ext,rk_topics}` | `python -m uol.final.ranker_tests fit --id <id> --go`; `python -m uol.final.classifier_baseline --go` |
| Do co-citation, statute structure or graph smoothing help? | `footing/{rk_cocite,rk_structure,rk_smooth}` vs `footing/rt_lm_tesis` | `python -m uol.final.footing arm-cols --arm {S2,S3,E3} --go`, then `python -m uol.final.rank_rerun ranker --id <id> --go` |

### 6.4 Dense retrievers

| Question | Systems | Rerun |
|---|---|---|
| What does the encoder add to the ranker, and do smaller encoders add less? | `footing/rt_lm_tesis` vs `footing/rk_noenc`; `footing/rk_noenc+{bge_m3,qwen3_0p6b,qwen3_4b,qwen3_8b}` | `python -m uol fit` (both rankers); `python -m uol.final.rank_rerun ranker --id rk_noenc+qwen3_4b --go` |
| Does size explain the encoders' order? | `oct25/{qwen3_0p6b,qwen3_4b,qwen3_8b,e5_*,arctic_l_v2,bge_m3,...}` | as 6.2, first row (needs a GPU) |
| Does legal pre-training or task training help? | `oct25/{euler_legal,bge_m3_legal,ft_e5_q,qwen3_0p6b_ft,...}`; `footing/rk_adapter` | as 6.2, first row; `python -m uol.final.rank_rerun ranker --id rk_adapter --go` |
| Do late interaction or learned sparse weights cross the vocabulary gap? | `oct25/{bge_m3,bge_m3_colbert,bge_m3_sparse}`; `footing/rk_mv` | as 6.2, first row; `python -m uol.final.multivector prep --go`, `... ingest --go`; `python -m uol.final.mv_ranker --go` |
| Does rewriting the facts with a language model help? | `oct25/{hyde,query2doc,qwen3_8b_int8q}`; `footing/{rk_q2d,rk_hyde,rk_issues,rk_patterns}` | `python -m uol.final.query_server {post,query2doc,hyde,stop} --go`; `python -m uol.final.rank_rerun ranker --id <id> --go` |

### 6.5 Reranking by a language model

| Question | Systems | Rerun |
|---|---|---|
| Are the ranker's errors in the law or in the article? | `footing/rt_lm_tesis` | `scripts/final_results.py --measure` |
| Does every model improve the ranker's first 30, and at what cost? | `footing/{gpt61sol,gpt61terra,gpt6solnw,gpt6luna,gpt56sol,gpt56terra}_rerank_rt_lm_tesis`, `footing/gpt61sol_{low,high}_rerank_rt_lm_tesis` | `python -m uol.final.rank_rerun model --kind <kind> --go` (Codex) |
| Does more reasoning effort help? | `footing/gpt61sol_alone_{low,high}` vs `footing/gpt61sol_alone` | `python -m uol.final.rank_rerun alone --effort {low,high} --go` |
| Do cross-encoders reorder the list as well? | `footing/{bge_ranker,qwen4b_ranker,ce_ranker}_rk{,_alone}` | `python -m uol.final.rank_rerun rr --kind <kind> --go` (GPU) |
| Is the gain from the model or from reading the candidates side by side? | pointwise judge vs `footing/gpt61sol_rerank_rt_lm_tesis` | `python -m uol.final.rank_rerun pointwise --go` |

### 6.6 The model alone

| Question | Systems | Rerun |
|---|---|---|
| How well does the model rank from the facts alone, and what limits its list? | `footing/gpt61sol_alone` | `python -m uol.final.llm_agents alone_full --go`; `python -m uol.final.model_limits` |
| Does list length or cases per call change its score? | list of 10, 40 and 100, one case per call | `python -m uol.final.prompt_check run --variant {list10,list40,one_call,list100} --go`, then `... score` |
| Does its score rest on recalling the published *tesis*? | probes, reworded facts, lead by year | `python -m uol.final.why_models {probe,reword,reword-alone,controls,report} --go` |
| What does the model know of an article without its text? | `oct25/llm_rerank_S_full_ext_blank`, `oct25/llm_rerank_S_full_ext`, `oct25/S_full_ext` | `research/programs/rank_rerun_20261105/reads/framing_numbers.py` |

### 6.7 The model and the ranker together

| Question | Systems | Rerun |
|---|---|---|
| What does the system gain over its parts, and do its settings matter? | `footing/gpt61sol_rk_system` and `_merged`, `_low`, `_high`, `_d60`, `_both` | `python -m uol answer --setup system`; `python -m uol.final.rank_rerun system --check <check> --go`; `research/programs/overnight_20261105/system_parts.py` |
| Does the encoder earn its place inside the system? | `footing/gpt61sol_rk_system_noenc` vs `footing/gpt61sol_rk_system` | `python -m uol answer --setup system` with the `noenc` models; `research/programs/rank_rerun_20261105/reads/e8_encoder_in_system.py` |
| Where do the system's misses lie? | first 30 of `results/dev_lists/system.parquet` | `research/programs/overnight_20261105/e4_misses.py`; `scripts/t6_failure.py` |
| Do other ways of using the model beat the system? | repeat, examples, agents (300-case sample) | `python -m uol.final.llm_agents {alone,repeat,examples} --go`; `python -m uol.final.rank_rerun agent --tool {words,embedding} --go` |

### 6.8 The test set

The final system was run once on the 398 test evaluation cases: nDCG@10 0.684 for the system, 0.660 for GPT-6.1 Sol's
list alone, 0.527 for the ranker and 0.468 for the ranker without the encoder. `python -m uol answer --setup
{noenc,ranker,system} --population TEST --group 5` answers the test cases with the models trained on every development
fold; `python -m uol score` scores the answers when the test labels are present. The research code's run is
`research/src/uol/final/whole_test.py`.

### 6.9 The three setups

| Question | Systems | Rerun |
|---|---|---|
| Does the ranker add to the model's answer, and does a longer model list replace it? | `footing/gpt61sol_final_self` vs `footing/gpt61sol_rk_system` | `python -m uol.final.llm_agents final --check self --go`; `python -m uol.final.overnight lists --go`, `... rerank --which list100_rerank --go` |
| Do the setups show supporting *tesis*, and how long and costly is each? | `footing/rt_lm_tesis`, `footing/gpt61sol_alone`, `footing/gpt61sol_rk_system` | `scripts/support_share.py`; timings and cost from `python -m uol answer` (it records both) |

### Benchmark and labels (Chapters 3-4, Appendix A)

| Question | Rerun |
|---|---|
| Cases, groups, pairs and articles after each construction step | `scripts/benchmark_full_counts.py` |
| Facts length, article length, shared words, articles cited per case | `scripts/final_results.py --measure` |
| How complete are the labels? | `scripts/label_completeness.py` |
| Does training on cases whose facts name a provision matter? (`footing/rk_train_no_masked`, `footing/rk_train_unnamed`) | `python -m uol.final.rank_rerun ranker --id <id> --go` |
| Seed and repetition variation (`footing/rk_seed1`, `footing/rk_seed2`) | `python -m uol.final.rank_rerun ranker --id <id> --go` |
| Does the code reproduce both rankers? (`footing/rk_repro_ranker`, `footing/rk_repro_noenc`: all 1,217 cases identical) | `python -m uol.final.rank_rerun ranker --id <id> --go` |
