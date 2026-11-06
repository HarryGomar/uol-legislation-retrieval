# Retrieval of Mexican federal legislation from court facts

Code of the thesis *Retrieval of Mexican Federal Legislation with Language Models and Learning to Rank* (ITAM, 2026).

Given the facts of a published Mexican court decision (*tesis*), with every name of a law or article masked, the task
is to rank the 39,178 articles of the 264 federal laws in force on 15 September 2024 so that the articles the court
cited come first. The benchmark is published separately (`uol-benchmark`, see [Data](#data)).

## The three setups

| Setup | What it does | nDCG@10 (development) |
|---|---|---:|
| `noenc` | a learned ranker over five lexical and citation sources, no encoder, no language model | 0.432 |
| `ranker` | the same ranker with three sources built on Qwen3-Embedding-8B vectors | 0.489 |
| `system` | the ranker's list merged with GPT-6.1 Sol's list of 20, then the first 30 reordered by GPT-6.1 Sol | 0.661 |

Scores are on the 1,217 development evaluation cases, five outer folds, averaged within each group of related cases
and then across groups.

**How a case is answered.** The first stage scores every article with each source: BM25 over Spanish stems fused with
character n-grams, BM25 over articles expanded with the facts of training cases that cite them, the articles cited by
the most similar training cases, the citation prior, similarity to article prototypes, and (in `ranker`) the cosine
to each article, the votes of similar cases and the prototypes, all with Qwen3-Embedding-8B. The union of each
source's 200 best articles is the pool. Each pooled article gets three columns per source, three describing the pair
and three from the votes of the older *tesis* most similar to the case (84,459 *tesis* published before 14 August
2020), and a LightGBM LambdaMART model orders the pool. The `system` setup asks GPT-6.1 Sol for 20 articles from the
facts alone, merges that list with the ranker's by reciprocal rank fusion, has the model reorder the first 30 in two
windows of 20, and fuses that order with the merged one.

## Install

Python 3.11 or later. The pinned numpy and LightGBM versions are the ones the trained trees depend on.

```bash
pip install -e .                    # the rankers
pip install -e ".[encoder]"         # Qwen3-Embedding-8B for new cases (a CUDA card, or the processor)
pip install -e ".[model,test]"      # the language-model steps (OpenAI API) and the tests
python -m nltk.downloader stopwords
pytest
```

## Data

| Folder | Content | Where |
|---|---|---|
| `data/benchmark/` | cases, labels, splits, catalogue, older *tesis* | the `uol-benchmark` dataset |
| `data/vectors/` | Qwen3-Embedding-8B vectors of the article windows, of every case (two instructions) and of the older *tesis* | the `vectors/` folder of the dataset |
| `models/` | the fitted LightGBM models, `models/<setup>/group<k>.txt` | the thesis's two group-5 models are in the `models/` folder of the dataset; `python -m uol fit` builds all six |

Set `UOL_DATA` or pass `--data` to read the benchmark from elsewhere.

## Use

```bash
python -m uol fit --setup noenc                          # out-of-fold table, six models, development score
python -m uol fit --setup ranker
python -m uol answer --setup ranker --group 0            # fold 0's development cases, one at a time
python -m uol answer --setup system --group 5 --population TEST --encoder cuda
python -m uol score work/system/TEST_group5.jsonl --population TEST
python -m uol compare work/system/DEV_group0.jsonl work/ranker/DEV_group0.jsonl --key final
```

A group is an outer fold (0-4), whose model never saw that fold's cases, or 5, the model trained on every development
fold, which answers new and test cases. `answer` writes one line per case with the ranker's list, the model's list,
the merged list, the final list and the seconds each step took. `--encoder stored` reads the benchmark's stored case
vectors instead of encoding the facts; `--route codex` reaches the model through the Codex command line, as in the
thesis, instead of the OpenAI API (`OPENAI_API_KEY`). Model answers are cached in `work/llm_cache.jsonl`.

## Reproduction

`python -m uol fit` rebuilds both rankers from the benchmark alone. The ranker without the encoder comes out as in
the thesis: the same first ten articles on all 1,217 development cases and the same 200 on 1,215 (the rest are ties
broken by float rounding), development nDCG@10 0.432, and a group-5 model that scores every pair as the stored one.
The three Qwen sources differ from the stored scores by about one part in a million, because a matrix product over
many cases rounds differently from one over a single case. These differences change the trees: the refitted ranker
with the encoder scores 0.484 on development against 0.489 for the thesis's (difference −0.005, 95% interval −0.011
to +0.001). The dataset holds the thesis's own group-5 models, and with a stored model `answer` gives the stored lists
(fold 0: the same first ten articles on all 228 cases).

Fresh encoding differs slightly from the stored vectors (median cosine 0.992 on a card in 8 bits, 0.997 on the
processor), and a language model does not answer twice the same way, so `answer` with a fresh encoder or fresh model
calls is close to, not identical with, the stored lists.

Every experiment of the thesis is listed in [`experiments/`](experiments/README.md), with the stored per-case scores
of all 414 systems and the command that reruns it. The research code as it ran is in [`research/`](research/README.md).

## Layout

| Path | What it holds |
|---|---|
| `src/uol/data.py` | the benchmark: cases, labels, folds, catalogue, older *tesis* |
| `src/uol/text.py`, `sources.py`, `dense.py` | the lexical indexes and the eight first-stage sources |
| `src/uol/features.py`, `history.py`, `ranker.py` | the pool and its columns, the older-*tesis* votes, LightGBM |
| `src/uol/system.py` | the two rankers and their per-group settings |
| `src/uol/train.py` | out-of-fold training |
| `src/uol/llm.py`, `prompts/` | the language-model steps and their prompts |
| `src/uol/encoder.py` | Qwen3-Embedding-8B for new cases |
| `src/uol/run.py`, `evaluate.py`, `__main__.py` | the whole run, scoring, the command line |
| `tests/` | the rules the stored results depend on |
| `experiments/` | every experiment: stored results, summaries, commands |
| `research/` | the research code as it ran |

## Citation

See `CITATION.cff`. The data keep the attributions of their sources: Semanario Judicial de la Federación, Suprema
Corte de Justicia de la Nación; Cámara de Diputados del H. Congreso de la Unión, Leyes Federales Vigentes.
