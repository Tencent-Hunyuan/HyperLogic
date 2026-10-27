<p align="left">
    <a href="README_CN.md">中文</a>&nbsp;｜&nbsp;English
</p>
<br>

<p align="center">
 <img src="assets/logo-en.png" alt="Tencent Hy" width="400"/> <br>
</p>

<h1 align="center">HyperLogic: A Hard, Forward-Authored Chinese Logical Reasoning Benchmark with Execution-Derived Answers</h1>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](#-license)
&nbsp;&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv-2605.19597-b31b1b)](https://arxiv.org/abs/2605.19597)
&nbsp;&nbsp;
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-HyperLogic-ffd21e)](https://huggingface.co/datasets/tencent/HyperLogic)
&nbsp;&nbsp;
[![Tiers](https://img.shields.io/badge/Tiers-Seed%20%7C%20Base%20%7C%20Hard-ff7043)](#-overview)
&nbsp;&nbsp;
[![Leaderboard](https://img.shields.io/badge/Leaderboard-7%20frontier%20models-ffc107)](#-leaderboard)
&nbsp;&nbsp;
[![Language](https://img.shields.io/badge/Language-Chinese-624aff)](#-data-format)
&nbsp;&nbsp;
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab)](#-quick-start)

</div>

<p align="center">
    📄&nbsp;<a href="https://arxiv.org/abs/2605.19597"><b>Paper</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
    🤗&nbsp;<a href="https://huggingface.co/datasets/tencent/HyperLogic"><b>Dataset</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
    🚀&nbsp;<a href="#-quick-start"><b>Quick Start</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
    📮&nbsp;<a href="#-evaluate-your-model"><b>Evaluate Your Model</b></a></p>

---

## 🔥 News

- **2026.10.27**: 💻 This repository and the [Hugging Face dataset](https://huggingface.co/datasets/tencent/HyperLogic) go public with about 30% of the items of every tier, their gold answers, the direct-answer prompt and scorer, and the complete Hard agent evaluation (both sandbox conditions, their exact payloads, and the logic modeling library).
- **2026.10.02**: 🎉 The paper is released on arXiv as HyperLogic (v2 of arXiv:2605.19597) — HyperLogic-Base with 195 items and 922 sub-questions, HyperLogic-Hard with 100 items, and results for 7 frontier models. [[Paper]](https://arxiv.org/abs/2605.19597)

## 📋 Contents

- [Overview](#-overview)
- [Leaderboard](#-leaderboard)
- [Key Findings](#-key-findings)
- [What Is Released](#-what-is-released)
- [Quick Start](#-quick-start)
- [Data Format](#-data-format)
- [Repository Structure](#-repository-structure)
- [Evaluate Your Model](#-evaluate-your-model)
- [License](#-license)
- [Citation](#-citation)
- [Contact](#-contact)

## 📄 Overview

Existing logic benchmarks mostly measure whether a model can answer a reasoning question directly, and many of them no longer separate frontier models. Benchmarks that generate text from formal structures make answers easy to compute, but fix the formalization before the problem is written. HyperLogic is built **forward**: problems are written first, and their answers are derived afterwards by executing formal models of them.

The construction pipeline separates problem authoring from answer generation. A multi-agent workflow hardens Chinese seeds written by logic-trained undergraduates, **without solving them**. Two agents from different model families then independently translate each finished item into executable finite-domain models, and logic solvers derive the answers. Encodings and answers go through layered, agent-assisted adjudication under human-expert oversight; an item is released only when its two corrected encodings agree. Because the solvers do the search, items can be harder than the models that built them, while every answer stays executable and auditable.

<p align="center">
  <img src="assets/overview.png" alt="Overview of HyperLogic" width="95%">
</p>

*Overview of HyperLogic. **(a)** A four-role agent loop hardens seeds with tier-specific strategy libraries. **(b)** Two independent formalizations provide solver-derived answers for layered adjudication. **(c)** Both tiers are evaluated by direct answering; Hard also uses sandbox agents, with or without the logic modeling library.*

**Stats**: 228 Seed items · 195 Base items with 922 sub-questions · 100 Hard items · 7 frontier models · 3 rollouts per model and condition · every gold answer derived by solvers from two agreeing independent formalizations

**Tiers**:
- **HyperLogic-Seed** — 228 single-question seeds written by undergraduates with coursework in logic. Like existing benchmarks, Seed barely separates frontier models: all seven are within 5.9 points of each other.
- **HyperLogic-Base** — 195 items, each a shared background (about 3,300 Chinese characters) with 4.7 sub-questions on average. An item counts as correct only if every sub-question is. Frontier models can still answer these directly, and Base is designed to separate them.
- **HyperLogic-Hard** — 100 single-question items with larger, coupled search spaces (backgrounds of about 8,700 characters). Direct solving is no longer practical; Hard also evaluates agents that write code in a sandbox, with or without our logic modeling library.

## 🏆 Leaderboard

Accuracy (%) on the full benchmark, mean ± standard deviation over three rollouts, every model at its highest reasoning setting. **Base Item** counts an item as correct only if all its sub-questions are. **Sandbox** gives the agent a code sandbox; **+ Library** also gives it the logic modeling library. Bold marks the best model in each column.

| Rank | Model | Seed | Base Item | Base Sub-Q | Hard Direct | Hard Sandbox | Hard + Library |
|:---:|---|---:|---:|---:|---:|---:|---:|
| 1 | GPT-5.6 Sol | **89.8** ± 3.7 | **77.6** ± 0.6 | **92.6** ± 0.2 | 13.7 ± 5.9 | 40.3 ± 4.2 | 45.7 ± 2.1 |
| 2 | Claude Opus 5 | 86.5 ± 0.7 | 74.2<sup>†</sup> ± 2.7 | 88.9<sup>†</sup> ± 0.7 | **15.6**<sup>†</sup> ± 1.8 | **55.7** ± 3.2 | **57.0** ± 2.6 |
| 3 | Kimi K3 | 88.6 ± 0.0 | 57.9 ± 1.8 | 81.6 ± 1.8 | 4.0 ± 0.0 | 41.3 ± 1.5 | 43.0 ± 8.9 |
| 4 | Grok 4.6 | 87.0 ± 1.8 | 52.5 ± 4.1 | 82.9 ± 0.7 | 12.0 ± 1.0 | 39.3 ± 1.5 | 38.3 ± 2.1 |
| 5 | DeepSeek V4 Pro | 83.9 ± 1.7 | 51.8 ± 1.5 | 81.2 ± 1.2 | 9.3 ± 1.2 | 26.0 ± 5.6 | 20.0 ± 1.7 |
| 6 | GLM-5.3 | 87.3 ± 0.9 | 45.6 ± 3.2 | 67.2 ± 4.9 | 3.0 ± 1.0 | 20.7 ± 0.6 | 31.3 ± 5.1 |
| 7 | Doubao Seed 2.1 Pro | 85.8 ± 0.7 | 44.6 ± 2.2 | 73.2 ± 1.0 | 5.0 ± 1.0 | 26.0 ± 1.0 | 33.7 ± 0.6 |

<sup>†</sup>Excludes items refused by the model's API service. Grok 4.6 runs at `xhigh`, every other model at `max`. Models are ranked by Base item accuracy.

## 💡 Key Findings

- **Base is hard and discriminative.** On five of seven existing logic benchmarks, three frontier models differ by at most three points, and Seed behaves the same way (83.9–89.8%). On Base every model loses at least 12 points from Seed and the seven models spread over 33.0 points (44.6–77.6%).
- **Longer reasoning does not mean higher accuracy.** Median reasoning length on Base ranges from 15.2k to 52.7k tokens per item and does not track accuracy: the two most accurate models use less than half the tokens of GLM-5.3, which hits its 131k-token limit without a parseable answer in 99 of 585 runs.
- **Error sources are shared across models, and large possibility spaces separate them.** For most models errors rise with the size of the gold answer; the two leaders err *less* on the larger half.
- **Hard is largely beyond direct answering, and a sandbox helps but does not solve it.** No model exceeds 16% directly. A code sandbox lifts every model by 16.7–40.1 points, yet the best reaches only 55.7% (57.0% with the library).
- **The logic modeling library does not consistently help.** Adding it changes Hard accuracy by −6.0 to +10.6 points: five models improve and two decline. Using it well requires understanding its modeling language.
- **Models reach for plain Python, not solvers.** With a sandbox alone, six of seven models mostly write plain Python that enumerates candidate assignments and filters them with handwritten rules.

## 📦 What Is Released

A benchmark whose every item is public stops measuring reasoning once its items enter training data. So we release a fixed, uniformly sampled **30% of the items of each tier**, together with everything needed to run the evaluation, and keep the rest private:

- **Public items with gold answers**: `data/seed.json`, `data/base.json`, and `data/hard.json`. Samples are item-level: all sub-questions of a selected Base item are kept together.
- **The direct-answer prompt and semantic scorer** in `evaluate.py`: it writes the prompts used for the paper and scores predictions with an LLM judge that checks semantic equivalence, reporting item and sub-question accuracy.
- **The complete Hard agent evaluation** in `hard_agent/`: the exact task payloads of both conditions for every public Hard item, the workspace builder, a Docker runner, the answer collector, the rollout summarizer, and the full 11-module `translation` library and `MANUAL.md` given to the library condition.
- **The selection procedure** in `build_public_subset.py` and `selection_manifest.json`, with SHA-256 commitments to the full corpus, so the split can be audited.

| Tier | Full items | Public items | Public sub-questions | Private items |
|---|---:|---:|---:|---:|
| Seed | 228 | 69 | 69 | 159 |
| Base | 195 | 59 | 274 | 136 |
| Hard | 100 | 30 | 30 | 70 |

The remaining 70% of the items and the expert-validated formal translations of **all** items, public ones included, are held out. Scores on the public subset cannot reproduce the full-benchmark numbers above; to be evaluated on the full benchmark, see [Evaluate Your Model](#-evaluate-your-model).

## 🚀 Quick Start

Requires Python 3.10 or newer. `evaluate.py` uses only the standard library; the optional solver backends of the Hard library are listed in `hard_agent/requirements.txt`.

### 1. Produce prompts

```bash
python evaluate.py prompts --tier seed --output seed_prompts.jsonl
python evaluate.py prompts --tier base --output base_prompts.jsonl
python evaluate.py prompts --tier hard --output hard_prompts.jsonl
```

Each line contains `id` and the direct-answer `prompt`. Send each prompt to the model being evaluated, then save one JSONL record per item:

```json
{"id": 12, "answers": ["第1小问答案", "第2小问答案"]}
```

The number and order of `answers` must match `question[]`; Seed IDs are strings. Answers may also be JSON scalars, arrays, or objects, which the scorer serializes as JSON text before judging. Mark a model refusal as `{"id": 12, "status": "refusal"}`. Missing and invalid predictions count as wrong; refusals are reported separately and excluded from the denominator, as in the paper.

### 2. Score predictions

The paper uses a semantic-equivalence LLM judge, not literal string matching; `evaluate.py` judges all sub-questions of an item together. The endpoint must implement OpenAI-compatible Chat Completions:

```bash
export OPENAI_BASE_URL="https://YOUR-ENDPOINT/v1"
export OPENAI_API_KEY="YOUR-KEY"
python evaluate.py score --tier base --predictions base_predictions.jsonl \
  --judge-model YOUR-JUDGE-MODEL --output base_scores.json
```

The scorer writes item accuracy (all sub-questions correct) and sub-question accuracy, plus a `*.verdicts.jsonl` audit file. To aggregate already checked verdicts without an API call, pass `--verdicts verdicts.jsonl` instead of `--judge-model`, one line per item such as `{"id": 12, "matches": [true, false]}`. A missing verdict for a valid prediction raises an error rather than being scored wrong. `--judge-repeats 2` reproduces the repeated judging and agreement report used for the Hard agent conditions.

### 3. Evaluate Hard agents

Condition `c2` gives the agent only the task and case files; `c3` also gives it the complete `translation` library and `MANUAL.md`. Run the same agent model and settings in both. From the repository root, prepare isolated workspaces, run each agent in its own Docker container, collect its `answer.json`, and score each rollout:

```bash
python hard_agent/prepare_workspaces.py --output-root /tmp/hyperlogic-agent --rollouts 3

for condition in c2 c3; do
  python hard_agent/run_agents.py \
    --workspaces /tmp/hyperlogic-agent --condition "$condition" \
    --image YOUR_AGENT_IMAGE --env YOUR_API_KEY_VAR --concurrency 4 \
    -- YOUR_AGENT_COMMAND task.md

  for rollout in 1 2 3; do
    python hard_agent/collect_predictions.py \
      --workspaces /tmp/hyperlogic-agent --condition "$condition" \
      --rollout "$rollout" --output "/tmp/${condition}-r${rollout}.jsonl"
    python evaluate.py score --tier hard \
      --predictions "/tmp/${condition}-r${rollout}.jsonl" \
      --judge-model YOUR-JUDGE-MODEL --judge-repeats 2 \
      --output "/tmp/${condition}-r${rollout}-scores.json"
  done

  python hard_agent/summarize_rollouts.py \
    "/tmp/${condition}-r1-scores.json" \
    "/tmp/${condition}-r2-scores.json" \
    "/tmp/${condition}-r3-scores.json" \
    --output "/tmp/${condition}-summary.json"
done
```

The Docker image must contain your agent CLI and its dependencies. Each agent must write `answer.json` in its workspace, keyed by the question ID in `case/questions.json`. The runner defaults to a 2-CPU, 10-GB, three-hour limit per workspace. **Mount only one prepared workspace per container**: running agents directly on a checkout lets them read the public gold file. See [`hard_agent/README.md`](hard_agent/README.md) for workspace contents, answer shapes, and runner options.

## 🗂️ Data Format

Each file in `data/` is a JSON list of items. Every item has `id`, `background`, aligned `question[]` and `answer[]` arrays, and optionally `title`. A Seed item has one question and a content-derived `seed-...` ID; Base and Hard items keep their integer IDs. No formal translations, author information, or private items are included.

`build_public_subset.py` draws an independent uniform sample without replacement for each tier with `random.Random("2027:<tier>")`, releasing `ceil(0.30 × full item count)` items. The full normalized corpus is sorted by item ID before sampling, and the selected items keep that order. `selection_manifest.json` records the method, SHA-256 commitments to each full tier, and the hashes of the public subsets.

The public items are also on Hugging Face as three configurations, `seed`, `base`, and `hard`, each with a single `test` split:

```python
from datasets import load_dataset

base = load_dataset("tencent/HyperLogic", "base", split="test")   # also "seed" and "hard"
```

## 📁 Repository Structure

```
hyperlogic/
├── data/                      public items with gold answers
│   ├── seed.json              69 items
│   ├── base.json              59 items · 274 sub-questions
│   └── hard.json              30 items
├── evaluate.py                direct-answer prompts and semantic-equivalence scoring
├── hard_agent/                Hard agent evaluation
│   ├── payloads/{c2,c3}/      exact task.md and case files per public Hard item
│   ├── library/translation/   logic modeling library given to c3 (Z3, BDD, SAT, Ganak backends)
│   ├── MANUAL.md              library manual given to c3
│   ├── prepare_workspaces.py  builds isolated per-rollout workspaces
│   ├── run_agents.py          Docker runner with resource limits
│   ├── collect_predictions.py turns answer.json files into predictions
│   └── summarize_rollouts.py  mean and spread across rollouts
├── build_public_subset.py     the 30% selection procedure
├── selection_manifest.json    selection rule and corpus commitments
└── assets/                    figures for this README
```

## 📮 Evaluate Your Model

The held-out 70% of HyperLogic is not public, so that results on it stay free of contamination. To have a model evaluated on the full benchmark, [contact us](#-contact), and we will run it under the same protocol as the leaderboard.

## 📜 License

This repository is licensed under the [Apache-2.0 License](LICENSE).

## 📝 Citation

```bibtex
@article{zhang2026hyperlogic,
  title   = {HyperLogic: A Hard, Forward-Authored Chinese Logical Reasoning
             Benchmark with Execution-Derived Answers},
  author  = {Zhang, Ming and Peng, Qiyuan and Wei, Yinxi and Shen, Yujiong and
             Tan, Kexin and Wang, Yuhui and Xiang, Zhenghao and Ye, Junjie and
             Yin, Zhangyue and Xi, Zhiheng and Dou, Shihan and Wang, Weikang and
             Zhang, Yuhao and Gui, Tao and Yang, Ruizhi and Zhang, Qi and
             Huang, Xuanjing and Chen, Alex and Pan, Maxm},
  journal = {arXiv preprint arXiv:2605.19597},
  year    = {2026}
}
```

## 📧 Contact

For questions or evaluation requests, please open an issue or email Ming Zhang (mingzhang23@m.fudan.edu.cn) or Maxm Pan (maxmpan@tencent.com).

<div align="center">
<sub>HyperLogic · Fudan University · Hunyuan Team, Tencent</sub>
</div>
