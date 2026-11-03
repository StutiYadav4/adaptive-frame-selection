# Adaptive Frame Selection for Efficient Video Understanding

**Question-conditioned, diversity-aware frame selection for Video Question Answering using a frozen Qwen2.5-VL reader.**

This project presents an efficient frame-selection framework for Video Question Answering (VideoQA) on the **NExT-QA dataset**. It combines semantic relevance, visual coverage and temporal coverage through greedy submodular maximization, with an additional adaptive frame-budget strategy.

The objective is to select a compact set of informative video frames while maintaining answer accuracy and reducing redundant visual input.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Proposed Methodology](#proposed-methodology)
- [Mathematical Formulation](#mathematical-formulation)
- [Adaptive Frame Budgeting](#adaptive-frame-budgeting)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Ablation Studies](#ablation-studies)
- [Computational Efficiency](#computational-efficiency)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the Experiments](#running-the-experiments)
- [Reproducibility and Implementation Notes](#reproducibility-and-implementation-notes)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)
- [Contributors](#contributors)

---

## Overview

Video Question Answering requires a model to understand video content and answer questions involving objects, actions, events and temporal relationships.

A straightforward approach processes uniformly sampled video frames using a vision-language model. However, uniform sampling may miss important events, while selecting frames only according to question relevance can result in substantial visual redundancy.

This project investigates a more structured approach to frame selection.

The proposed framework:

- Extracts candidate video frames at 1 FPS.
- Computes question-conditioned relevance scores.
- Encourages visual diversity and coverage.
- Incorporates temporal coverage to represent different portions of a video.
- Uses greedy submodular optimization to select informative frames.
- Explores adaptive frame budgeting based on the information available for each question.
- Evaluates selected frames using a frozen Qwen2.5-VL reader.

The experiments are conducted on **NExT-QA**, with a video-level development/test split.

## Problem Statement

Given a video containing \(N\) candidate frames and a natural-language question, the objective is to select a subset of at most \(K\) frames that provides sufficient visual information for answering the question.

The challenge is to balance three competing considerations:

1. **Relevance:** Selected frames should be semantically related to the question.
2. **Visual Coverage:** Selected frames should represent diverse visual information rather than repeatedly capturing similar content.
3. **Temporal Coverage:** Selected frames should represent different portions of the video timeline.

A good selection strategy should maximize useful visual information while keeping the number of frames small.

---

## Proposed Methodology

The framework consists of the following stages.

### 1. Video Preprocessing and Feature Extraction

Videos are decoded at a standardized sampling rate of **1 frame per second (1 FPS)**.

The extracted frames are processed to obtain visual representations, which are stored in an on-disk feature cache.

This preprocessing stage is performed once so that different selection strategies can reuse the same candidate frames and cached features without repeatedly decoding the original videos.

### 2. Question-Conditioned Relevance Scoring

For a given question, each candidate frame receives a semantic relevance score based on its relationship with the question.

The relevance score is normalized to:

\[
r_i \in [0,1]
\]

A higher score indicates stronger semantic relevance between the frame and the question.

However, selecting only the highest-scoring frames may result in multiple visually similar frames. Therefore, relevance is combined with coverage-based objectives.

### 3. Visual Coverage

Visual coverage encourages the selected frames to collectively represent the visual information available throughout the candidate frame set.

Rather than simply choosing the most relevant frames, the objective rewards frames that cover visual content not already represented by the selected subset.

This reduces redundancy and promotes complementary frame selection.

### 4. Temporal Coverage

Temporal coverage encourages the selection of frames that collectively represent different portions of the video timeline.

A temporal similarity kernel measures the relationship between frame positions, allowing the selection process to account for temporal distribution.

### 5. Greedy Submodular Frame Selection

The proposed method combines relevance, visual coverage and temporal coverage into a unified objective function.

A greedy optimization algorithm iteratively selects the frame providing the highest marginal improvement to the objective.

The resulting selection order supports evaluation at different frame budgets using nested subsets.

---

## Mathematical Formulation

Let:

- \(V=\{1,2,\ldots,N\}\) denote the candidate frame set.
- \(S\subseteq V\) denote the selected frame subset.
- \(r_i\) denote the question relevance score of frame \(i\).
- \(s(i,j)\) denote visual similarity between frames \(i\) and \(j\).
- \(\omega_j\) denote the importance weight of candidate frame \(j\).
- \(\kappa(i,j)\) denote temporal similarity between frames \(i\) and \(j\).

The proposed objective is:

\[
\begin{aligned}
F(S) ={}&
\frac{w_{\mathrm{rel}}}{k_{\mathrm{ref}}}
\sum_{i\in S}r_i\\
&+w_{\mathrm{div}}\sum_{j=1}^{N}
\omega_j\max_{i\in S}s(i,j)\\
&+\frac{w_{\mathrm{time}}}{N}
\sum_{j=1}^{N}\max_{i\in S}\kappa(i,j)
\end{aligned}
\]

where:

- \(w_{\mathrm{rel}}\) controls the relevance contribution.
- \(w_{\mathrm{div}}\) controls visual coverage.
- \(w_{\mathrm{time}}\) controls temporal coverage.
- \(k_{\mathrm{ref}}\) is the reference frame budget.

The maximum over an empty selected set is defined as zero.

The visual coverage weights are relevance-aware and normalized:

\[
\omega_j \propto
c+(1-c)r_j
\]

where \(c\) is the coverage floor.

The objective combines a modular relevance term with two facility-location-style coverage terms.

Each component is non-negative, monotone and submodular. Consequently, greedy maximization under a cardinality constraint has the classical approximation guarantee:

\[
F(S_{\mathrm{greedy}})
\geq
(1-1/e)F(S_{\mathrm{optimal}})
\]

The greedy selection procedure chooses:

\[
i^* =
\arg\max_{i\notin S}
\left[
F(S\cup\{i\})-F(S)
\right]
\]

at each iteration.

Because the relevance term uses a fixed reference budget, the greedy selection sequence is nested: smaller-budget selections are prefixes of larger-budget selections.

---

## Adaptive Frame Budgeting

A fixed frame budget does not necessarily work equally well for every question.

Some questions can be answered using a small number of informative frames, while others require additional visual context.

To investigate this, the framework includes adaptive frame-budget strategies.

The adaptive approach uses the marginal gains obtained during greedy selection to determine when additional frames provide diminishing returns.

The implementation explores:

- **Gain-based adaptive selection:** Uses the relative marginal gain of additional frames.
- **Entropy-based adaptive selection:** Uses an entropy-based criterion to guide the frame budget.

The adaptive strategy aims to allocate visual context according to the information requirements of individual questions instead of relying entirely on one fixed frame budget.

---

## Experimental Setup

### Dataset

The experiments are conducted exclusively on **NExT-QA**, a video question-answering benchmark designed to evaluate temporal and causal reasoning in videos.

The evaluated subset contains:

| Split | Number of Questions |
|---|---:|
| Development | 145 |
| Test | 436 |
| Total | 581 |

The development/test split is performed at the video level to avoid having videos from the same source appear in both splits.

### Vision-Language Reader

The framework uses a frozen **Qwen2.5-VL-3B** reader for answer prediction.

The reader evaluates the candidate answer options using option-letter logits in a forward pass, rather than generating free-form answers.

Keeping the reader frozen allows the experiments to focus on the effect of frame-selection strategies.

### Frame Selection Baselines

The proposed method is compared against the following strategies:

| Method | Description |
|---|---|
| Uniform | Uniformly samples frames across the video. |
| Random | Randomly selects candidate frames. |
| Top-K | Selects frames with the highest question relevance scores. |
| MMR | Uses Maximal Marginal Relevance to balance relevance and diversity. |
| OURS | Combines relevance, visual coverage and temporal coverage using greedy submodular optimization. |
| OURS Adaptive | Extends the proposed method with adaptive frame-budget selection. |
| Oracle-style | Evaluates the best-performing subset among eight candidate subsets using reader outputs. |

---

## Experimental Results

### Overall Accuracy

At a fixed frame budget of \(K=4\), the reported test accuracies are:

| Selection Strategy | Test Accuracy |
|---|---:|
| Uniform | 68.8% |
| Top-K | 67.7% |
| OURS | **70.4%** |
| Oracle-style | 82.1% |

The proposed method achieves an improvement of **1.6 percentage points** over uniform sampling at the same frame budget.

The results demonstrate that relevance-only selection does not necessarily outperform uniform sampling. Incorporating visual and temporal coverage helps select more complementary frames.

The oracle-style result is a best-of-eight comparison, not an exhaustive optimum.

### Ablation Studies

Ablation experiments investigate the contribution of individual components of the proposed objective.

The evaluated variants include:

- Removing visual coverage.
- Removing relevance weighting.
- Removing or modifying diversity selection through MMR.
- Random-subset selection as a control.

Paired statistical comparisons and bootstrap confidence intervals are used to analyze the results.

The visual-coverage ablation is the only component ablation that survives the stated multiple-comparison correction.

### Needle-in-a-Video Experiment

A synthetic needle-in-a-video experiment is conducted to evaluate whether the selection strategy can identify a specific visual cue within a video.

The experiment examines:

- Whether the selected subset contains the needle frame.
- How the needle hit rate changes with the frame budget.
- How the reader's answer accuracy changes with the selected frames.

The results illustrate the importance of selecting frames containing the relevant visual evidence, particularly at smaller frame budgets.

---

## Computational Efficiency

The framework is designed to minimize the additional computational overhead introduced by frame selection.

The reported measurements are:

| Component | Approximate Time |
|---|---:|
| Fixed-budget greedy selection | 0.62 ms/question |
| Adaptive selection | 2.23 ms/question |
| Reader latency (fixed-budget methods) | Approximately 0.296 seconds |
| Reader latency (adaptive method) | Approximately 0.317 seconds |

At a fixed frame budget, the reader receives approximately the same number of visual tokens across selection strategies.

Small differences in total latency arise from the additional frame-selection overhead.

The fixed-budget greedy selection stage adds very little overhead compared with the reader inference stage.

---

## Repository Structure

The project is organized into preprocessing, feature extraction, frame selection, adaptive selection, evaluation and analysis components.

```text
afs-project/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── run_all.sh
│
├── configs/
│   └── default.yaml
│
├── src/
│   └── afs/
│       ├── data.py
│       ├── features.py
│       ├── ctx.py
│       ├── selectors.py
│       ├── submodular.py
│       ├── adaptive.py
│       ├── oracle.py
│       ├── reader.py
│       ├── evaluate.py
│       └── utils.py
│
├── scripts/
│   ├── 00_download_small_nextqa.py
│   ├── 01_build_subset.py
│   ├── 02_cache_features.py
│   ├── 03_run_selector.py
│   ├── 04_run_oracle.py
│   ├── 05_tune_diversity.py
│   ├── 07_needle_probe.py
│   ├── 08_analyze.py
│   └── 09_demo.py
│
├── tests/
│   └── test_core.py
│
├── data/
│   └── subsets/
│
├── cache/
│
└── results/
```

Generated datasets, cached features, reader outputs and experimental results are stored separately and are not required to be committed to GitHub.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/StutiYadav4/adaptive-frame-selection.git

cd adaptive-frame-selection
```

### 2. Create a Conda Environment

```bash
conda create -n afs python=3.11 -y
conda activate afs
```

### 3. Install PyTorch

For a compatible CUDA 12.1 environment:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Choose the appropriate PyTorch installation for your hardware and CUDA version.

### 4. Install Project Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install the Project

```bash
pip install -e .
```

Alternatively, configure the Python path:

```bash
export PYTHONPATH=$PWD/src
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "$PWD/src"
```

### 6. Run Unit Tests

```bash
python tests/test_core.py
```

The core tests are designed to run without requiring GPU inference or downloading the full dataset.

---

## Running the Experiments

The following commands illustrate the NExT-QA experimental workflow.

Update dataset and model paths in `configs/default.yaml` before execution.

### Phase 1: Data Preparation and Feature Extraction

```bash
python scripts/00_download_small_nextqa.py --n_videos 100

python scripts/01_build_subset.py --kind nextqa

python scripts/02_cache_features.py --subset nextqa
```

### Phase 2: Baseline Evaluation

```bash
python scripts/03_run_selector.py \
    --subset nextqa \
    --split test \
    --selectors uniform random stride topk mmr
```

### Phase 3: Oracle-Style Comparison

```bash
python scripts/04_run_oracle.py \
    --subset nextqa \
    --split test

python scripts/04_run_oracle.py \
    --subset nextqa \
    --split test \
    --control
```

The control experiment evaluates the best-performing random candidate subsets.

### Phase 4: Proposed Frame Selection

Tune diversity-related parameters on the development split:

```bash
python scripts/05_tune_diversity.py \
    --subset nextqa \
    --K 8
```

Run the proposed selector:

```bash
python scripts/03_run_selector.py \
    --subset nextqa \
    --split test \
    --selectors ours
```

### Phase 5: Adaptive Frame Selection

```bash
python scripts/03_run_selector.py \
    --subset nextqa \
    --split test \
    --selectors ours_adaptive ours_adaptive_entropy \
    --ks 4 8 16
```

### Phase 6: Ablation Experiments

```bash
python scripts/03_run_selector.py \
    --subset nextqa \
    --split test \
    --selectors ours_no_div ours_no_time ours_no_rel ours_unweighted \
    --ks 4 8 16
```

### Phase 7: Needle Experiment and Analysis

```bash
python scripts/07_needle_probe.py \
    --subset nextqa \
    --trials 100 \
    --use_reader

python scripts/08_analyze.py \
    --subset nextqa \
    --split test
```

### Phase 8: Visualization and Demo

```bash
python scripts/09_demo.py \
    --subset nextqa \
    --K 8
```

**Note:** These commands assume the corresponding scripts and options are present in the repository. Check individual script arguments and configuration settings before running them.

---

## Reproducibility and Implementation Notes

The experimental pipeline follows several principles to maintain consistency:

1. Videos are decoded at 1 FPS and their extracted features are cached.
2. The same candidate frames are reused across selection strategies.
3. Development and test sets are split at the video level.
4. Accuracy-related hyperparameter tuning is performed using the development split.
5. Adaptive thresholds are calibrated label-free using test-question features. This is feature-based transductive calibration and does not use test answers.
6. Reader outputs are cached by question and selected frame set, allowing repeated experiments to reuse previous predictions.
7. Bootstrap confidence intervals and paired statistical comparisons are used to analyze experimental differences.

### Generated Outputs

The analysis pipeline produces experimental artifacts such as:

| Output | Description |
|---|---|
| `acc_vs_K.png` | Accuracy versus frame budget. |
| `pareto_gflops.png` | Accuracy and computational cost comparison. |
| `pareto_gflops_total.png` | Cost comparison including encoder computation. |
| `pareto_latency.png` | Accuracy versus wall-clock latency. |
| `summary.csv` | Overall experimental metrics. |
| `ablation.csv` | Component ablation results. |
| `per_category_K8.csv` | Performance across question categories. |
| `paired_ours_vs_uniform.csv` | Paired comparison with uniform sampling. |
| `adaptive_K_hist.png` | Distribution of adaptive frame budgets. |
| `needle_hit_rate.png` | Needle-frame retrieval performance. |
| `demo_*.png` | Selected frames and their relevance and coverage contributions. |

These files are generated during experimentation and are not necessarily included in the repository.

---

## Limitations

- Experiments are conducted on NExT-QA; generalization to other video benchmarks remains to be evaluated.
- The oracle-style comparison considers only eight candidate subsets and is not an exhaustive upper bound.
- Reader-based candidate selection can benefit from selection on reader variability, so the oracle-style result should be interpreted cautiously.
- Computational cost estimates based on GFLOPs are analytical approximations and do not fully capture all hardware-dependent operations.
- Adaptive thresholds use label-free calibration on test-question features, which should be distinguished from a strictly inductive evaluation protocol.
- Results depend on the evaluated reader, feature representations, video sampling rate and experimental configuration.

---

## Future Work

Potential extensions include:

- Evaluating the framework on additional video question-answering benchmarks.
- Exploring learned adaptive frame-budget policies.
- Incorporating motion-aware and event-aware representations.
- Improving the efficiency of feature extraction and similarity computation.
- Evaluating the framework with different vision-language model architectures.
- Investigating stronger accuracy-compute trade-offs under smaller visual budgets.

---

---

## Contributors

This project was collaboratively developed by:

- **Stuti Yadav**
- **Anuj Trivedi**
- **Iha Goyal**

**Institution:** Indian Institute of Technology Jodhpur

---

*Adaptive Frame Selection for Efficient Video Understanding — NExT-QA Experimental Study.*
