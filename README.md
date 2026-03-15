# Fake News Detection with Recurrent Neural Networks
Final project for the UPC postgraduate in Deep learning and Aritificial Intelligence

**Postgraduate course: Artificial Intelligence with Deep Learning (UPC School, 2026)**

| | |
|---|---|
| **Team members** | Valentina Martínez · Selena Rodríguez · Marc Humet |
| **Advisor** | Pol Caselles |
| **Dataset** | [ISOT Fake News Dataset](https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/index.php) (~44K news articles) |
| **Task** | Binary classification: Fake (0) vs True (1) |

## Table of Contents

- [Fake News Detection with Recurrent Neural Networks](#fake-news-detection-with-recurrent-neural-networks)
  - [Table of Contents](#table-of-contents)
  - [Motivation](#motivation)
  - [Dataset](#dataset)
  - [Preprocessing Pipeline](#preprocessing-pipeline)
    - [Text Cleaning (train/val only)](#text-cleaning-trainval-only)
    - [Vocabulary Exclusions](#vocabulary-exclusions)
    - [Test Set Tokenization](#test-set-tokenization)
    - [Output Artifacts](#output-artifacts)
  - [Model Architectures](#model-architectures)
  - [Experiments](#experiments)
    - [Experiment 1: Bidirectional LSTM](#experiment-1-bidirectional-lstm)
      - [Hypothesis](#hypothesis)
      - [Experiment Setup](#experiment-setup)
      - [Results](#results)
      - [Conclusions](#conclusions)
    - [Experiment 2: Unidirectional LSTM](#experiment-2-unidirectional-lstm)
      - [Hypothesis](#hypothesis-1)
      - [Experiment Setup](#experiment-setup-1)
      - [Results](#results-1)
      - [Conclusions](#conclusions-1)
    - [Experiment 3: Leaky Word Removal Sweep](#experiment-3-leaky-word-removal-sweep)
      - [Hypothesis](#hypothesis-2)
      - [Experiment Setup](#experiment-setup-2)
      - [Results](#results-2)
      - [Conclusions](#conclusions-2)
  - [How to Run](#how-to-run)
    - [Prerequisites](#prerequisites)
    - [1. Preprocessing](#1-preprocessing)
    - [2. Training](#2-training)
    - [3. EDA (optional, local)](#3-eda-optional-local)
  - [Repository Structure](#repository-structure)

## Motivation

Fake news is a growing problem in the digital age. Automated detection systems can help flag potentially misleading content before it spreads. This project explores how well recurrent neural networks (RNN, GRU, LSTM) can distinguish between fake and legitimate news articles using only their text content, and investigates whether these models learn genuine linguistic patterns or rely on spurious signals like topic-specific vocabulary.

## Dataset

The [ISOT Fake News Dataset](https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/index.php) contains ~44,900 articles:
- **~21,400 true** articles sourced from Reuters
- **~23,500 fake** articles from various unreliable sources flagged by Politifact and Wikipedia

After removing exact-text duplicates (`drop_duplicates(subset="text", keep=False)`), the dataset shrinks to **33,506 unique articles** (12,527 fake + 20,979 true). The data is then split 90/10 into train+val and test, and train+val is further split 90/10 into train and val. All splits are stratified by label. The train/test split happens **before** any text cleaning or vocabulary building to prevent data leakage.

| Split | Total | Fake | True |
|-------|-------|------|------|
| Train | 27,139 | 10,146 | 16,993 |
| Val | 3,016 | 1,128 | 1,888 |
| Test | 3,351 | 1,253 | 2,098 |

**Important design choice:** The training and validation sets are cleaned (lowercase, stopword removal, lemmatization), but the test set is intentionally tokenized **without cleaning** (no stopword removal, no lemmatization, no `light_clean`). This simulates a realistic deployment scenario where input text arrives in its raw form, and tests whether the model generalizes beyond the cleaned distribution.

## Preprocessing Pipeline

Implemented in `ISOT_Preprocessing_v3.ipynb`:

| Parameter | Value |
|-----------|-------|
| Vocab size (config) | 75,000 |
| Actual vocab (after min_freq filtering) | ~46,864 |
| Min frequency | 2 |
| Max sequence length | 512 |
| Special tokens | `<pad>` (ID 0), `<unk>` (ID 1) |

### Text Cleaning (train/val only)

The `light_clean` function applies the following steps before tokenization:

| Step | What it removes | Example |
|------|----------------|---------|
| Reuters prefix | `"WASHINGTON (Reuters) - ..."` before article body | Source attribution |
| HTML tags | `<p>`, `<br>`, etc. | Markup artifacts |
| URLs & emails | `https://...`, `user@domain.com` | Non-content links |
| @handles | `@realdonaldtrump` | Social media mentions |
| Bracketed content | `[video]`, `(tweets)`, `<iframe>` | Embedded media references |
| Credits & domains | `getty`, `flickr`, `pic.twitter.com`, `youtube` | Media source noise |
| Slash tokens | `somodevilla/getty`, `wong/reuters` | Photo credits |
| Colon tokens | `below:featured`, `via:` | Navigation artifacts |
| Digit-letter glue | `2017the` → `2017 the` | OCR/scraping errors |
| `featured` patterns | `below:featured` | CMS artifacts |

After cleaning: lowercase, regex tokenization (`[a-z]+(?:'[a-z]+)?`), stopword removal (NLTK English), and WordNet lemmatization.

### Vocabulary Exclusions

Three topic-specific words are explicitly excluded from the vocabulary to reduce source-leakage signals: `rohingya`, `catalan`, `catalonia`. These appear almost exclusively in Reuters (true) articles and could act as spurious shortcuts.

### Test Set Tokenization

The test set receives **no `light_clean`, no stopword removal, and no lemmatization** — only the raw title + text is tokenized with the regex pattern. This intentional mismatch tests whether models generalize to uncleaned text.

### Output Artifacts

Saved to Google Drive (`artifacts/`): `train_ids.pt`, `val_ids.pt`, `test_ids.pt`, label tensors, vocabulary mappings (`stoi.json`, `itos.json`), and `summary.json`.

## Model Architectures

All models share the same structure: Embedding (128-dim) + recurrent encoder + linear classifier outputting a single logit. Loss function is `BCEWithLogitsLoss`.

| Model | Status | Notes |
|-------|--------|-------|
| GRU | Implemented | Bidirectional, single layer |
| RNN | Implemented | Bidirectional, single layer |
| LSTM | Implemented | Supports multi-layer + dropout |
| Transformer | Stub | Not yet implemented |

## Experiments

---

### Experiment 1: Bidirectional LSTM

#### Hypothesis

A bidirectional LSTM should capture both forward and backward context across the full article, learning robust features for distinguishing fake from true news even when the test data is uncleaned.

#### Experiment Setup

**Model architecture:** LSTM (bidirectional)

| Hyperparameter | Value |
|----------------|-------|
| Embedding dim | 128 |
| Hidden size | 128 |
| Bidirectional | True |
| Num layers | 1 |
| Dropout | 0.4 |
| Learning rate | 0.001 |
| Batch size | 32 |
| Max epochs | 10 |
| Early stopping patience | 3 |
| Seed | 42 |

#### Results

Trained for 6 epochs (best val acc: 0.9738 at epoch 3).

**Test set evaluation** (3,351 samples):

| Metric | Fake (0) | True (1) | Macro Avg |
|--------|----------|----------|-----------|
| Precision | 0.9348 | 0.9254 | 0.9301 |
| Recall | 0.8699 | 0.9638 | 0.9168 |
| F1-score | 0.9012 | 0.9442 | 0.9227 |

**Test accuracy:** 0.9287 | **Weighted F1:** 0.9281

**Training curves:**

![Training curves](training_curves.png)

**Confusion matrix (test set):**

![Confusion matrix](confusion_matrix.png)

#### Conclusions

The bidirectional LSTM achieves ~93% test accuracy on uncleaned text, demonstrating strong generalization. The model favors True recall (0.96) over Fake recall (0.87) — it is more conservative about labeling articles as fake. The gap between train accuracy (~99%) and val/test accuracy suggests some overfitting after epoch 3, which early stopping mitigates by restoring the best checkpoint.

---

### Experiment 2: Unidirectional LSTM

#### Hypothesis

Removing the backward pass should degrade performance, but the forward-only model should still learn useful sequential patterns from the article text.

#### Experiment Setup

Same hyperparameters as Experiment 1, except `Bidirectional = False`.

#### Results

**Test accuracy:** ~0.66

| Metric | Fake (0) | True (1) |
|--------|----------|----------|
| Precision | 0.53 | — |
| Recall | 0.93 | 0.50 |

#### Conclusions

The unidirectional model collapsed to a near-degenerate decision boundary, predicting almost everything as Fake. True recall of ~0.50 means it is essentially flipping a coin on true articles. This reveals that a forward-only LSTM reading 512 tokens loses early signals by the final hidden state — the vanishing gradient problem in practice. The backward pass in the bidirectional model independently captures signals from the end of articles (where conclusions and key claims appear), explaining the massive performance gap (0.93 vs 0.66).

The train/test preprocessing mismatch amplifies this: the unidirectional model overfits to surface-level patterns in cleaned training text and defaults to Fake when encountering uncleaned test input it cannot recognize.

---

### Experiment 3: Leaky Word Removal Sweep

#### Hypothesis

Topic-specific words that appear almost exclusively in true (Reuters) articles — such as `myanmar`, `rohingya`, `hariri` — could act as spurious shortcuts. If the model relies on them, masking these words at test time should cause a measurable drop in performance.

#### Experiment Setup

Rather than retraining the model with different word lists (which introduces variance from random initialization), we:
1. Train the model **once** with a fixed seed (42)
2. Evaluate the frozen model on the test set with the top-N leaky words replaced by `<pad>`, for N in [0, 5, 10, 15]

The leaky words are sourced from `most_true_words.csv` — words with the highest true/fake ratio (e.g., `myanmar` appears in 3,084 true articles but only 4 fake ones).

#### Results

**Bidirectional LSTM:** No measurable change across all masking levels. Fake recall and True recall stayed flat. Masking 15 words from a 45,000-word vocabulary (~0.03%) is too small a perturbation. The bidirectional model aggregates all 512 tokens through its hidden state and learned distributed style/structure features rather than relying on individual topic words.

**Unidirectional LSTM:** Accuracy dropped slightly but per-class recalls were near-flat. The model's fundamental problem is its degenerate bias toward Fake, not reliance on a handful of topic tokens. True recall did decrease marginally (0.4990 to 0.4967) while Fake recall stayed fixed — consistent with the theory that fake articles never contained these true-biased words in the first place.

#### Conclusions

The bidirectional LSTM is robust to leaky word removal, suggesting it learned genuine distributional features rather than topic shortcuts. The main finding from these experiments is that **bidirectionality is the dominant factor** in this task — far more important than vocabulary-level interventions. The backward pass captures signals from article conclusions that the forward-only model loses after 512 time steps.

---

## How to Run

### Prerequisites

- Google account (for Colab + Drive)
- [Kaggle API key](https://www.kaggle.com/docs/api) configured

### 1. Preprocessing

Open `ISOT_Preprocessing_v3.ipynb` in Google Colab. It will:
- Download the ISOT dataset from Kaggle
- Remove duplicates, split into train/val/test
- Clean and tokenize train/val (test left uncleaned)
- Build vocabulary, convert to padded ID tensors
- Save artifacts to `Google Drive/aidl-final-project/artifacts/`

### 2. Training

Open `ISOT_Training.ipynb` in Google Colab:
1. Set `MODEL_TYPE` in the Configuration cell (`"RNN"`, `"GRU"`, `"LSTM"`, or `"Transformer"`)
2. Adjust hyperparameters as needed
3. Run all cells

Results are saved to `Google Drive/aidl-final-project/outputs/{model_type}/`.

### 3. EDA (optional, local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd dataset
streamlit run eda_app.py
# Expects Fake.csv and True.csv in the dataset/ directory
```

## Repository Structure

```
.
├── ISOT_Preprocessing_v3.ipynb   # Stage 1: download, deduplicate, clean, tokenize, save artifacts
├── ISOT_Training.ipynb           # Stage 2: train & evaluate models
├── dataset/
│   ├── eda_app.py                # Streamlit EDA app
│   ├── Fake.csv                  # (not tracked) fake news articles
│   └── True.csv                  # (not tracked) true news articles
├── docs/                         # Presentations and reference docs
├── most_true_words.csv           # Words with highest true/fake ratio
├── most_fake_words.csv           # Words with highest fake/true ratio
├── training_curves.png           # Loss & accuracy plots (LSTM)
├── confusion_matrix.png          # Test set confusion matrix (LSTM)
├── requirements.txt
└── README.md
```
