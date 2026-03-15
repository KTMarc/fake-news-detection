# fake-news-detection
Final project for the UPC postgraduate in Deep learning and Aritificial Intelligence

## Training Notebook Evolution

### 1. Split in 2 notebooks
It was not easy to work with a single file, so we decided to split it into two and use Google Drive to pass output from one as input to the other.
Notebook created from scratch. Pipeline only for GRU: setup + mount Drive, load artifacts (train/val), a single `GRUClassifier`, training loop, loss/accuracy plots, and save to `outputs/gru/`.
### 3. Adding more models and configuration
Major restructuring to support multiple models. Added `MODEL_TYPE` selector variable. Hyperparameters are split into shared, RNN/GRU/LSTM-specific, and Transformer-specific. Added stubs (`NotImplementedError`) for RNN, LSTM, and Transformer. Created a `MODELS` dictionary to dispatch to the right class. Output path became dynamic (`outputs/{model_type}`).
Same for preprocessing Notebook, a config cell would provide values like vocabulary size, or max length.

### 4. Added RNN and LSTM
Implemented `RNNClassifier` and `LSTMClassifier` (replacing the stubs). LSTM includes `num_layers` and `dropout` support with detailed shape comments. Default `MODEL_TYPE` changed to `"LSTM"`. The performance on these models is still really good. We decide we don't need a transformer and continue investigating our dataset.

### 5. Add Early stop, use test for confusion matrix, new MD reports
Biggest change. Added early stopping (patience + accuracy threshold). Loaded and evaluated on the **test set** (previously unused). Added sklearn classification report (precision/recall/F1 per class), confusion matrix plot, and saved both as PNGs. Fixed the epochs range bug for early stopping. Expanded `metrics.json` with test P/R/F1. Added two new sections: "Experiment Summary" (prints full config + results) and "Generate Markdown Report" (produces `experiment_report.md` ready for the GitHub README).

### 6. Leaky word removal + reproducibility
Topic-specific words that appear predominantly in true (Reuters) articles — such as `myanmar`, `rohingya`, `hariri` — act as spurious signals: the model can learn "this word → Reuters → True" rather than detecting fake news from linguistic content. Earlier attempts to measure this by running the model with `REMOVE_LEAKY_WORDS=True/False` yielded inconsistent results (0.95, 0.957, 0.989, 0.9648) because **each measurement was a separate training run with a different random seed** — variance from random initialization dominated any masking signal.

#### 7. Freeze weights and progressively remove leaky words
- **Removed** `REMOVE_LEAKY_WORDS` and `LEAKY_WORDS_TOP_N` as single-run hyperparameters.
- **Added `SEED = 42`** and full seed-fixing (`random`, `numpy`, `torch`, `cudnn`) so every run of the same config produces the same model.
- **Added Section 10.3 "Leaky Word Removal Sweep"**: trains the model once, then evaluates the frozen model across `[0, 5, 10, 15]` masked words (capped to the 15 rows in `most_true_words.csv`). Each step clones `X_test` and replaces the token IDs of the top-N leaky words with PAD. Because the model weights never change, differences between steps isolate masking effects from training variance.

#### Results

**Bidirectional LSTM** (test acc ~0.965): The sweep showed essentially no change across all masking levels. Fake recall and True recall stayed flat. Conclusion: the bidirectional model learned distributed style/structure features. Masking 15 words from a 45k vocabulary (~0.03%) is too small a perturbation — each article still contains many correlated topic words, and the bidirectional hidden state aggregates all 512 tokens.

**Unidirectional LSTM** (test acc ~0.66): A real accuracy drop was visible on the left plot, but per-class recalls appeared near-flat. This reveals two things:
1. The unidirectional model learned a **degenerate decision boundary** — it predicts almost everything as Fake (Fake recall 0.93, True recall 0.50, Fake precision only 0.53). Masking leaky words moves accuracy by ~0.001 because the fundamental problem is the model's bias, not the presence of a few topic tokens.
2. The y-axis scale (now auto-zoomed) exposes that True recall does drop slightly (0.4990 → 0.4967) while Fake recall stays fixed — consistent with the theory: Fake articles never contained these True-biased words.

The bidirectional vs. unidirectional gap (0.965 vs 0.66) is the main finding: a forward-only LSTM reading 512 tokens loses early signals by the final step. The backward pass in the bidirectional model starts at the end of the article (where conclusions appear) and captures those signals independently, which explains the large performance difference.

#### Additional observations on the unidirectional collapse

**Confusion matrix scope:** The confusion matrix is generated from the original unmasked test set (sweep top_n=0). It is not recomputed per masking level — it corresponds exactly to the first row of the sweep table.

**The ~50/50 True split explained:** True recall ≈ 0.499 means roughly half of all True articles are correctly classified, and half are misclassified as Fake — almost a coin flip on that class. This is not a coincidence. The model defaulted to predicting Fake for anything it was uncertain about. High Fake recall (0.93) + low True recall (0.50) + low Fake precision (0.53) together confirm the model is biased toward class 0 for most inputs.

**Why unidirectional causes this bias:** The model learned features tied to the cleaned training distribution (stopwords removed, lemmatized). The test set was intentionally left uncleaned, creating a domain mismatch. A weaker unidirectional model overfits to surface-level cleaned-text patterns and defaults to 'Fake' when encountering uncleaned text it doesn't recognize. The bidirectional model, being more expressive, learned more robust features that survived this mismatch — explaining why its per-class recalls remain healthy even on the uncleaned test set.
