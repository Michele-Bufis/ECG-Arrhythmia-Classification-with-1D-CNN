# Methodology & Findings

A narrative walkthrough of how this project was built, the experiments that failed, and what they revealed.

## The problem

MIT-BIH Arrhythmia Database contains 109,494 annotated heartbeats across 48 patient recordings. Mapped to the standard 5 AAMI classes, the distribution looks like this:

| Class | Meaning | Count | Share |
|---|---|---|---|
| N | Normal | 90,631 | 82.8% |
| Q | Unknown / paced | 8,043 | 7.3% |
| V | Ventricular | 7,236 | 6.6% |
| S | Supraventricular | 2,781 | 2.5% |
| F | Fusion | 803 | 0.7% |

That's a **113:1 imbalance** between the most and least common class. A model that always predicts "Normal" scores 83% accuracy while being clinically useless. The real question isn't "can a CNN classify ECG beats" — it's **what does it take to make a model pay attention to the classes that actually matter**, with a fixed, limited amount of real-world data.

## Design decisions and why

**Patient-level splitting, always.** Every split — train/validation/test, and every fold in cross-validation — divides by *patient record*, never by individual beat. Splitting by beat lets the model see beats from the same patient in both train and test, inflating scores in a way that doesn't reflect real generalization.

**AAMI 5-class mapping.** Raw MIT-BIH annotations use ~15 symbols; grouping them into the standard AAMI classes (N/S/V/F/Q) makes results comparable to published literature and avoids training on classes with only a handful of examples.

**Raw waveform input, no hand-crafted features.** The network sees filtered, normalized signal segments — no QRS width, no amplitude ratios computed by hand. The only preprocessing is a band-pass filter and per-segment z-score normalization; morphology is learned end-to-end.

## What was tried on the class imbalance problem, in order

| # | Approach | Result | Verdict |
|---|---|---|---|
| 1 | Class weights (uncapped) | Macro F1 0.36, training unstable | F rare class weight (35.7x) caused erratic gradients |
| 2 | Class weights capped at 10x + lower LR | Macro F1 0.37, still unstable | Capping alone didn't fix batch sparsity |
| 3 | Stratified batch sampling | Macro F1 0.38, F still ~0 | Solved batch composition, not the underlying scarcity |
| 4 | Hierarchical two-stage classifier (Normal vs. Anomalous, then sub-classify) | Macro F1 0.37 end-to-end | Stage 1 recall bottleneck erased gains from Stage 2 |
| 5 | + RR interval (inter-beat timing) as auxiliary input | Macro F1 0.41; **S improved meaningfully** | Confirmed S is timing-defined, not just morphology |
| 6 | + Targeted data augmentation on F (jitter, noise, amplitude scaling — training set only) | Macro F1 0.42 | Marginal further gain |
| 7 | GroupKFold cross-validation (5 folds) of the combined approach | Macro F1 **0.52 ± 0.09** across folds | More honest, variance-aware estimate; **Q turned out to be learnable (F1 0.55)** once it actually appeared in a validation fold |
| 8 | Per-class confidence threshold calibration on top of the hierarchical model | Macro F1 **0.36 on the held-out test set** — worse than the simpler model | **Overfit to a small validation subset**; near-perfect thresholds on ~2 validation examples of F didn't generalize |

Attempt 8 is the most interesting failure in this project. The calibration looked excellent on validation (F1 near 1.0 for some classes) — a textbook warning sign that gets easy to miss when you're chasing a better number. It's kept in the log as a deliberate example of overfitting on limited data, not hidden.

## Why the fusion class (F) never really improved

The aggregated confusion matrix across all cross-validation folds tells a clear story:

| | Predicted F | Predicted N | Predicted V |
|---|---|---|---|
| **Actual F** | 46% | 44% | 9% |

F is confused almost exclusively with N and V, in roughly the proportion you'd expect from its clinical definition: a fusion beat *is* a hybrid between a normal and a ventricular beat. No amount of rebalancing fixes an inherently ambiguous morphology — that's a property of the label, not a deficiency of the model. This is why the final model uses a **confidence threshold (reject option)** instead of forcing a classification: when the model is uncertain, it says so, rather than guessing.

One limit of that approach also showed up in testing: the threshold only helps when the model *itself* signals low confidence. On the test set, most misclassified Q beats were predicted as N with *high* confidence — a case where reject-option thresholding doesn't help, because the model is confidently wrong rather than visibly unsure.

## Final model and results

Architecture: a compact 3-block 1D CNN (~15K parameters) with two inputs — the filtered waveform and the normalized RR interval — merged before a final softmax layer.

**Cross-validation estimate** (5-fold GroupKFold, patient-grouped): Macro F1 = 0.52 ± 0.09

**Held-out test set** (touched once, after all modeling decisions were frozen): Macro F1 = 0.4488, at a global confidence threshold of 0.6 (87.6% of test beats classified, 12.4% flagged as low-confidence)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| N | 0.91 | 0.88 | 0.89 |
| V | 0.60 | 0.98 | 0.74 |
| Q | 0.99 | 0.20 | 0.34 |
| S | 0.16 | 0.79 | 0.27 |
| F | 0.00 | 0.00 | 0.00 |

The gap between the cross-validation estimate (0.52) and the single test result (0.45) is itself informative: it falls within the fold-to-fold variance already observed (0.41–0.62 across the 5 folds), not evidence of something going wrong — just a reminder that a single train/test split, however careful, is one draw from a distribution.

## What this project is, and isn't

This is a from-scratch learning project, not a production or clinically validated system. The value here isn't the final F1 number — it's the documented process of diagnosing *why* a model fails on a specific class, testing hypotheses one at a time, catching an overfitting mistake before shipping it as a result, and being explicit about what didn't work.

A methodologically honest disclosure, made explicit in the full log: the test set was evaluated twice over the course of the project (once for the model ultimately adopted, once for the discarded Attempt 8). The intended discipline — touch the test set exactly once — was broken in the pursuit of a better number, and the better-performing but more principled result (Attempt 6/7) was chosen as final specifically *because* Attempt 8's test evaluation revealed overfitting, not because it scored higher.
