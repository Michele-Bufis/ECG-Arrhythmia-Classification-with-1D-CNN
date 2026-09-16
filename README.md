# ECG Arrhythmia Classification with 1D CNN

A from-scratch exploration of arrhythmia classification on raw ECG waveforms using a compact 1D CNN, focused on solving real-world data scarcity and class imbalance rather than chasing a single accuracy number.

**Dataset**: [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) (PhysioNet)
**Approach**: raw signal input, no hand-crafted feature engineering — the network learns morphology directly from the filtered waveform.


## Motivation

Most beginner ECG classification projects report a single accuracy number on an imbalanced dataset and stop there. This project instead documents the full diagnostic process of building a usable classifier under realistic constraints: a 113:1 class imbalance, one class (fusion beats) that is morphologically ambiguous by clinical definition, and a patient-level train/test split to avoid data leakage.

## Key methodological decisions

- **Patient-level splitting** (not beat-level) throughout, to prevent leakage between train/validation/test
- **AAMI 5-class mapping** (N, S, V, F, Q) instead of raw MIT-BIH annotation symbols, for comparability with published literature
- **RR interval as an auxiliary input** (dual-input network: waveform + inter-beat timing), added after diagnosing that the supraventricular class is defined more by timing than morphology
- **Label-preserving data augmentation** (jitter, noise, amplitude scaling) targeted specifically at the rarest class, grounded in accepted biomedical ML practice
- **GroupKFold cross-validation** to obtain an honest, variance-aware performance estimate instead of trusting a single lucky split
- **Confidence threshold (reject option)** instead of forcing a classification when the model is uncertain — standard practice in clinical ML, though the results also expose its limits (see below)

## Results

Final model (single 5-class CNN with RR interval, targeted augmentation, and a global confidence threshold), evaluated once on a held-out patient test set:

**Macro F1: 0.45** (coverage 87.6% at threshold 0.6)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| N | 0.91 | 0.88 | 0.89 |
| V | 0.60 | 0.98 | 0.74 |
| Q | 0.99 | 0.20 | 0.34 |
| S | 0.16 | 0.79 | 0.27 |
| F | 0.00 | 0.00 | 0.00 |

GroupKFold cross-validation (5 folds, patient-grouped) confirms this is representative: macro F1 = 0.50 ± 0.07 across folds.

## What didn't work (and why it's worth reading)

- **Class weights, batch balancing, hierarchical two-stage classification** — four different rebalancing strategies were tried before concluding, empirically, that the rarest class (fusion beats) is not a data-imbalance problem but a *morphological ambiguity* problem: the aggregated confusion matrix shows it splits almost evenly between "Normal" and "Ventricular," consistent with its clinical definition as a hybrid beat.
- **Per-class confidence threshold calibration** — looked excellent on a small internal validation split (near-perfect F1) but collapsed on the held-out test set. A documented case of threshold overfitting on limited data, and a reminder that a metric that looks too good usually is.

Full diagnostic log of every attempt, including this one, is in [`docs/progress-log-ita.md`](docs/progress-log-ita.md) (Italian).

## Project structure

```
notebooks/    -> single self-contained Colab notebook, full pipeline
docs/         -> detailed development log (Italian)
```

## Reproducing

Open the notebook in Colab (badge above), mount your own Google Drive, and run top to bottom. The notebook downloads MIT-BIH directly via `wfdb` — no manual dataset setup required.

## Stack

Python, TensorFlow/Keras, scikit-learn, `wfdb`, Google Colab.

## License

MIT
