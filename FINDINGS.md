# Methodology & Findings

A narrative walkthrough of how this project was built, the experiments that failed, and what they revealed. Numbers state which data they were measured on, because the project mixes single validation splits, cross-validation and a test set.

## The problem

The MIT-BIH Arrhythmia Database has 109,494 annotated heartbeats across 48 recordings (47 subjects). Mapped to the standard 5 AAMI classes:

| Class | Meaning | Count | Share |
|---|---|---|---|
| N | Normal | 90,631 | 82.8% |
| Q | Unknown / paced | 8,043 | 7.3% |
| V | Ventricular | 7,236 | 6.6% |
| S | Supraventricular | 2,781 | 2.5% |
| F | Fusion | 803 | 0.7% |

The imbalance between N and F is **113:1**. A model that always predicts "Normal" scores about 83% accuracy and is clinically useless. After segmentation 109,446 beats remain (48 were too close to a record edge: 42 N, 4 Q, 1 V, 1 F).

## Design decisions

- **Splitting by record, not by beat**, for train/validation/test and every cross-validation fold. Caveat: records 201 and 202 belong to the same subject (PhysioNet documentation) and are on different sides of the split (201 in train, 202 in test; also in different CV folds). The effect is probably small but "patient-level" is not perfectly true.
- **AAMI 5-class mapping** (N/S/V/F/Q). This makes the class definitions standard, not the evaluation protocol: published inter-patient work usually uses a fixed DS1/DS2 split and excludes paced records (102, 104, 107, 217), which are included here. Results are not directly comparable to that literature.
- **Waveform input plus one timing feature.** The network sees filtered, per-segment z-scored beats. It also receives the **RR interval** (samples since the previous beat), a hand-crafted feature computed from the dataset's reference annotations rather than from a detector. No QRS width or amplitude ratios are computed by hand. Per-segment z-scoring removes absolute amplitude information.

## What was tried on the class imbalance problem

Numbering below follows this document; the notebook titles ("Tentativo N") use a slightly different numbering.

| # | Approach | Macro F1 | Evaluated on | Takeaway |
|---|---|---|---|---|
| 1 | Class weights (uncapped) | 0.36 | validation split | F weight (35.7x) caused erratic gradients |
| 2 | Weights capped at 10x + lower LR | 0.37 | validation split | Capping alone did not fix batch sparsity |
| 3 | Stratified batch sampling | 0.38 | validation split (14,911 beats) | Fixed batch composition, not scarcity; F still about 0 |
| 4 | Hierarchical two-stage (N vs. anomalous, then S/V/F/Q) | 0.37 | validation split | Stage-1 recall bottleneck erased Stage-2 gains. Also 0.37 when combined with RR and augmentation |
| 5 | RR interval + targeted augmentation on F (training only) | 0.41 | validation split | The notebook tested the two together, so the individual contribution of RR was not isolated. S is plausibly timing-defined |
| 6 | All techniques combined, early stopping on validation Macro F1 | 0.42 | validation split | Marginal gain |
| 7 | GroupKFold (5 folds) of the approach in 6 | 0.52 ± 0.09 | cross-validation, full coverage | See "Cross-validation caveat" |
| 8 | Per-class confidence thresholds on the two-stage model | 0.43 (val) / **0.36 (test)** | validation, then test (second look at the test set, full coverage) | Overfit a tiny validation subset |

Attempt 8 is the most instructive failure. The per-class threshold search reported F1 of 1.00 for S and 0.9993 for Q (F reached only 0.40), but these are measured on the subset each threshold lets through, which inflates them. End to end on the same validation set S had F1 0.00, and on the test set Macro F1 fell to 0.36. It is kept in the log on purpose.

## Cross-validation caveat

Early stopping monitored `val_macro_f1` on the **fold being scored**, with the best weights restored. Each fold's reported score therefore equals its best epoch (0.5766, 0.4080, 0.5725, 0.4099, 0.6198), while per-epoch values within a fold swing widely (for example 0.36 to 0.58 in fold 1). The 0.52 ± 0.09 estimate is optimistic. A cleaner protocol would choose the stopping epoch on a separate inner validation set.

| Fold | Macro F1 | F beats / recall | Q beats / F1 |
|---|---|---|---|
| 1 | 0.5766 | 17 / 0.00 | 2,083 / 0.97 |
| 2 | 0.4080 | 378 / 0.00 | 5 / 0.00 |
| 3 | 0.5725 | 13 / 0.00 | 3,879 / 0.91 |
| 4 | 0.4099 | 16 / 0.00 | 4 / 0.00 |
| 5 | 0.6198 | 378 / 0.97 (precision 0.06) | 2,068 / 0.88 |

## Fusion class (F)

The aggregated confusion matrix over all folds shows 368 of 802 F beats predicted F (46%), 349 as N (44%) and 76 as V (9%). This pattern comes almost entirely from two folds:

- Fold 5: F recall 0.97 but precision 0.06 (5,046 N beats in total are predicted F across folds, so the model over-predicts F in some patients).
- Fold 2: 378 F beats with recall 0.00.
- Three other folds have only 13 to 17 F beats each.

So the aggregate mixes a near-perfect fold, a zero fold and tiny ones. It looks like a **patient-concentration problem** (F beats come from very few records, so the model sees almost no independent patients) as much as a label-ambiguity problem. Fusion beats being a hybrid of N and V remains a plausible explanation, supported by the test set where 10 of 13 classified F beats were predicted N, but the project does not demonstrate it. Per-record F counts would settle the question.

## Supraventricular (S) and paced (Q) classes

- **S** stays weak everywhere (CV F1 0.26, test F1 0.27 with recall 0.79 and precision 0.16): many N beats are predicted S (921 on the test set).
- **Q** has CV F1 0.55, but that mean includes two folds with only 5 and 4 Q beats (F1 0.00). In the three folds with a real number of Q beats the F1 is 0.97, 0.91 and 0.88. On the test records Q recall is **0.20**: 1,133 of 1,428 classified Q beats were predicted N, above the confidence threshold. Cross-validation does not predict this failure, so it should not be read as "CV confirms the test result" for Q.

## Final model and results

Architecture: 3-block 1D CNN (15,061 parameters), two inputs (waveform and normalized RR interval) merged before a softmax layer. Training: balanced class weights, F augmented 8x (training only), early stopping on Macro F1 of an internal validation set (13,498 beats from 6 held-out records of the training pool), final fit on train+validation records (40 records).

**Threshold choice** (internal validation, which also drove epoch selection and contains only 2 F beats):

| Threshold | Coverage | Macro F1 on retained beats |
|---|---|---|
| 0.3 | 100.0% | 0.5234 |
| 0.5 | 99.4% | 0.5270 |
| 0.6 | 97.8% | 0.5406 |
| 0.7 | 94.8% | 0.5410 |
| 0.8 | 87.7% | 0.5552 |
| 0.9 | 61.5% | 0.4709 |

0.6 was set manually. The best validation Macro F1 is at 0.8, and at 0.6 none of the 2 validation F beats is flagged. On the test set the coverage at 0.6 is 87.6%, about ten points lower than on validation, so the threshold does not transfer exactly to unseen patients.

**Held-out test set** (8 records: 103, 104, 202, 203, 205, 219, 221, 222; 19,140 beats), threshold 0.6, 87.6% of beats classified, 12.4% flagged uncertain: Macro F1 = **0.4488** on the classified beats.

| Class | Precision | Recall | F1 | Support (classified) |
|---|---|---|---|---|
| N | 0.91 | 0.88 | 0.89 | 14,147 |
| V | 0.60 | 0.98 | 0.74 | 955 |
| Q | 0.99 | 0.20 | 0.34 | 1,428 |
| S | 0.16 | 0.79 | 0.27 | 229 |
| F | 0.00 | 0.00 | 0.00 | 13 |

Comparability: the test Macro F1 excludes the 12.4% uncertain beats, whereas the CV figure and the 0.36 of Attempt 8 are at full coverage. Macro F1 at full coverage was **not computed** for the final model, so "0.45 beats 0.36" is not a like-for-like comparison, and neither is "0.52 vs. 0.45" as a measure of variance. The cross-validation gap should be read together with the optimism described above.

## Test-set discipline

The final model (Macro F1 0.4488) was evaluated on the test set **once**, after architecture, class weights, augmentation, the stopping epoch and the threshold (0.6, set from the internal validation table) had been fixed. That is the figure to read as the project's result.

Afterwards the author tried to improve it with a two-stage model and per-class thresholds (Attempt 8) and evaluated it on the **same** test set (Macro F1 0.3558, full coverage). It was worse and was discarded; the final model did not change. Two caveats remain:

- The second look turned the test set into a comparison between two candidates. Had Attempt 8 scored higher it could have been adopted on test evidence, so the test set is no longer a clean hold-out for any further work.
- The Attempt 8 notebook still describes its test evaluation as the "single" use of the test set. That wording is outdated and should be annotated as a second look.

## Limitations and known issues

- Optimistic cross-validation (early stopping on the scored fold).
- Test Macro F1 on a subset of beats; no full-coverage figure for the final model.
- F and S are determined by very few patients; F1 on 13 test F beats is not informative.
- Records 201 and 202 (same subject) split across train and test; paced records included.
- RR interval taken from reference annotations, not from a detector.
- Training is not seeded for TensorFlow, so figures vary slightly between runs.
- Augmentation (circular shifts, noise, scaling) adds variants of existing beats, not new patients.

## What this project is, and isn't

This is a from-scratch learning project, not a production or clinically validated system. The value is the documented process: diagnosing why a model fails on specific classes, testing hypotheses one at a time, catching a threshold-overfitting mistake, and stating what remains unproven.
