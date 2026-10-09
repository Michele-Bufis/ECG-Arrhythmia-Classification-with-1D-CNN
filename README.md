# ECG Arrhythmia Classification with 1D CNN

A from-scratch exploration of heartbeat classification on the MIT-BIH Arrhythmia Database with a compact 1D CNN (~15K parameters), focused on what it takes to handle severe class imbalance and data scarcity, and on documenting honestly what worked and what did not. This is a learning project, not a clinically validated system.

**Dataset**: [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) (PhysioNet), 48 recordings from 47 subjects.

## Motivation

Many ECG classification projects report a single accuracy figure on an imbalanced dataset and stop there. Here the class distribution is 113:1 between the most and least frequent class (N vs. F), the fusion class (F) is concentrated in a handful of recordings, and evaluation is done at the patient (record) level. The interesting part is the diagnostic process, which is written up in [`FINDINGS.md`](FINDINGS.md).

## Approach

- **Input**: band-pass filtered (0.5-40 Hz) beat segments (300 samples around the R peak), z-score normalized per segment. The network learns morphology from the waveform.
- **RR interval as a second input**: the time since the previous beat, computed from the dataset's reference beat annotations (not from an R-peak detector), normalized with training-set statistics. This is a hand-crafted timing feature and the only one used. A deployed system would need its own beat detector.
- **Classes**: standard AAMI mapping (N, S, V, F, Q). Paced recordings (102, 104, 107, 217) are kept, which is why Q is large. Note that published inter-patient protocols usually exclude them and use a fixed DS1/DS2 split, so absolute numbers here are **not directly comparable** to that literature.
- **Splitting by record, never by beat**, for train/validation/test and for every cross-validation fold. Known caveat: records 201 and 202 come from the same subject and end up on different sides of the split (201 in train, 202 in test).
- **Targeted augmentation** (circular time shift, light Gaussian noise, amplitude scaling) on the rarest class (F), training data only. It adds variants of existing beats, not new patients.
- **GroupKFold cross-validation** (5 folds, grouped by record).
- **Confidence threshold (reject option)**: predictions with maximum softmax probability below a global threshold are flagged as uncertain instead of being forced into a class.

## Results

> **Test-set disclosure.** The final model below was evaluated on the held-out test set (8 records, 19,140 beats) **once**, with architecture, augmentation, stopping point and threshold all fixed beforehand. Afterwards the same test set was reused to evaluate a further variant (a two-stage model with per-class thresholds, Macro F1 0.3558 at full coverage), which was worse and discarded. That second look did not change the final model, but it turned the test set into a comparison between two candidates, so it is no longer a clean hold-out for future work. Details in [`FINDINGS.md`](FINDINGS.md).

**Final model**: single 5-class dual-input CNN (waveform + RR interval), augmentation on F, balanced class weights, global confidence threshold 0.6.

**Held-out test set, threshold 0.6**: Macro F1 **0.4488**, computed on the **87.6% of beats** the model classified (16,772 of 19,140); the other 12.4% were flagged as uncertain and are excluded from the metrics.

| Class | Precision | Recall | F1 | Support (classified beats) |
|---|---|---|---|---|
| N | 0.91 | 0.88 | 0.89 | 14,147 |
| V | 0.60 | 0.98 | 0.74 | 955 |
| Q | 0.99 | 0.20 | 0.34 | 1,428 |
| S | 0.16 | 0.79 | 0.27 | 229 |
| F | 0.00 | 0.00 | 0.00 | 13 |

**Cross-validation** (5-fold GroupKFold, full coverage): Macro F1 **0.52 ± 0.09** (folds range from 0.41 to 0.62). This estimate is **optimistic**: early stopping monitored the Macro F1 of the very fold being scored, so each fold's reported score is its best epoch. Per-class means: N 0.89, V 0.86, Q 0.55, S 0.26, F 0.02.

How to read these numbers:

- The test Macro F1 is computed on a subset (confident beats only), while the cross-validation figure uses all beats. They are not like-for-like.
- F has 13 classified beats in the test set, so its F1 of 0.00 is close to noise. Q looks good in cross-validation when it is well represented in the fold (F1 around 0.9) but recall drops to 0.20 on the test records, mostly predicted as N with confidence above the threshold.
- The confidence threshold does not rescue these cases: the model is confidently wrong, not uncertain.

## What did not work

- **Class weights, stratified batches, hierarchical two-stage classification**: none produced a clear gain on the fusion class. The evidence points to fusion beats being concentrated in very few patients (two cross-validation folds contain about 378 F beats each; recall is 0.00 in one and 0.97 in the other), rather than being a simple class-weight problem. Whether the label is also morphologically ambiguous (fusion = hybrid of N and V) is plausible but not demonstrated here.
- **Per-class confidence thresholds on a two-stage model**: near-perfect F1 on tiny validation subsets, Macro F1 0.36 on the test set. A documented case of threshold overfitting.

Every attempt, with the data it was evaluated on, is in [`FINDINGS.md`](FINDINGS.md).

## Reproducing

Open the notebook in Colab (badge above), mount your own Google Drive and run top to bottom. The notebook downloads MIT-BIH through `wfdb` and caches processed arrays and cross-validation checkpoints on Drive.

- Training is not seeded for TensorFlow, so exact figures (including 0.4488) will vary between runs.
- The cross-validation checkpoint (`groupkfold_risultati.pkl`) is reused if it already exists on Drive. Delete it to recompute the folds.

## Repository layout

- `ecg_cnn_progetto_completo.ipynb`: final notebook (data, cross-validation, final model, threshold, test evaluation)
- `FINDINGS.md`: full diagnostic log, including failed attempts and limitations
- `attempts/`: notebooks of earlier attempts (optional, for traceability)
  
## Stack

Python, TensorFlow/Keras, scikit-learn, SciPy, `wfdb`, pandas, matplotlib/seaborn, Google Colab.

## License

MIT
