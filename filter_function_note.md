# ArtifactFilter — Design Notes

## What it does

`ArtifactFilter.py` is a post-processing step that runs after the REST model produces its final sleep stage scores. It independently inspects the raw EEG signal and overrides any epoch it judges to be an artifact, setting that epoch to stage 4 regardless of what the model predicted.

It uses two rules in OR logic — if either rule fires, the epoch is flagged:

1. **Smooth-PTP** (primary): within-recording peak-to-peak amplitude, temporally smoothed
2. **Saturation** (secondary): fraction of samples clipped at the ADC rail

---

## Why rule-based instead of relying on the model

The REST model's artifact class has near-zero F1 on most datasets during training. The reason is data imbalance — artifacts make up roughly 0.02–0.6% of labeled epochs across the five test datasets. The model sees so few artifact examples that it essentially learns to never predict them.

Artifacts have a physically grounded signature in the raw EEG: they are grossly out of range compared to genuine sleep signal from the same animal. This is the kind of pattern that a hand-tuned signal-processing rule can catch reliably without any training data.

The filter is intentionally EEG-only. EMG activity is elevated during active Wake (which is normal behavior), so using EMG as an artifact indicator would false-flag large amounts of Wake. Analysis of the labeled data confirmed that EMG is elevated during artifact epochs (2–10× Wake median depending on dataset), but adding an EMG rule in OR mode increased false positives far faster than true positives, reducing overall F1.

---

## Rule 1: Smooth-PTP

### Intuition

The peak-to-peak amplitude of an artifact epoch is almost always much larger than a normal EEG epoch from the same recording. But the absolute amplitude cannot be thresholded globally because mouse strains differ: DBA mice have roughly 2× the baseline EEG amplitude of C57 mice. A threshold tuned for C57 would either miss DBA artifacts or flag normal DBA Wake.

The solution is to compute the **within-recording percentile rank** of each epoch's PTP. An epoch at the 99th percentile of its own recording is extreme for that animal, regardless of whether the global amplitude is large or small. This is computed once per recording at inference time — no training required.

### Soft score and temporal smoothing

A hard percentile threshold (flag if PTP > 99th percentile) would fire on isolated high-amplitude arousal epochs during active Wake, producing too many false positives.

The key observation is that artifact events are not isolated — electrode pops, cable movement, and gross signal corruption typically persist across multiple consecutive epochs (8–40 seconds). A single high-PTP Wake epoch is more likely a transient arousal.

The algorithm converts the PTP into a soft score:

```
ref   = percentile(ptp_raw, 97)
soft  = clip((ptp_raw - ref) / ref, 0, 2)
score = uniform_filter1d(soft, size=7)        # 7 epochs × 4 s = 28 s window
```

The uniform filter (box convolution) averages the soft score over a 28-second window. An isolated spike gets diluted by its neighbors and falls below threshold. A genuine artifact burst that spans several epochs sustains a high score and crosses the threshold.

The threshold (0.6) was tuned on ~460 training recordings (not the test datasets) via `tune_artifact_filter.py`.

### Performance by dataset (EEG-only, held-out evaluation)

Parameters tuned on ~460 training recordings; these 5 datasets were never used during tuning.

| Dataset | F1    | Recall | Precision | Wake false-flag |
|---------|-------|--------|-----------|-----------------|
| CD1     | 0.878 | 0.897  | 0.859     | 0.03%           |
| DBAKA   | 0.610 | 0.833  | 0.481     | 0.02%           |
| DBASA   | 0.522 | 0.429  | 0.667     | 0.01%           |
| C57KA   | 0.088 | 0.051  | 0.333     | 0.01%           |
| C57SA   | 0.017 | 0.008  | 0.571     | 0.00%           |
| **ALL** | **0.275** | 0.169 | 0.743 | **0.02%**   |

CD1 is the primary use case and achieves F1=0.878 — excellent for a parameter-free rule.

C57SA is the known hard case. Those artifacts sit at the 95–97th percentile of their recording's PTP distribution, which is indistinguishable from high-amplitude active Wake. No EEG amplitude rule can separate them without also generating massive false positives on Wake. The ML model is the right tool for C57SA.

---

## Rule 2: Saturation

When the EEG amplifier clips at the ADC rail, the waveform flatlines at a fixed ceiling. The saturation rule measures the fraction of samples within 1% of the recording's absolute maximum:

```
rail_thresh = max(|EEG|) × 0.99
saturation  = mean(|epoch| >= rail_thresh)
```

If more than 10% of samples in an epoch touch the rail, it is flagged. This rule has 97% precision on labeled data — nearly every flagged epoch is a genuine saturation artifact. It catches events that the Smooth-PTP rule can miss: brief, hard-clipped epochs where PTP is paradoxically small (the waveform is stuck at the ceiling rather than swinging high).

---

## Parameters

Stored in `artifact_params.json` (loaded automatically at inference time):

| Parameter | Value | Effect |
|-----------|-------|--------|
| `ptp_percentile` | 97.0 | Within-recording PTP reference level. Higher = only the most extreme epochs qualify. |
| `smooth_window` | 7 | Temporal averaging window in epochs (7 × 4 s = 28 s). Larger = requires longer-duration artifact bursts. |
| `score_threshold` | 0.6 | Minimum smoothed score to flag. Higher = more conservative (fewer flags). |
| `saturation_threshold` | 0.10 | Fraction of samples at ADC rail to flag. |

To retune these parameters, run `tune_artifact_filter.py` against any dataset with manual labels.

---

## Integration

The filter runs as the **last step** of the scoring pipeline, after the REST model, Viterbi/HMM smoothing, and bout duration filtering:

```
EEG raw signal
    → REST transformer  →  softmax probs
    → Viterbi smoother  →  sleep stage sequence
    → Bout filter       →  removes implausibly short bouts
    → ArtifactFilter    →  overrides flagged epochs to stage 4  ← here
    → save .mat
```

Toggled by `ARTIFACT_FILTER = True/False` in `Inference.py` and the "Artifact Filter" checkbox in `Inference_GUI.py`.
