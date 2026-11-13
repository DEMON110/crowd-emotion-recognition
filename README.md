# Crowd-Level Emotion Recognition — Reference Evaluation Harness

Official evaluation harness and reproducibility artefacts for the manuscript:

> **Crowd-Level Emotion Recognition from Still-Image Sequences via a Cascaded
> Viola–Jones Detector, HOG–Quadratic-SVM Classification, and Temporally
> Decayed Evidence Voting** *(submitted)*

The pipeline estimates the **dominant affective state of a crowd** from a
sequence of still images (single-image operation is a special case):

```
image → Viola–Jones cascade (scaleFactor 1.1, minNeighbors 3)
      → FP suppression (average-width threshold + upper-body consistency)
      → HOG (64×64 win, 16×16 cells, 2×2 blocks, stride 8, 9 bins, L2-Hys)
      → quadratic SVM (c = 1, C = 1, one-vs-rest)  +  Ø (below-confidence)
      → temporal decay of evidence, E_e(t) = (1+ρ)·E_e(t−1) + n_e(t), ρ = −0.10
      → majority voting → one crowd-level label
```

Every stage is deterministic and individually inspectable — the system is a
deliberate, auditable, resource-constrained alternative to black-box
transformer models for safety-critical monitoring.

---

## Quickstart

```bash
pip install -r requirements.txt

# 1. sanity: full pipeline on synthetic data, no downloads needed
python tests/smoke_test.py

# 2. get the public data (licences: see data/README.md — JAFFE forbids redistribution)
python data/download_data.py jaffe
python data/download_data.py fer2013

# 3. reproduce Table XIV (per-face external validation)
python src/eval_crowd_emotion.py train-jaffe --jaffe data/raw/jaffe/jaffe --model-out jaffe_model.pkl
python src/eval_crowd_emotion.py train-fer  --fer2013 data/raw/fer2013_train.parquet --model-out fer_model.pkl

# 4. (optional) reproduce the FindingEmo external-validity pilot
python data/download_data.py findingemo
python src/eval_crowd_emotion.py infer-dir --model fer_model.pkl \
    --manifest data/processed/findingemo_manifest.json --out results/findingemo_rerun.json
```

The four notebooks in `notebooks/` are **committed with executed outputs** —
you can read the full analysis on GitHub without running anything.

## Headline results (all reproducible with the commands above)

### Table XIV — per-face recognition core on public corpora (10-fold CV)

| Corpus | Composition | Top-1 (%) | Macro-F1 (%) | Balanced acc. (%) | MCC |
|---|---|---|---|---|---|
| FER-2013 | 10,000 in-the-wild faces, balanced over 4 classes | 63.7 | 63.7 | 63.7 | 0.516 |
| JAFFE | 122 posed faces over 4 classes | 95.9 | 95.9 | 95.9 | 0.946 |

Reproduce: notebooks [01](notebooks/01_per_face_validation_jaffe.ipynb) and
[02](notebooks/02_per_face_validation_fer2013.ipynb).
FER-2013 substitutes for licence-restricted FER+ (stated openly in §IV-H).

### Table XV — Green AI energy analysis (TDP-bounded worst-case estimates)

| System | Runtime (ms/img) | Energy (J/img) | Images per kWh | kWh per 10⁶ images/day |
|---|---|---|---|---|
| Proposed pipeline | 14 | 1.12 | 3.21 × 10⁶ | 0.31 |
| CNN-4 | 46 | 3.68 | 9.78 × 10⁵ | 1.02 |
| ResNet-50 | 68 | 5.44 | 6.62 × 10⁵ | 1.51 |
| VGG-16 | 218 | 17.44 | 2.06 × 10⁵ | 4.84 |

Estimates = measured runtimes (manuscript Table VIII, Xeon E-2176G host) ×
TDP 80 W — **bounded estimates, not direct power measurements**; they
preserve the ordering and give conservative absolute values. Reproduce:
notebook [04](notebooks/04_energy_analysis.ipynb).

### Human-level comparison (in-house crowd study, §IV-E)

The system agrees with the 60-rater majority consensus in **94.1%** of scenes
(95% CI [91.0, 96.1]) and remains at **81.3%** on the 48 scenes whose human
majority label is *not* Anger. Mean pairwise inter-rater agreement derived
from Fleiss' κ = 0.71 is ≈ **82.7%** — the system deviates from the
collective judgment less than two human observers typically disagree
(quantities are not strictly commensurate; see the Discussion).

### External-validity pilot on FindingEmo — documented negative result

A balanced 800-scene pilot on FindingEmo whole-scene labels scored
**29.6% top-1 (chance = 25%)**. This is a *construct mismatch*, not an
implementation defect (the same harness scores 63.7%/95.9% in-distribution):
FindingEmo labels describe whole-scene context emotion, not the aggregate of
facial expressions. Full analysis: notebook
[03](notebooks/03_findingemo_external_validity.ipynb) and
`results/findingemo_results.json`. The crowd-level benchmark claim therefore
rests on the pre-registered GAF 3.0 / VGAF protocol (§IV-H), pending licence
approval from the EmotiW organisers.

## Repository layout

```
├── src/eval_crowd_emotion.py   # the reference harness (single file, CLI)
├── notebooks/                  # 01 JAFFE · 02 FER-2013 · 03 FindingEmo pilot · 04 energy (executed)
├── results/                    # machine-readable JSON for every experiment
│   └── tables/                 # CSV + Markdown of Tables XIV/XV and pilot metrics
├── figures/                    # confusion matrices, energy charts
├── data/                       # download_data.py + per-dataset licences (no data committed)
├── tests/smoke_test.py         # dataset-free CI test
└── .github/workflows/ci.yml
```

## Citing

See `CITATION.cff`. If you use the harness, please cite the manuscript and
the datasets you download (JAFFE [Lyons et al. 1998], FER-2013 [Goodfellow
et al. 2015], FindingEmo [Van Hove et al. 2024]).

## License

Code: MIT (see `LICENSE`). Datasets: their own licences — read
`data/README.md` before downloading; **JAFFE may not be redistributed**.
