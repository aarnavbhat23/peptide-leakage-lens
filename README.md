# Peptide Leakage Lens

A peptide-model evaluation project with a large trained-model benchmark, uncertainty analysis and source/split audits.

The [larger benchmark](large_study/README.md) uses **49,172 GRAMPA records, 6,391 peptides and 750 target organisms**. A frozen **ESM-2 35M transformer** plus three trained species-conditioned neural heads reaches **0.685 test RMSE in log10 MIC units**, essentially tied with a gradient-boosting baseline. It also shows how the calibration unit changes simultaneous coverage across an entire peptide component: **62.4% with row calibration versus 91.4% with component calibration**, at the cost of wider intervals.

![Large model benchmark](large_study/results/benchmark.png)

[Dataset builder and card](large_study/DATASET_CARD.md) · [Trained checkpoints](large_study/results/checkpoints) · [Run the model](large_study/README.md#run-the-released-prediction-heads) · [Test predictions](large_study/results/test-predictions.csv)

## The smaller source-split study

On a public 1,947-row molar-unit assay subset, a frozen ESM-2 8M probe scores **0.679 AUROC with random row splits**, **0.594 with sequence groups**, and **0.597 with paper groups**. In the random split, **92.7%** of held-out rows have a peptide sequence already in training. These are pooled-organism, retrospective low-MIC results; they do not establish an antimicrobial classifier.

![Evaluation splits and exact overlap](results/split-study/split-sensitivity.png)

[Read the methods and limitations](results/split-study/REPORT.md) · [Raw predictions](results/split-study/predictions.csv) · [Run manifest](results/split-study/report.json) · [Fixed study protocol](study-protocol.json)

## Run the audit

The small audit command needs only Python:

```bash
python tool.py examples/input.json --output report.json
python -m unittest discover -s tests -v
```

It produces sequence-pair and chemistry/label ledgers. The [schema](docs/SCHEMA.md) defines the edit-distance screen and its limits. No flags is not proof of homology-free evaluation. The [complete-table exact cross-paper audit](results/public-assay/FINDINGS.md) is a separate application on 2,449 rows; its denominator differs from the fitted study’s eligible subset.

## Reproduce the fitted study

Install [bio-evidence](https://github.com/aarnavbhat23/bio-evidence) and the optional study dependencies:

```bash
python -m pip install -r requirements-study.txt
python -m pip install 'bio-evidence @ git+https://github.com/aarnavbhat23/bio-evidence.git'
HF_HOME=./cache python run_study.py --device cpu
python plot_study.py
```

The source table and model revision are pinned. `--device cuda` and `--device mps` are available; the included run used CPU. `--skip-esm` runs the composition baseline only. Classifier settings are fixed across folds. Whole-paper bootstrap intervals are computed from saved out-of-fold predictions; no model is fitted by the plotting script.

## What changed in the code

The study derives threshold labels only when exact values or censoring bounds determine them. It builds joint paper/sequence components without splitting a source across folds. Group labels are canonicalized by member row IDs, preventing equivalent partitions from changing fold assignments merely because they were renamed. All exclusions, graph edges and predictions are retained.

The joint constraints add no source merges in this eligible subset. That is reported explicitly. This is a specific reproducible analysis built from established methods; it is not a novel homology algorithm or a claim of model leadership.

Initial tooling and this study were developed with Codex. Aarnav Bhat maintains the project. No independent researcher review, wet-lab validation or downstream adoption is claimed. Code is MIT licensed; external sources and model weights retain their own terms.
