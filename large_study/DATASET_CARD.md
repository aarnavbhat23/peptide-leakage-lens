# GRAMPA-derived component benchmark

This is a derived evaluation dataset, not a new experimental measurement collection. It reorganizes the full public GRAMPA table into a versioned source ledger, retained chemistry metadata and non-overlapping sequence components.

The pinned upstream source contains 51,345 records. The builder retains **49,172 records, 6,391 canonical peptide sequences and 750 target organisms**, excluding 2,173 rows with noncanonical sequences or length outside 8–100 residues. There is no small-data sampling cap.

## Target and provenance

The target is the upstream `value`, interpreted as **log10(MIC / µM)** from the authors' [loader](https://github.com/zswitten/Antimicrobial-Peptides/blob/a6819dcd671291af8013627a2d2fbc32858346bc/src/load_data.py). The original concentration inequalities are not available in this processed table. No censored likelihood, newly verified wet-lab label or correction of those upstream measurements is claimed.

Rows retain the upstream database, source URL, strain, modification string and four modification flags. Unknown modification metadata stays unknown. Flags are not a complete molecular representation of terminal chemistry, stereochemistry or intrachain bonding. Database records may repeat underlying experiments; 49,172 is a record count, not a count of independent assays.

## Split construction

All unique sequences are screened with compiled normalized Levenshtein similarity. Edges at ≥0.8 define transitive connected components, yielding **2,775 components and 14,358 edges**. This is an edit-based family proxy, not certified sequence homology or an MMseqs2 percentage-identity threshold.

Whole components are assigned with fixed group-shuffle seeds to training, validation, calibration and test. No sequence or component crosses a split. Actual row counts are **31,561 / 6,799 / 4,339 / 6,473** respectively. Group-count fractions do not imply equal row fractions. The separate calibration partition is never used to fit or select a prediction head.

Publication IDs are absent, so publication holdout is unavailable. Source/laboratory dependencies can remain between different components. Organism vocabularies and feature normalization are fitted on training only; unseen test organisms map to an explicit unknown category.

## Rebuild and use

`python large_study/build_dataset.py` downloads the pinned source and writes `data/records.csv.gz`, `split-ledger.csv`, `manifest.json` and `exclusions.json`. The split ledger and manifest are public. The complete source-derived record snapshot is generated locally; original database terms are not relabeled as the repository's MIT code license. This keeps the release inspectable without asserting redistribution permissions absent from the upstream repository.

Cite Witten and Witten, *Deep learning regression model for antimicrobial peptide design*, and the underlying GRAMPA database sources. Cite the original ESM-2 paper for pretrained representations. This derived benchmark has no DOI or independent scientific validation yet.
