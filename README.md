# Peptide Leakage Lens

Audit cross-split sequence neighbors, chemistry collisions and conflicting labels before benchmarking peptide models.

**Status: working research prototype.** The bundled input is a synthetic fixture for exercising code. Its report is not evidence about real biological data or real model performance.

## Try it

Python 3.13 was used for local validation. From this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tool.py examples/input.json --output report.json
python -m unittest discover -s tests -v
```

See [`examples/input.json`](examples/input.json) for a complete input and [`examples/report.json`](examples/report.json) for the actual output from that fixture. [`docs/SCHEMA.md`](docs/SCHEMA.md) describes the fields and interpretation. The report records the input SHA-256. No API key is needed for this demo.

## Contribution and related work

An inspectable sequence-pair ledger that combines a simple identity screen with stereochemistry and terminal-modification checks.

Sequence leakage benchmarks already exist. This is a lightweight audit utility, not a new homology algorithm. Related primary source: [https://github.com/afaquealam-nitp/pdb-homology-leakage-benchmark](https://github.com/afaquealam-nitp/pdb-homology-leakage-benchmark). The related-work scan is preliminary; no claim of being the first implementation is made.

## Next experiment

Collect independently sourced positive and negative assays; use MMseqs2 clusters; compare composition and frozen ESM probes on held-out papers.

This milestone is planned, not completed. Preregister splits, metrics and exclusions before inspecting test results. Preserve source licenses and version identifiers when adding external data.

## Why this project

Protein ML teams; builds on your peptide work. This describes technical relevance, not company affiliation, a hiring prediction or a claim that the company requested the tool.

## Validation

The initial local run passed 8 regression tests and the documented CLI example. CI repeats the core tests and example on Python 3.13. Network/model inference is not run in CI.

## Limitations and authorship

See the schema notes for method-specific limitations. This software has not been independently validated by researchers and does not establish experimental activity, clinical usefulness or model superiority.

Initial implementation and toy fixtures were generated with Codex. Maintainer: Aarnav Bhat. Real-data studies and human review remain future work unless a separately documented result explicitly says otherwise. Code and original synthetic fixtures are MIT licensed; externally sourced datasets and model weights retain their own licenses.

## Contributing

Useful contributions include a small reproducible failure case, an independently reviewed dataset adapter, or a metric correction. Include the input schema, expected behavior and source provenance. Avoid committing personal data, credentials, model caches or data without redistribution rights.

## Real public-data application

The initial release includes an exact cross-paper audit of the maintainer's existing public assay table. See [the findings](results/public-assay/FINDINGS.md) and [report](results/public-assay/report.json). Reproduce with `python audit_public_data.py`. Chemistry is unspecified in that source and is never guessed.
