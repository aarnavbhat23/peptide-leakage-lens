# Exact cross-paper sequence audit

This real-data diagnostic uses the existing public antimicrobial-assay table at the pinned commit recorded in `report.json`. It is an exact letter-sequence screen, not an assay-validity judgment.

The 2,449 assay rows collapse to 601 sequence–paper groups, covering 596 unique sequences and 22 papers. There are 5 cross-paper exact-match pairs involving 10 sequence–paper groups.

| Sequence | Paper DOI A | Paper DOI B |
|---|---|---|
| `GIIWATWKTRWYSMKKTTMKIIPFNRLTIG` | 10.3390/ijms26188914 | 10.1038/s41551-021-00801-1 |
| `GVVWMNWKGSWYSMRKMSMKIRPFFPQQ` | 10.3390/ijms26188914 | 10.1038/s41551-021-00801-1 |
| `GVVWVSFRGADYSLRAVRMKIRPLVTQ` | 10.3390/ijms26188914 | 10.1038/s41551-021-00801-1 |
| `SFRGADYSLRAVRMKIRPLVTQ` | 10.1038/s41551-021-00801-1 | 10.3390/ijms26188914 |
| `TWKTRWYSMKKTTMKIIPFNRLTIG` | 10.1038/s41551-021-00801-1 | 10.3390/ijms26188914 |

Each paper DOI acts as a separate split label for this diagnostic. No classifier was fitted and no model score was measured. Independently repeating an assay is useful science; repeated sequences would limit a claim that a paper-held-out evaluation consists entirely of unseen peptide sequences. Chemistry fields are absent from this source, so letter equality cannot establish molecular identity. Repeated assay rows within the same paper were collapsed only for this diagnostic; source row numbers remain in the input ledger.

Reproduce with `python audit_public_data.py`. External-source availability is required; the included input/report can be analyzed offline. The source dataset's assay-evidence review has its own limitations and is not independently re-adjudicated here.
