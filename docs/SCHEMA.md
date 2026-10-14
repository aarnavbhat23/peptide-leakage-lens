# Input schema and interpretation

Input `rows` contains unique string `id`, canonical uppercase `sequence`, string `split`, explicit `chemistry`, and optional `label`. `similarity_threshold` is in (0,1], default 0.9. Length limit: 300 residues; unique-sequence limit: 1000. Chemistry is caller-supplied, not inferred from letters. Use `unknown` when absent. All distinct split pairs are screened, including train/validation and validation/test.

`edit_similarity` is one minus unit-cost Levenshtein distance divided by maximum length. This is not percentage sequence identity, homology, a substitution-matrix alignment score or proof of pretraining overlap. The length-ratio prefilter cannot omit a pair meeting this edit threshold. This pure-Python quadratic screen is for small datasets; use MMseqs2 for large libraries.

`cross_split_pairs` provides row IDs, split names, similarity and chemistry equality. `chemistry_or_label_collisions` groups identical letter sequences carrying differing chemistry or labels. A flag is a review lead, not an invalid assay. Threshold 1 gives an exact-sequence cross-source audit. Different papers can validly test the same peptide; leakage is relevant when claiming paper-held-out novelty.
