"""Sequence/chemistry leakage preflight, without declaring homology-free splits."""
from collections import defaultdict
from common import need, number, run, unique_ids

AA = set('ACDEFGHIKLMNPQRSTVWY')

def edit_similarity(a, b):
    """1 - unit-cost Levenshtein distance / maximum length; not alignment identity."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        curr = [i]
        for j, y in enumerate(b, 1):
            curr.append(min(curr[-1]+1, prev[j]+1, prev[j-1]+(x != y)))
        prev = curr
    return 1 - prev[-1] / max(len(a), len(b))

def analyze(data):
    """Audit supplied split labels, exact chemistry collisions and sequence neighbors."""
    need(data, 'rows')
    rows = data['rows']
    if not rows:
        raise ValueError('At least one sequence is required')
    threshold = number(data.get('similarity_threshold', .9))
    if not 0 < threshold <= 1:
        raise ValueError('similarity_threshold must be in (0, 1]')
    for r in rows:
        need(r, 'id', 'sequence', 'split', 'chemistry')
        if not r['sequence'] or not set(r['sequence']) <= AA:
            raise ValueError('Use uppercase canonical residues; encode chemistry separately')
        if len(r['sequence']) > 300:
            raise ValueError('This small-peptide prototype supports up to 300 residues')
        if not r['split'] or not isinstance(r['chemistry'], str) or not r['chemistry'].strip():
            raise ValueError('split and explicit chemistry are required')
    unique_ids(rows)
    by_sequence = defaultdict(list)
    for r in rows:
        by_sequence[r['sequence']].append(r)
    collisions = []
    for seq, group in by_sequence.items():
        chemistry = sorted({r['chemistry'] for r in group})
        labels = sorted({str(r['label']) for r in group if 'label' in r})
        if len(chemistry) > 1 or len(labels) > 1:
            collisions.append({'sequence': seq, 'ids': [r['id'] for r in group],
                               'chemistries': chemistry, 'labels': labels})
    sequences = list(by_sequence)
    if len(sequences) > 1000:
        raise ValueError('For >1000 unique sequences use MMseqs2, then audit supplied groups')
    pairs, affected, checked = [], set(), 0
    for i, a in enumerate(sequences):
        for b in sequences[i:]:
            ga, gb = by_sequence[a], by_sequence[b]
            if not any(x['split'] != y['split'] for x in ga for y in gb):
                continue
            if min(len(a),len(b))/max(len(a),len(b)) < threshold:
                continue
            similarity = 1.0 if a == b else edit_similarity(a, b)
            checked += 1
            if similarity < threshold:
                continue
            for x in ga:
                for y in gb:
                    if x['split'] == y['split'] or (a == b and x['id'] >= y['id']):
                        continue
                    affected.update([x['id'], y['id']])
                    pairs.append({'left': x['id'], 'right': y['id'],
                                  'splits': [x['split'], y['split']],
                                  'edit_similarity': similarity,
                                  'same_chemistry': None if any(r['chemistry'].casefold() in ('unknown','unspecified') for r in (x,y)) else x['chemistry'] == y['chemistry']})
    return {'n_rows': len(rows), 'n_unique_sequences': len(sequences),
            'threshold': threshold, 'screened_sequence_pairs': checked,
            'cross_split_pairs': pairs, 'affected_ids': sorted(affected),
            'chemistry_or_label_collisions': collisions,
            'interpretation': 'Edit-distance screen only; absence of flags does not establish absence of homology or pretraining contamination.'}

if __name__ == '__main__':
    run(analyze)
