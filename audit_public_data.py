"""Reproducible exact cross-paper screen of an existing public assay dataset."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request
from tool import analyze

SOURCE_COMMIT = '5ac295c08f48aef43c991ba40a706f0e2fedb5b7'
SOURCE_URL = f'https://raw.githubusercontent.com/aarnavbhat23/spear-wetlab-false-positive-audit/{SOURCE_COMMIT}/data/df_wetlab_validated.tsv'

def prepare(raw):
    original = list(csv.DictReader(io.StringIO(raw.decode('utf-8')), delimiter='\t'))
    groups = {}
    for i,r in enumerate(original,1):
        key = (r['sequence'],r['paper_doi'])
        if not all(key):
            raise ValueError('Public source has an empty sequence or DOI')
        if key not in groups:
            groups[key] = {'id':f'source-row-{i}','sequence':key[0], 'split':key[1],
                           'chemistry':'unknown', 'source_row_numbers':[]}
        groups[key]['source_row_numbers'].append(i)
    return {'data_kind':'real public assay table; paper-as-split diagnostic; chemistry unspecified',
            'similarity_threshold':1.0,'rows':list(groups.values())},len(original)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('results/public-assay'))
    args=parser.parse_args()
    raw=urllib.request.urlopen(SOURCE_URL, timeout=60).read()
    data, count=prepare(raw)
    result=analyze(data)
    result['source']={'url':SOURCE_URL,'commit':SOURCE_COMMIT,'sha256':hashlib.sha256(raw).hexdigest(),
                      'n_original_assay_rows':count,'n_sequence_paper_groups':len(data['rows']),
                      'n_papers':len({r['split'] for r in data['rows']})}
    result['study_note']='Repeated peptides across papers are valid assays. This is relevant to a paper-held-out novelty claim, not evidence of false positives. Chemistry is absent in this five-column source and is never inferred.'
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'input.json').write_text(json.dumps(data,indent=2)+'\n')
    (args.output_dir/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'source_rows':count,'sequence_paper_groups':len(data['rows']),
                      'cross_paper_exact_pairs':len(result['cross_split_pairs']),
                      'affected_sequence_paper_groups':len(result['affected_ids'])}))

if __name__=='__main__':
    main()
