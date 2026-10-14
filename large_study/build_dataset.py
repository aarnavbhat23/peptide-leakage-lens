"""Build a full GRAMPA benchmark with chemistry metadata and edit-component splits."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import urllib.request
import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from sklearn.model_selection import GroupShuffleSplit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from split_groups import UnionFind

COMMIT='a6819dcd671291af8013627a2d2fbc32858346bc'
URL=f'https://raw.githubusercontent.com/zswitten/Antimicrobial-Peptides/{COMMIT}/data/grampa.csv'
AA=set('ACDEFGHIKLMNPQRSTVWY')
SEED=20261014

def prepare(raw):
    records=list(csv.DictReader(io.StringIO(raw.decode())))
    rows=[]; exclusions=[]
    for i,r in enumerate(records):
        reason=None
        try: value=float(r['value'])
        except ValueError: value=float('nan')
        if not math.isfinite(value): reason='nonfinite_target'
        elif r['unit']!='uM': reason='unsupported_unit'
        elif not 8<=len(r['sequence'])<=100 or not set(r['sequence'])<=AA: reason='length_or_noncanonical'
        elif not r['bacterium'].strip(): reason='missing_target_organism'
        if reason:
            exclusions.append({'source_row':i,'reason':reason}); continue
        rows.append({'id':f'grampa-{i}','source_row':i,'sequence':r['sequence'],
                     'organism':r['bacterium'],'strain':r['strain'],'log10_mic_uM':value,
                     'modifications':r['modifications'],'modification_recorded':r['datasource_has_modifications']=='True',
                     'is_modified':r['is_modified']=='True','unusual_modification':r['has_unusual_modification']=='True',
                     'c_amidated':r['has_cterminal_amidation']=='True','source_database':r['database'],'source_url':r['url_source']})
    return rows,exclusions,len(records)

def components(sequences,threshold=.8):
    """Chunked compiled edit screening avoids materializing an N x N matrix."""
    union=UnionFind(range(len(sequences))); edges=0
    for start in range(0,len(sequences),128):
        scores=process.cdist(sequences[start:start+128],sequences,
                             scorer=Levenshtein.normalized_similarity,score_cutoff=threshold,dtype=np.float32,workers=4)
        for local,line in enumerate(scores):
            i=start+local
            for j in np.flatnonzero(line>=threshold):
                if j>i: union.join(i,int(j)); edges+=1
        if start%1024==0: print(f'Cluster screen {start}/{len(sequences)} sequences',flush=True)
    return [union.find(i) for i in range(len(sequences))],edges

def split_indices(groups):
    groups=np.asarray(groups)
    remaining=np.arange(len(groups)); assignments={}
    for name,fraction,seed in [('test',.15,SEED),('calibration',.10/.85,SEED+1),('validation',.15/.75,SEED+2)]:
        keep,hold=next(GroupShuffleSplit(n_splits=1,test_size=fraction,random_state=seed).split(remaining,groups=groups[remaining]))
        for index in remaining[hold]: assignments[int(index)]=name
        remaining=remaining[keep]
    for index in remaining: assignments[int(index)]='train'
    return [assignments[i] for i in range(len(groups))]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-file',type=Path)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'data')
    args=p.parse_args()
    raw=args.source_file.read_bytes() if args.source_file else urllib.request.urlopen(URL,timeout=60).read()
    rows,exclusions,total=prepare(raw)
    sequences=sorted({r['sequence'] for r in rows})
    labels,n_edges=components(sequences)
    mapping=dict(zip(sequences,labels))
    groups=[mapping[r['sequence']] for r in rows]
    splits=split_indices(groups)
    for r,group,split in zip(rows,groups,splits):
        r['component']=f'edit80-{group}';r['split']=split
    # Assertions guard the actual benchmark, rather than assuming the split worked.
    for key in ('sequence','component'):
        membership={}
        for r in rows:
            if r[key] in membership and membership[r[key]]!=r['split']: raise RuntimeError('Split leakage: '+key)
            membership[r[key]]=r['split']
    args.output.mkdir(parents=True,exist_ok=True)
    with gzip.open(args.output/'records.csv.gz','wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (args.output/'split-ledger.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['id','source_row','component','split']);writer.writeheader()
        writer.writerows({k:r[k] for k in writer.fieldnames} for r in rows)
    from collections import Counter
    manifest={'source_url':URL,'source_commit':COMMIT,'source_sha256':hashlib.sha256(raw).hexdigest(),
              'n_source_records':total,'n_retained':len(rows),'n_unique_sequences':len(sequences),
              'n_organisms':len({r['organism'] for r in rows}),'n_components':len(set(groups)),
              'n_similarity_edges':n_edges,'split_counts':dict(Counter(splits)),
              'exclusion_counts':dict(Counter(r['reason'] for r in exclusions)),
              'seed':SEED,'similarity_metric':'1 - unit-cost Levenshtein distance / max length; threshold .8; transitive connected components',
              'target_definition':'The upstream loader applies log10 to MIC expressed in uM; value is modeled as log10(MIC/uM), not raw MIC or negative-log pMIC.',
              'target_source_code':f'https://github.com/zswitten/Antimicrobial-Peptides/blob/{COMMIT}/src/load_data.py',
              'limitations':['Original censoring operators are not present in this processed source; no censor-aware claim can be made.',
                             'Database provenance is not publication provenance; no paper-held-out claim.',
                             'Modification flags are inherited annotations, not complete molecular identities.',
                             'Edit-component separation is not certified homology separation.',
                             'Rows from multiple databases may repeat the same underlying experiments; independent-assay count is unknown.'],
              'release_note':'Build recipe, split ledger and result artifacts are public. The source-derived record snapshot stays local; underlying database terms are not relicensed as MIT.'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output/'exclusions.json').write_text(json.dumps(exclusions,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)

if __name__=='__main__': main()
