"""Run the fixed low-MIC split-sensitivity study on a pinned public assay table."""
import argparse
import csv
import hashlib
import io
import json
import platform
from collections import Counter
from pathlib import Path
import urllib.request
import numpy as np
from rapidfuzz.distance import Levenshtein
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from bio_evidence.assay_units import threshold_label, parse_measurement
from split_groups import source_sequence_components, canonical_group_ids

SOURCE_COMMIT='5ac295c08f48aef43c991ba40a706f0e2fedb5b7'
SOURCE=f'https://raw.githubusercontent.com/aarnavbhat23/spear-wetlab-false-positive-audit/{SOURCE_COMMIT}/data/df_wetlab_validated.tsv'
MODEL='facebook/esm2_t6_8M_UR50D'
REVISION='c731040fcd8d73dceaa04b0a8e6329b345b0f5df'
AA='ACDEFGHIKLMNPQRSTVWY'
SEED=20261014

def prepare(raw):
    rows, exclusions=[],[]
    original=list(csv.DictReader(io.StringIO(raw.decode()),delimiter='\t'))
    for index,row in enumerate(original,1):
        label=threshold_label(row['mic'],16,'uM')
        reason=None
        if not row['paper_doi']:
            reason='missing_source'
        elif not row['sequence'] or not set(row['sequence']) <= set(AA):
            reason='noncanonical_sequence'
        elif label is None:
            parsed=parse_measurement(row['mic'])
            reason='missing_or_unparsed' if parsed is None else ('unsupported_mass_unit' if parsed['unit'] not in ('uM','nM','mM','M') else 'indeterminate_censoring')
        if reason:
            exclusions.append({'source_row':index,'reason':reason,'mic':row['mic']})
        else:
            rows.append({'id':f'assay-{index}','sequence':row['sequence'],'paper':row['paper_doi'],
                         'assay_description':row['wetlab_experiment'],'mic':row['mic'],'label':label})
    return rows,exclusions,len(original)

def composition(seq):
    return [seq.count(a)/len(seq) for a in AA]+[len(seq)]

def encode(sequences,device):
    import torch
    from transformers import AutoModel,AutoTokenizer
    torch.manual_seed(SEED)
    if device=='cpu': torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model=AutoModel.from_pretrained(MODEL,revision=REVISION,add_pooling_layer=False).to(device).eval()
    output=[]
    with torch.inference_mode():
        for start in range(0,len(sequences),32):
            batch=sequences[start:start+32]
            tokens=tokenizer(batch,padding=True,truncation=False,return_tensors='pt',return_special_tokens_mask=True)
            special=tokens.pop('special_tokens_mask').to(device)
            tokens={k:v.to(device) for k,v in tokens.items()}
            hidden=model(**tokens).last_hidden_state
            mask=tokens['attention_mask'].bool() & ~special.bool()
            pooled=(hidden*mask.unsqueeze(-1)).sum(1)/mask.sum(1).unsqueeze(-1)
            output.extend(pooled.cpu().numpy())
    return np.asarray(output)

def auc_interval(rows,labels,predictions,n=2000):
    papers=sorted({r['paper'] for r in rows})
    indices={p:np.array([i for i,r in enumerate(rows) if r['paper']==p]) for p in papers}
    rng=np.random.default_rng(SEED)
    values=[]
    for _ in range(n):
        sample=np.concatenate([indices[p] for p in rng.choice(papers,len(papers),replace=True)])
        if len(set(labels[sample]))==2:
            values.append(roc_auc_score(labels[sample],predictions[sample]))
    return {'low':float(np.quantile(values,.025)),'high':float(np.quantile(values,.975)),
            'n_valid_bootstrap':len(values),'method':'paper-block bootstrap of fixed OOF predictions'}

def assess(rows,features,splits,scheme,model_name):
    y=np.array([r['label'] for r in rows])
    prediction=np.full(len(rows),np.nan)
    fold_ids=np.full(len(rows),-1)
    fold_reports=[]
    exact_overlap=np.zeros(len(rows),dtype=bool)
    for fold,(train,test) in enumerate(splits):
        if set(train)&set(test): raise RuntimeError('Overlapping train/test indices')
        if len(set(y[train]))<2: raise ValueError('Training fold contains one class; protocol is not estimable')
        classifier=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=SEED))
        classifier.fit(features[train],y[train])
        probability=classifier.predict_proba(features[test])[:,1]
        prediction[test]=probability; fold_ids[test]=fold
        train_sequences={rows[i]['sequence'] for i in train}
        exact_overlap[test]=[rows[i]['sequence'] in train_sequences for i in test]
        fold_reports.append({'fold':fold,'n_train':len(train),'n_test':len(test),
                             'test_positive_fraction':float(y[test].mean()),
                             'auc':float(roc_auc_score(y[test],probability)) if len(set(y[test]))==2 else None,
                             'exact_sequence_overlap_fraction':float(exact_overlap[test].mean()),
                             'n_shared_papers':len({rows[i]['paper'] for i in train}&{rows[i]['paper'] for i in test})})
    if not np.isfinite(prediction).all() or (fold_ids<0).any(): raise RuntimeError('Incomplete OOF coverage')
    report={'scheme':scheme,'model':model_name,'auc':float(roc_auc_score(y,prediction)),
            'average_precision':float(average_precision_score(y,prediction)),
            'balanced_accuracy_at_0_5':float(balanced_accuracy_score(y,prediction>=.5)),
            'exact_sequence_overlap_fraction':float(exact_overlap.mean()),
            'auc_95_interval':auc_interval(rows,y,prediction),'folds':fold_reports}
    ledger=[{'id':r['id'],'paper':r['paper'],'sequence':r['sequence'],'label':int(y[i]),
             'probability':float(prediction[i]),'fold':int(fold_ids[i]),'scheme':scheme,
             'model':model_name,'exact_sequence_in_training':bool(exact_overlap[i])} for i,r in enumerate(rows)]
    return report,ledger

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-file',type=Path,help='Local copy of the pinned TSV; its hash is recorded')
    parser.add_argument('--output-dir',type=Path,default=Path('results/split-study'))
    parser.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu')
    parser.add_argument('--skip-esm',action='store_true')
    args=parser.parse_args()
    protocol=(Path(__file__).parent/'study-protocol.json').read_bytes()
    raw=args.source_file.read_bytes() if args.source_file else urllib.request.urlopen(SOURCE,timeout=60).read()
    rows,exclusions,total=prepare(raw)
    seqs=sorted({r['sequence'] for r in rows})
    edges=[]
    for i,a in enumerate(seqs):
        for b in seqs[i+1:]:
            if min(len(a),len(b))/max(len(a),len(b)) >= .9 and Levenshtein.normalized_similarity(a,b)>=.9:
                edges.append((a,b))
    exact_groups,exact_components=source_sequence_components(rows)
    near_groups,near_components=source_sequence_components(rows,edges)
    y=np.array([r['label'] for r in rows])
    schemes={'random_row':list(StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED).split(np.zeros(len(rows)),y))}
    for name,groups in [('sequence',[r['sequence'] for r in rows]),('paper',[r['paper'] for r in rows]),
                        ('joint_exact',exact_groups),('joint_edit_0.9',near_groups)]:
        groups=canonical_group_ids(rows,groups)
        n=min(5,len(set(groups)))
        if n<2: raise ValueError('All sources connected: joint holdout cannot be estimated')
        schemes[name]=list(GroupKFold(n_splits=n).split(np.zeros(len(rows)),y,groups))
    models={'composition_length':np.array([composition(r['sequence']) for r in rows])}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if not args.skip_esm:
        embeddings=encode(seqs,args.device)
        np.savez_compressed(args.output_dir/'embeddings.npz',sequences=np.array(seqs),embeddings=embeddings)
        mapping={s:e for s,e in zip(seqs,embeddings)}
        models['esm2_8m_frozen']=np.array([mapping[r['sequence']] for r in rows])
    reports,ledger=[],[]
    for scheme,splits in schemes.items():
        for name,features in models.items():
            report,predictions=assess(rows,features,splits,scheme,name)
            reports.append(report); ledger.extend(predictions)
            print(f"{scheme}/{name}: AUC={report['auc']:.3f}; exact overlap={report['exact_sequence_overlap_fraction']:.1%}",flush=True)
    import importlib.metadata
    metadata={'source_url':SOURCE,'source_commit':SOURCE_COMMIT,'source_sha256':hashlib.sha256(raw).hexdigest(),
              'protocol_sha256':hashlib.sha256(protocol).hexdigest(),'n_source_rows':total,'n_retained':len(rows),
              'n_sequences':len(seqs),'n_papers':len(set(r['paper'] for r in rows)),
              'n_positive':int(y.sum()),'n_above_threshold':int((1-y).sum()),
              'exclusions':dict(Counter(e['reason'] for e in exclusions)),
              'n_edit_similarity_edges':len(edges),'n_joint_exact_components':len(exact_components),
              'n_joint_near_components':len(near_components),'model':MODEL,'model_revision':REVISION,
              'group_label_policy':'Canonicalize group partitions by sorted member row IDs before splitting; equivalent partitions give identical folds.',
              'python':platform.python_version(),'device':args.device,'seed':SEED,
              'packages':{p:importlib.metadata.version(p) for p in ('numpy','scikit-learn','rapidfuzz','torch','transformers')},
              'claim_boundary':'Pooled-assay threshold task on a positive-only source. Not antimicrobial vs non-antimicrobial classification. Organism/lab confounding, unknown chemistry and pretraining overlap are unresolved. Protocol fixed locally before the run, not externally preregistered.'}
    for filename,content in [('report.json',{'runs':reports,'manifest':metadata}),('input.json',{'rows':rows}),
                             ('exclusions.json',exclusions),('components.json',{'exact':exact_components,'edit_0.9':near_components,'edges':edges})]:
        (args.output_dir/filename).write_text(json.dumps(content,indent=2,allow_nan=False)+'\n')
    with (args.output_dir/'predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(ledger[0])); writer.writeheader(); writer.writerows(ledger)

if __name__=='__main__':
    main()
