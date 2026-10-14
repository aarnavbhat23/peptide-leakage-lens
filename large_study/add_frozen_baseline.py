"""Add the fixed same-representation ridge comparator without retraining heads."""
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from train import numeric,regression_metrics

def main():
    root=Path(__file__).resolve().parent
    with gzip.open(root/'data/records.csv.gz','rt') as f: rows=list(csv.DictReader(f))
    data=np.load(root/'data/embeddings.npz',allow_pickle=False)
    mapping=dict(zip(data['sequences'],data['vectors']))
    features=np.array([np.concatenate([mapping[r['sequence']],numeric(r)]) for r in rows],dtype=np.float32)
    train=np.array([i for i,r in enumerate(rows) if r['split']=='train'])
    test=np.array([i for i,r in enumerate(rows) if r['split']=='test'])
    y=np.array([float(r['log10_mic_uM']) for r in rows])
    scaler=StandardScaler().fit(features[train])
    x=scaler.transform(features)
    species=np.array([r['organism'] for r in rows])[:,None]
    encoder=OneHotEncoder(handle_unknown='ignore').fit(species[train])
    x=sparse.hstack([sparse.csr_matrix(x),encoder.transform(species)],format='csr')
    model=Ridge(alpha=10).fit(x[train],y[train])
    prediction=model.predict(x[test])
    report=json.loads((root/'results/report.json').read_text())
    report['test_metrics']['esm35m_chemistry_species_ridge']=regression_metrics(y[test],prediction)
    report['baseline_note']='Same-representation ridge(alpha=10) added as a fixed comparator after initial head training. No head or hyperparameter was selected using this test comparison.'
    with (root/'results/test-predictions.csv').open() as f: records=list(csv.DictReader(f))
    if [r['id'] for r in records] != [rows[i]['id'] for i in test]:raise RuntimeError('Prediction row alignment mismatch')
    for row,value in zip(records,prediction):row['frozen_ridge_prediction']=float(value)
    with (root/'results/test-predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (root/'results/report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['test_metrics']['esm35m_chemistry_species_ridge'])

if __name__=='__main__':main()
