"""Train a frozen-transformer neural ensemble and assess cluster-level uncertainty."""
import argparse
import csv
import gzip
import json
import math
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from model import PotencyHead,gaussian_loss

AA='ACDEFGHIKLMNPQRSTVWY'
SEED=20261014

def chemistry(row):
    flags=[row[k]=='True' for k in ('modification_recorded','is_modified','unusual_modification','c_amidated')]
    flags += ['disulfide' in row['modifications'].lower(),'N-Terminus: ACT' in row['modifications']]
    return [float(v) for v in flags]

def numeric(row):
    seq=row['sequence']
    return [seq.count(a)/len(seq) for a in AA]+[len(seq)]+chemistry(row)

def regression_metrics(y,pred):
    return {'n':len(y),'mae_log10_uM':float(mean_absolute_error(y,pred)),
            'rmse_log10_uM':float(math.sqrt(mean_squared_error(y,pred))),
            'pearson':float(pearsonr(y,pred).statistic) if len(y)>2 and np.std(y)>0 and np.std(pred)>0 else None,
            'r_squared':float(r2_score(y,pred))}

def conformal_quantile(scores,coverage=.9):
    """Finite-sample rank, with infinity if too few calibration units exist."""
    scores=np.sort(np.asarray(scores,dtype=float))
    rank=math.ceil((len(scores)+1)*coverage)
    if rank>len(scores): return math.inf
    return float(scores[rank-1])

def calibration(scores,groups):
    maxima=defaultdict(list)
    for score,group in zip(scores,groups): maxima[group].append(float(score))
    return conformal_quantile([max(v) for v in maxima.values()]),len(maxima)

def coverage_metrics(y,mean,sigma,q,groups):
    inside=np.abs(y-mean)<=q*sigma
    family={}
    for yes,group in zip(inside,groups): family[group]=family.get(group,True) and bool(yes)
    return {'row_coverage':float(inside.mean()),'component_simultaneous_coverage':sum(family.values())/len(family),
            'n_components':len(family),'median_interval_width_log10_uM':float(np.median(2*q*sigma))}

def train_one(seed,x,organism,y,train,validation,device,output,max_epochs=40):
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    n_species=int(organism.max())
    model=PotencyHead(x.shape[1],n_species).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    tx=torch.from_numpy(x).to(device); ts=torch.from_numpy(organism).to(device); ty=torch.from_numpy(y).to(device)
    best=math.inf;best_state=None;patience=0;history=[]
    start=time.perf_counter()
    for epoch in range(max_epochs):
        model.train(); losses=[]
        for indices in np.array_split(rng.permutation(train),math.ceil(len(train)/512)):
            idx=torch.from_numpy(indices).to(device)
            mean,sigma=model(tx[idx],ts[idx]);loss=gaussian_loss(mean,sigma,ty[idx])
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            idx=torch.from_numpy(validation).to(device)
            mu,sd=model(tx[idx],ts[idx]);val=float(gaussian_loss(mu,sd,ty[idx]).cpu())
        history.append({'seed':seed,'epoch':epoch+1,'train_nll':float(np.mean(losses)),'validation_nll':val})
        if val<best-1e-4:
            best=val;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};patience=0
        else: patience+=1
        if epoch%5==0:print(f'seed={seed} epoch={epoch+1} validation NLL={val:.3f}',flush=True)
        if patience>=8:break
    model.load_state_dict(best_state);model.eval()
    means=[];sigmas=[]
    with torch.inference_mode():
        for batch in np.array_split(np.arange(len(x)),math.ceil(len(x)/1024)):
            idx=torch.from_numpy(batch).to(device)
            mean,sigma=model(tx[idx],ts[idx]);means.extend(mean.cpu().numpy());sigmas.extend(sigma.cpu().numpy())
    return np.array(means),np.array(sigmas),best_state,history,{'seed':seed,'epochs':len(history),'best_validation_nll':best,
                                                                  'trainable_parameters':sum(p.numel() for p in model.parameters()),
                                                                  'training_seconds':time.perf_counter()-start}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path(__file__).resolve().parent/'data')
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results')
    p.add_argument('--device',choices=['cpu','cuda','mps'],default='mps')
    p.add_argument('--max-epochs',type=int,default=40)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with gzip.open(args.data/'records.csv.gz','rt') as f:rows=list(csv.DictReader(f))
    matrix=np.load(args.data/'embeddings.npz',allow_pickle=False)
    representation={s:e for s,e in zip(matrix['sequences'],matrix['vectors'])}
    frozen=np.array([representation[r['sequence']] for r in rows],dtype=np.float32)
    numbers=np.array([numeric(r) for r in rows],dtype=np.float32)
    features=np.concatenate([frozen,numbers],axis=1)
    indices={s:np.array([i for i,r in enumerate(rows) if r['split']==s],dtype=np.int64) for s in ('train','validation','calibration','test')}
    for a,ia in indices.items():
        for b,ib in indices.items():
            if a>=b:continue
            if {rows[i]['component'] for i in ia}&{rows[i]['component'] for i in ib}:raise RuntimeError('Component crosses splits')
    train,validation,cal,test=[indices[s] for s in ('train','validation','calibration','test')]
    scaler=StandardScaler().fit(features[train])
    x=scaler.transform(features).astype(np.float32)
    vocabulary={s:i+1 for i,s in enumerate(sorted({rows[j]['organism'] for j in train}))}
    organisms=np.array([vocabulary.get(r['organism'],0) for r in rows],dtype=np.int64)
    target=np.array([float(r['log10_mic_uM']) for r in rows],dtype=np.float32)
    y_mean,y_scale=float(target[train].mean()),float(target[train].std())
    y=(target-y_mean)/y_scale
    if args.device=='cpu':torch.set_num_threads(4)
    checkpoints=args.output/'checkpoints';checkpoints.mkdir(exist_ok=True)
    all_means=[];all_sigmas=[];history=[];training=[]
    for seed in (SEED,SEED+1,SEED+2):
        mu,sd,state,log,metadata=train_one(seed,x,organisms,y,train,validation,args.device,args.output,args.max_epochs)
        all_means.append(mu*y_scale+y_mean);all_sigmas.append(sd*y_scale);history.extend(log);training.append(metadata)
        torch.save({'state_dict':state,'feature_dim':x.shape[1],'n_organisms':len(vocabulary),
                    'feature_mean':torch.from_numpy(scaler.mean_),'feature_scale':torch.from_numpy(scaler.scale_),
                    'target_mean':y_mean,'target_scale':y_scale,'organism_vocabulary':vocabulary},checkpoints/f'seed-{seed}.pt')
    means=np.asarray(all_means);sigmas=np.asarray(all_sigmas)
    prediction=means.mean(axis=0)
    variance=np.mean(sigmas**2+means**2,axis=0)-prediction**2
    uncertainty=np.sqrt(np.maximum(variance,1e-6)).clip(.02)
    # Fixed, train-fitted baselines use the same target organisms and chemistry flags.
    category=OneHotEncoder(handle_unknown='ignore',sparse_output=True).fit(np.array([r['organism'] for r in rows])[train,None])
    onehot=category.transform(np.array([r['organism'] for r in rows])[:,None])
    number_scaler=StandardScaler().fit(numbers[train])
    base=sparse.hstack([sparse.csr_matrix(number_scaler.transform(numbers)),onehot],format='csr')
    ridge=Ridge(alpha=10).fit(base[train],target[train]);ridge_prediction=ridge.predict(base[test])
    frozen_base=sparse.hstack([sparse.csr_matrix(x),onehot],format='csr')
    frozen_ridge=Ridge(alpha=10).fit(frozen_base[train],target[train])
    frozen_ridge_prediction=frozen_ridge.predict(frozen_base[test])
    dense=base.toarray().astype(np.float32)
    boost=HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=31,l2_regularization=1,early_stopping=False,random_state=SEED)
    boost.fit(dense[train],target[train]);boost_prediction=boost.predict(dense[test])
    scores=np.abs(target[cal]-prediction[cal])/uncertainty[cal]
    row_q=conformal_quantile(scores)
    cluster_q,n_cal_clusters=calibration(scores,[rows[i]['component'] for i in cal])
    groups=[rows[i]['component'] for i in test]
    training_by_organism=defaultdict(list)
    for i in train: training_by_organism[rows[i]['organism']].append(float(target[i]))
    organism_means={k:float(np.mean(v)) for k,v in training_by_organism.items()}
    result={'dataset':json.loads((args.data/'manifest.json').read_text()),
            'encoder':json.loads((args.data/'embedding-manifest.json').read_text()),
            'training':training,'device':args.device,'max_epochs':args.max_epochs,
            'test_metrics':{'species_mean':regression_metrics(target[test],np.array([organism_means.get(rows[i]['organism'],y_mean) for i in test])),
                            'composition_chemistry_species_ridge':regression_metrics(target[test],ridge_prediction),
                            'esm35m_chemistry_species_ridge':regression_metrics(target[test],frozen_ridge_prediction),
                            'composition_chemistry_species_hist_gradient_boosting':regression_metrics(target[test],boost_prediction),
                            'esm35m_neural_ensemble':regression_metrics(target[test],prediction[test])},
            'calibration':{'nominal_coverage':.9,'n_calibration_components':n_cal_clusters,'row_quantile':row_q,'component_max_quantile':cluster_q},
            'test_intervals':{'row_calibrated':coverage_metrics(target[test],prediction[test],uncertainty[test],row_q,groups),
                              'component_calibrated':coverage_metrics(target[test],prediction[test],uncertainty[test],cluster_q,groups)},
            'test_unknown_organism_rows':int((organisms[test]==0).sum()),
            'protocol':'Fixed architecture/settings; choose checkpoints by validation NLL; interval multipliers use calibration only; test split is not used to select a model.',
            'claim_boundary':'Retrospective point-valued source; no censor-aware, clinical, certified homology-free, publication-held-out or unseen-species performance claim. Component-level conformal interpretation requires exchangeable components; source/lab dependencies remain unresolved.'}
    strata={}
    for name,mask in [('modification_recorded',[rows[i]['modification_recorded']=='True' for i in test]),
                      ('modification_unknown',[rows[i]['modification_recorded']!='True' for i in test]),
                      ('unusual_modification',[rows[i]['unusual_modification']=='True' for i in test])]:
        m=np.array(mask)
        if m.any():strata[name]=regression_metrics(target[test][m],prediction[test][m])
    result['chemistry_strata']=strata
    (args.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (args.output/'training.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    with (args.output/'test-predictions.csv').open('w',newline='') as f:
        fieldnames=['id','source_row','component','organism','observed_log10_mic_uM','predicted_log10_mic_uM','sigma_log10_uM','row_interval_low','row_interval_high','component_interval_low','component_interval_high','ridge_prediction','boosting_prediction','frozen_ridge_prediction']
        writer=csv.DictWriter(f,fieldnames=fieldnames);writer.writeheader()
        for pos,i in enumerate(test):
            mu,sd=float(prediction[i]),float(uncertainty[i])
            writer.writerow(dict(zip(fieldnames,[rows[i]['id'],rows[i]['source_row'],rows[i]['component'],rows[i]['organism'],float(target[i]),mu,sd,mu-row_q*sd,mu+row_q*sd,mu-cluster_q*sd,mu+cluster_q*sd,float(ridge_prediction[pos]),float(boost_prediction[pos]),float(frozen_ridge_prediction[pos])])) )
    print(json.dumps({'test_metrics':result['test_metrics'],'test_intervals':result['test_intervals']}),flush=True)

if __name__=='__main__': main()
