"""Run the released ensemble locally; prediction is not experimental evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
from model import PotencyHead
from encode import MODEL,REVISION
from train import numeric

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path,help='JSON containing sequence, organism and optional explicit modification metadata')
    p.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu')
    args=p.parse_args()
    row=json.loads(args.input.read_text())
    if not row.get('organism') or not 8<=len(row.get('sequence',''))<=100 or not set(row['sequence'])<=set('ACDEFGHIKLMNPQRSTVWY'):
        p.error('Require organism and canonical sequence of 8-100 residues')
    for key in ('modification_recorded','is_modified','unusual_modification','c_amidated'):
        value=row.get(key,False)
        if not isinstance(value,bool):p.error(key+' must be a Boolean')
        row[key]=str(value)
    row.setdefault('modifications','unknown')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    encoder=AutoModel.from_pretrained(MODEL,revision=REVISION,add_pooling_layer=False).to(args.device).eval()
    tokens=tokenizer(row['sequence'],return_tensors='pt',return_special_tokens_mask=True)
    special=tokens.pop('special_tokens_mask').to(args.device)
    tokens={k:v.to(args.device) for k,v in tokens.items()}
    with torch.inference_mode():
        hidden=encoder(**tokens).last_hidden_state
        mask=tokens['attention_mask'].bool() & ~special.bool()
        representation=((hidden*mask.unsqueeze(-1)).sum(1)/mask.sum(1).unsqueeze(-1)).cpu().numpy()[0]
    features=np.concatenate([representation,numeric(row)])
    root=Path(__file__).resolve().parent
    means=[];sigmas=[];known=None
    for path in sorted((root/'results/checkpoints').glob('seed-*.pt')):
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        model=PotencyHead(checkpoint['feature_dim'],checkpoint['n_organisms']).eval()
        model.load_state_dict(checkpoint['state_dict'])
        standardized=(features-checkpoint['feature_mean'].numpy())/checkpoint['feature_scale'].numpy()
        index=checkpoint['organism_vocabulary'].get(row['organism'],0);known=index!=0
        with torch.inference_mode():
            mean,sigma=model(torch.tensor(standardized[None,:],dtype=torch.float32),torch.tensor([index]))
        means.append(float(mean)*checkpoint['target_scale']+checkpoint['target_mean'])
        sigmas.append(float(sigma)*checkpoint['target_scale'])
    if not means:p.error('No released ensemble checkpoints found')
    mean=float(np.mean(means));sigma=float(np.sqrt(max(np.mean(np.array(sigmas)**2+np.array(means)**2)-mean**2,1e-6)))
    report=json.loads((root/'results/report.json').read_text())
    q=report['calibration']['component_max_quantile']
    print(json.dumps({'predicted_log10_mic_uM':mean,'component_calibrated_interval_log10_uM':[mean-q*sigma,mean+q*sigma],
                      'organism_seen_in_training':known,'modification_metadata_recorded':row['modification_recorded']=='True',
                      'interpretation':'Model prediction on retrospective source data. Not an experimentally measured MIC, safety assessment or guarantee of coverage for this query.'},indent=2))

if __name__=='__main__':main()
