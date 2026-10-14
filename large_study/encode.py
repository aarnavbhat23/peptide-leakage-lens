"""Encode every unique eligible peptide using a pinned 35M protein transformer."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer

MODEL='facebook/esm2_t12_35M_UR50D'
REVISION='6fbf070e65b0b7291e7bbcd451118c216cff79d8'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path(__file__).resolve().parent/'data')
    p.add_argument('--device',choices=['cpu','cuda','mps'],default='mps')
    args=p.parse_args()
    with gzip.open(args.data/'records.csv.gz','rt') as f: records=list(csv.DictReader(f))
    sequences=sorted({r['sequence'] for r in records})
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model=AutoModel.from_pretrained(MODEL,revision=REVISION,add_pooling_layer=False).to(args.device).eval()
    torch.manual_seed(20261014)
    if args.device=='cpu':torch.set_num_threads(4)
    vectors=[]
    with torch.inference_mode():
        for start in range(0,len(sequences),32):
            tokens=tokenizer(sequences[start:start+32],padding=True,truncation=False,return_tensors='pt',return_special_tokens_mask=True)
            special=tokens.pop('special_tokens_mask').to(args.device)
            tokens={k:v.to(args.device) for k,v in tokens.items()}
            hidden=model(**tokens).last_hidden_state
            mask=tokens['attention_mask'].bool() & ~special.bool()
            pooled=(hidden*mask.unsqueeze(-1)).sum(1)/mask.sum(1).unsqueeze(-1)
            vectors.extend(pooled.cpu().numpy())
            if start%1024==0:print(f'Encoded {start}/{len(sequences)} peptides on {args.device}',flush=True)
    np.savez_compressed(args.data/'embeddings.npz',sequences=np.array(sequences),vectors=np.array(vectors,dtype=np.float32))
    manifest={'model':MODEL,'revision':REVISION,'n_sequences':len(sequences),'dimension':len(vectors[0]),
              'device':args.device,'pooling':'final-layer residue mean; BOS/EOS/padding excluded','torch':torch.__version__,
              'file_sha256':hashlib.sha256((args.data/'embeddings.npz').read_bytes()).hexdigest()}
    (args.data/'embedding-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
