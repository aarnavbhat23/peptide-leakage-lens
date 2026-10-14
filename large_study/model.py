"""Species-conditioned neural potency head on frozen protein-transformer features."""
import torch
from torch import nn

class PotencyHead(nn.Module):
    def __init__(self,feature_dim,n_organisms):
        super().__init__()
        self.organism=nn.Embedding(n_organisms+1,32,padding_idx=0)
        self.network=nn.Sequential(nn.Linear(feature_dim+32,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.15),
                                   nn.Linear(256,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,2))
    def forward(self,features,organism):
        output=self.network(torch.cat([features,self.organism(organism)],dim=1))
        mean=output[:,0]
        sigma=torch.nn.functional.softplus(output[:,1])+.05
        return mean,sigma

def gaussian_loss(mean,sigma,target):
    return (torch.log(sigma)+.5*((target-mean)/sigma).square()).mean()
