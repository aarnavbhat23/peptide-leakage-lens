"""Plot the saved study; no fitting or result selection occurs here."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    path=Path(__file__).resolve().parent/'results/split-study'
    data=json.loads((path/'report.json').read_text())
    schemes=['random_row','sequence','paper','joint_exact','joint_edit_0.9']
    labels=['Random rows','Sequence groups','Paper groups','Paper + exact','Paper + edit ≥0.9']
    colors={'composition_length':'#84766a','esm2_8m_frozen':'#197875'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.7),layout='constrained',gridspec_kw={'width_ratios':[1.5,1]})
    for offset,model,name in [(-.12,'composition_length','Composition + length'),(.12,'esm2_8m_frozen','Frozen ESM-2 8M')]:
        rows=[next(r for r in data['runs'] if r['scheme']==s and r['model']==model) for s in schemes]
        auc=np.array([r['auc'] for r in rows])
        low=np.array([r['auc_95_interval']['low'] for r in rows])
        high=np.array([r['auc_95_interval']['high'] for r in rows])
        axes[0].errorbar(auc,np.arange(5)+offset,xerr=[auc-low,high-auc],fmt='o',capsize=3,label=name,color=colors[model])
    axes[0].set(yticks=np.arange(5),yticklabels=labels,xlim=(.4,.8),xlabel='Pooled out-of-fold AUROC')
    axes[0].invert_yaxis(); axes[0].axvline(.5,linestyle=':',color='#bbb'); axes[0].legend(frameon=False,loc='lower right')
    overlap=[next(r for r in data['runs'] if r['scheme']==s)['exact_sequence_overlap_fraction']*100 for s in schemes]
    axes[1].barh(np.arange(5),overlap,color='#b55b43',height=.55)
    axes[1].set(yticks=np.arange(5),yticklabels=[],xlim=(0,105),xlabel='Test rows with an exact sequence in training (%)')
    axes[1].invert_yaxis()
    for i,value in enumerate(overlap): axes[1].text(value+2,i,f'{value:.1f}%',va='center')
    fig.suptitle('Peptide assay scores depend on the evaluation split',fontweight='bold')
    fig.supxlabel('1,947 molar-unit assay rows • 447 sequences • 16 papers\nEndpoint: reported MIC ≤16 µM; intervals resample papers over fixed OOF predictions',fontsize=9)
    for suffix in ('png','svg'): fig.savefig(path/f'split-sensitivity.{suffix}',dpi=220)
    plt.close(fig)

if __name__=='__main__':main()
