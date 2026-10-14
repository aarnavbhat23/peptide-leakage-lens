"""Summarize the completed run, including paired component bootstrap uncertainty."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    root=Path(__file__).resolve().parent
    report=json.loads((root/'results/report.json').read_text())
    with (root/'results/test-predictions.csv').open() as f:rows=list(csv.DictReader(f))
    y=np.array([float(r['observed_log10_mic_uM']) for r in rows])
    neural=np.array([float(r['predicted_log10_mic_uM']) for r in rows])
    boost=np.array([float(r['boosting_prediction']) for r in rows])
    groups=sorted({r['component'] for r in rows})
    indices={g:np.array([i for i,r in enumerate(rows) if r['component']==g]) for g in groups}
    rng=np.random.default_rng(20261014);differences=[]
    for _ in range(2000):
        selected=np.concatenate([indices[g] for g in rng.choice(groups,len(groups),replace=True)])
        differences.append(np.sqrt(np.mean((y[selected]-neural[selected])**2))-np.sqrt(np.mean((y[selected]-boost[selected])**2)))
    report['paired_component_bootstrap']={'comparison':'neural RMSE minus gradient-boosting RMSE',
                                         'replicates':2000,'estimate':float(np.sqrt(np.mean((y-neural)**2))-np.sqrt(np.mean((y-boost)**2))),
                                         'interval_95':[float(np.quantile(differences,.025)),float(np.quantile(differences,.975))],
                                         'limitations':'Fixed fitted models; resamples test components only; excludes retraining and source/lab uncertainty.'}
    (root/'results/report.json').write_text(json.dumps(report,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    models=report['test_metrics'];keys=list(models)
    names={'species_mean':'Organism mean','composition_chemistry_species_ridge':'Composition + ridge',
           'composition_chemistry_species_hist_gradient_boosting':'Composition + boosting',
           'esm35m_neural_ensemble':'ESM-2 + neural ensemble','esm35m_chemistry_species_ridge':'ESM-2 + ridge'}
    axes[0].barh(np.arange(len(keys)),[models[k]['rmse_log10_uM'] for k in keys],color=['#197875' if k=='esm35m_neural_ensemble' else '#b7aaa0' for k in keys])
    axes[0].set(yticks=np.arange(len(keys)),yticklabels=[names[k] for k in keys],xlabel='Test RMSE in log10(MIC / µM)',xlim=(0,.9))
    axes[0].invert_yaxis()
    for i,k in enumerate(keys):axes[0].text(models[k]['rmse_log10_uM']+.008,i,f"{models[k]['rmse_log10_uM']:.3f}",va='center',fontsize=9)
    intervals=report['test_intervals']
    x=np.arange(2)
    axes[1].bar(x-.16,[intervals[k]['row_coverage']*100 for k in intervals],width=.3,label='Individual assay rows',color='#b7aaa0')
    axes[1].bar(x+.16,[intervals[k]['component_simultaneous_coverage']*100 for k in intervals],width=.3,label='All rows in a component',color='#197875')
    axes[1].axhline(90,color='#b55b43',linestyle='--',label='Nominal 90%')
    axes[1].set(xticks=x,xticklabels=['Row calibration','Component calibration'],ylabel='Observed test coverage (%)',ylim=(0,108))
    axes[1].legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,1.02),ncol=1,fontsize=9)
    fig.suptitle('A larger peptide benchmark: prediction error and coverage unit',fontweight='bold')
    fig.supxlabel('49,172 eligible GRAMPA records • 6,391 peptides • 750 source organism labels\n6,473 test records in 417 held-out edit components; database/lab dependence remains unresolved',fontsize=9)
    for extension in ('png','svg'):fig.savefig(root/'results'/f'benchmark.{extension}',dpi=220)
    plt.close(fig)
    print(report['paired_component_bootstrap'])

if __name__=='__main__':main()
