from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta
ROOT=Path(__file__).resolve().parent;FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','axes.labelcolor':'#102A43','text.color':'#102A43'})
analysis=json.loads((ROOT/'results/analysis.json').read_text());pc={r['method']:r for r in analysis['primary_cell']}
labels={'retain':'Retain baseline','ips_h':'IPS / Hoeffding','robust_ips_h':'Corrected IPS / Hoeffding','centered_dr_b':'Centered DR / Bernstein','robust_centered_dr_b':'Corrected DR / Bernstein','tabular_h':'Tabular / Hoeffding','robust_tabular_h':'Corrected tabular','sn_eslb':'Published SN-ESLB adapter','robust_sn_eslb':'Corrected SN-ESLB adapter'}
order=['robust_centered_dr_b','ips_h','centered_dr_b','tabular_h','robust_ips_h','robust_tabular_h','sn_eslb','robust_sn_eslb','retain']
fig,ax=plt.subplots(figsize=(7.4,4.2));y=np.arange(len(order));values=np.array([pc[m]['mean_improvement']*100 for m in order]);lo=np.array([pc[m]['improvement_low']*100 for m in order]);hi=np.array([pc[m]['improvement_high']*100 for m in order])
ax.barh(y,values,color=['#007F7A']+['#7591A3']*8,height=.67);ax.errorbar(values,y,xerr=np.maximum(0,np.array([values-lo,hi-values])),fmt='none',ecolor='#102A43',capsize=2,lw=1)
ax.set_yticks(y,labels=[labels[m] for m in order]);ax.invert_yaxis();ax.set_xlabel('Mean worst-attacked true improvement (percentage points)');ax.set_xlim(0,4.25);ax.set_xticks([0,1,2,3,4]);ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
for i,v in enumerate(values):ax.text(max(v,.0)+.18,i,f'{v:.2f}',va='center',fontsize=9)
ax.set_title('Primary condition: 0.5% edits, 5% rare-action coverage',loc='left',fontsize=11,pad=15)
fig.tight_layout();fig.savefig(FIG/'primary_comparison.png',dpi=240,bbox_inches='tight');fig.savefig(FIG/'primary_comparison.svg',bbox_inches='tight');plt.close(fig)

front=json.loads((ROOT/'frontier_results/frontier.json').read_text());rows=list(csv.DictReader((ROOT/'frontier_results/runs.csv').open()));rng=np.random.default_rng(816);ix=rng.integers(0,100,size=(20000,100))
for a in front:
    values=np.array([np.mean([float(r['improvement']) for r in rows if float(r['multiplier'])==a['multiplier'] and int(r['seed'])==s]) for s in range(3000,3100)])
    means=values[ix].mean(axis=1);a['clustered_bootstrap_ci']=np.quantile(means,[.025,.975]).tolist()
    a['values_by_seed']=values.tolist()
    a['simultaneous_upper_harm_if_zero']=1-(.05/8)**(1/100) if a['max_cell_harm']==0 else None
selected=next(a for a in front if a['multiplier']==1)
for a in front:
    d=np.array(selected['values_by_seed'])-a['values_by_seed'];a['paired_gain_of_correct_budget']=float(d.mean());a['paired_gain_ci']=np.quantile(d[ix].mean(axis=1),[.025,.975]).tolist()
(ROOT/'frontier_results/analysis.json').write_text(json.dumps(front,indent=2))
fig,ax=plt.subplots(figsize=(6.7,4.3))
off={0:(-36,-18),.25:(-36,8),.5:(-35,10),.75:(8,-8),1:(10,8),1.5:(10,-12)}
for a in front:
    x=100*a['max_cell_harm'];y=100*a['mean_improvement'];v=a['multiplier'];color='#007F7A' if v>=1 else '#B74E43'
    ci=np.array(a['clustered_bootstrap_ci'])*100
    ax.errorbar(x,y,yerr=[[y-ci[0]],[ci[1]-y]],fmt='o',color=color,markersize=8 if v==1 else 6,capsize=3)
    ax.annotate(f'{v:g}x budget', (x,y),xytext=off[v],textcoords='offset points',fontsize=9,color=color)
ax.set_xlim(-4,109);ax.set_ylim(-.25,.52);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Worst-setting harmful deployments (%)');ax.set_ylabel('Mean worst-attacked improvement (pp)');ax.grid(alpha=.17)
ax.set_title('Fresh-seed Pareto comparison',loc='left',fontsize=12,pad=14)
fig.tight_layout();fig.savefig(FIG/'pareto_frontier.png',dpi=240,bbox_inches='tight');fig.savefig(FIG/'pareto_frontier.svg',bbox_inches='tight');plt.close(fig)

ss=json.loads((ROOT/'results/summary.json').read_text());methods=['ips_h','robust_ips_h','centered_dr_b','robust_centered_dr_b','tabular_h','robust_tabular_h']
colors=['#839CA8','#9BB0BD','#B74E43','#007F7A','#9671A2','#C69D55'];line_styles=['--',':','--','-', '--',':']
for name in ['diverse_matched_gain','original_control']:
    fig,axes=plt.subplots(2,3,figsize=(11.5,5.8),sharex=True,sharey='row')
    for col,eta in enumerate([.5,.2,.05]):
        axes[0,col].set_title(f'Coverage {eta:.0%}',fontsize=11)
        for m,c,ls in zip(methods,colors,line_styles):
            r=sorted([r for r in ss if r['environment']==name and r['eta']==eta and r['method']==m],key=lambda r:r['epsilon'])
            x=[a['epsilon']*100 for a in r]
            for row,key in [(0,'harm_rate'),(1,'mean_improvement')]:axes[row,col].plot(x,[a[key]*100 for a in r],color=c,ls=ls,marker='o',ms=3,label=labels[m])
        axes[1,col].axhline(0,color='#627487',lw=.7);axes[1,col].set_xlabel('Rewards editable (%)');axes[1,col].set_xticks([0,.5,1,2.5,5]);axes[0,col].set_ylim(-3,104);axes[1,col].set_ylim(-4.3,4.3)
        for row in [0,1]:axes[row,col].grid(alpha=.15)
    axes[0,0].set_ylabel('Harmful deployment (%)');axes[1,0].set_ylabel('Worst-attacked improvement (pp)')
    handles,labs=axes[0,0].get_legend_handles_labels();fig.legend(handles,labs,loc='lower center',ncol=3,frameon=False,fontsize=9,bbox_to_anchor=(.5,-.015));fig.subplots_adjust(left=.07,right=.99,top=.93,bottom=.18,wspace=.12,hspace=.20)
    fig.savefig(FIG/(name+'_grid.png'),dpi=220,bbox_inches='tight');fig.savefig(FIG/(name+'_grid.svg'),bbox_inches='tight');plt.close(fig)
print('Figures ready. Correct-cap mean pp:',100*selected['mean_improvement'],'simultaneous upper harm:',selected['simultaneous_upper_harm_if_zero'])
