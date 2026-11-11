"""Plots from saved diagnostic records. Sampling is local and post-run only."""
from __future__ import annotations

import math

import numpy as np

from .diagnostic_trace import ALLOWED_OUTCOMES

POLICIES=("mira_sw_ucb","random_grouping","fixed_grouping")
LABELS=("MIRA / SW-UCB","Random","Fixed (a=4)")


def _array(db, sql, args=(), width=None):
    rows=db.execute(sql,args).fetchall()
    if not rows: return np.empty((0,width or 2))
    return np.asarray([[math.nan if v is None else float(v) for v in r] for r in rows],float)


def generate_plots(db, out, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    names=[]
    colors=plt.get_cmap('tab20')(np.linspace(0,1,len(ALLOWED_OUTCOMES)))

    def save(fig,name):
        fig.tight_layout();fig.savefig(out/name,dpi=145);plt.close(fig);names.append(name)

    def panels():
        return plt.subplots(1,3,figsize=(14,4),squeeze=False)

    def empty(ax): ax.text(.5,.5,'0 events / no eligible data',ha='center',transform=ax.transAxes)

    # Per-slot candidate and scheduled counts; time bins contain observed slots only.
    def counts_plot(name, policies):
        fig,axes=plt.subplots(len(policies),1,figsize=(10,3*len(policies)),squeeze=False)
        for ax,p in zip(axes[:,0],policies):
            a=_array(db,"SELECT CAST(time_s AS INT),SUM(candidate_edge_groups)/COUNT(DISTINCT slot),SUM(candidate_nonedge_groups)/COUNT(DISTINCT slot),SUM(scheduled_edge_groups)/COUNT(DISTINCT slot),SUM(scheduled_nonedge_groups)/COUNT(DISTINCT slot) FROM c WHERE policy=? GROUP BY CAST(time_s AS INT) ORDER BY 1",(p,),5)
            for i,label in enumerate(['candidate edge','candidate nonedge','scheduled edge','scheduled nonedge'],1): ax.plot(a[:,0],a[:,i],label=label,marker='.')
            ax.set(title=p,xlabel='Time (s), 1-second bins',ylabel='Groups per physical slot');ax.legend(fontsize=8)
        save(fig,name)
    counts_plot('01_edge_nonedge_counts_over_time.png',[POLICIES[0]])
    counts_plot('01b_edge_nonedge_counts_policy_comparison.png',POLICIES)
    fig,axes=panels()
    for ax,p,label in zip(axes[0],POLICIES,LABELS):
        a=_array(db,"SELECT CAST(time_s AS INT),SUM(scheduled_edge_groups)/NULLIF(SUM(candidate_edge_groups),0.0),SUM(scheduled_nonedge_groups)/NULLIF(SUM(candidate_nonedge_groups),0.0) FROM c WHERE policy=? GROUP BY CAST(time_s AS INT) ORDER BY 1",(p,),3)
        for i,s in enumerate(['edge','nonedge'],1): ax.plot(a[:,0],a[:,i],label=s)
        ax.set(title=label,xlabel='Time (s)',ylabel='Scheduled / candidate',ylim=(0,1.05));ax.legend()
    save(fig,'02_edge_nonedge_scheduling_rate.png')

    # Exact per-slot CSI quantiles, with NaN gaps when a class is absent.
    import csv
    csi={p:{e:[] for e in (0,1)} for p in POLICIES}
    with (out/'group_time_summary.csv').open(encoding='utf-8',newline='') as handle:
        for r in csv.DictReader(handle):
            csi[r['policy']][int(r['is_edge'])].append([int(r['slot']),float(r['time_s']),float(r['p10_gain_db']),float(r['median_gain_db']),float(r['p90_gain_db'])])
    fig,axes=panels()
    for ax,p,label in zip(axes[0],POLICIES,LABELS):
        n=int(db.execute('SELECT MAX(slot)+1 FROM c WHERE policy=?',(p,)).fetchone()[0]);times=np.arange(n)*float(config.get('experiment.slot_dt_s'))
        for e,color in [(0,'#1f77b4'),(1,'#d62728')]:
            values=np.full((n,3),np.nan)
            for row in csi[p][e]: values[row[0]]=row[2:]
            ax.plot(times,values[:,1],color=color,label='edge' if e else 'nonedge',lw=.8)
            ax.fill_between(times,values[:,0],values[:,2],color=color,alpha=.18)
        ax.set(title=label,xlabel='Time (s)',ylabel='Worst-user gain (dB), median / p10-p90');ax.legend()
    save(fig,'03_group_csi_over_time_edge_vs_nonedge.png');del csi
    fig,axes=panels()
    for ax,p,label in zip(axes[0],POLICIES,LABELS):
        for e in (0,1):
            x=np.fromiter((r[0] for r in db.execute('SELECT gain_db FROM g WHERE policy=? AND is_edge=?',(p,e))),float)
            if len(x):
                x.sort();inds=np.unique(np.linspace(0,len(x)-1,min(len(x),2000)).astype(int))
                ax.plot(x[inds],(inds+1)/len(x),label='edge' if e else 'nonedge')
        ax.set(title=label,xlabel='Worst-user gain (dB)',ylabel='ECDF (all observations)');ax.legend()
    save(fig,'04_csi_distribution_edge_vs_nonedge.png')

    # At most 100k sampled records per policy, without storing the entire trace.
    fields=['gain_db','max_elevation_deg','worst_user_elevation_deg','weighted_post_group_aoi_slots',
            'power','rate_margin_bps_hz','is_edge','transmission_attempted','gain_change_from_boundary_db',
            'csi_rank_changed_from_boundary','priority','interference','sinr','group_size']
    index={k:i for i,k in enumerate(fields)};index['outcome']=len(fields)
    samples={};rng=np.random.default_rng(0)
    case='CASE outcome_category '+ ' '.join(f"WHEN '{c}' THEN {i}" for i,c in enumerate(ALLOWED_OUTCOMES))+' END'
    for p in POLICIES:
        n,first,last=db.execute('SELECT COUNT(*),MIN(rowid),MAX(rowid) FROM g WHERE policy=?',(p,)).fetchone()
        if last-first+1!=n: raise ValueError('Diagnostic import must retain contiguous policy row ids')
        ids=np.sort(rng.choice(n,min(n,100000),replace=False)+first);chunks=[]
        for start in range(0,len(ids),5000):
            clause=','.join(str(int(i)) for i in ids[start:start+5000])
            chunks.append(_array(db,f"SELECT {','.join(fields)},{case} FROM g WHERE rowid IN ({clause}) ORDER BY rowid",width=len(fields)+1))
        samples[p]=np.concatenate(chunks)

    def scatter(name,x,y,color_by='edge',transmitted=False,phi=False,zero=False):
        fig,axes=panels()
        for ax,p,label in zip(axes[0],POLICIES,LABELS):
            a=samples[p]
            if transmitted: a=a[a[:,index['transmission_attempted']]==1]
            if not len(a): empty(ax)
            if color_by=='outcome':
                for code,category in enumerate(ALLOWED_OUTCOMES):
                    v=a[a[:,index['outcome']]==code]
                    if len(v): ax.scatter(v[:,index[x]],v[:,index[y]],s=3,alpha=.3,color=colors[code],label=category)
            elif color_by=='edge_outcome':
                markers=['.','v','>','s','x','^','P','*','D','o','X']
                for e,color in [(0,'#1f77b4'),(1,'#d62728')]:
                    for code,category in enumerate(ALLOWED_OUTCOMES):
                        v=a[(a[:,index['is_edge']]==e)&(a[:,index['outcome']]==code)]
                        if len(v): ax.scatter(v[:,index[x]],v[:,index[y]],s=5,alpha=.25,color=color,
                            marker=markers[code],label=('E / ' if e else 'N / ')+category)
            else:
                for e in (0,1):
                    v=a[a[:,index['is_edge']]==e]
                    if len(v): ax.scatter(v[:,index[x]],v[:,index[y]],s=3,alpha=.18,label='edge' if e else 'nonedge')
            if phi: ax.axvline(float(config.get('algorithm.phi_theta_deg')),ls='--',color='black',lw=.8)
            if zero: ax.axhline(0,color='black',lw=.8)
            ax.set(title=label+' (sampled)',xlabel=x,ylabel=y)
            handles,labels=ax.get_legend_handles_labels()
            if handles: ax.legend(handles,labels,fontsize=5)
        save(fig,name)
    scatter('05_elevation_vs_csi_scatter.png','max_elevation_deg','gain_db',phi=True)
    scatter('05b_worst_user_elevation_vs_csi.png','worst_user_elevation_deg','gain_db')
    scatter('12_aoi_vs_elevation.png','max_elevation_deg','weighted_post_group_aoi_slots','outcome',phi=True)
    scatter('13_aoi_vs_csi.png','gain_db','weighted_post_group_aoi_slots','edge_outcome')
    scatter('14_power_vs_csi.png','gain_db','power','edge_outcome',transmitted=True)
    scatter('15_rate_margin_vs_csi.png','gain_db','rate_margin_bps_hz',transmitted=True,zero=True)
    scatter('15b_rate_margin_vs_power.png','power','rate_margin_bps_hz',transmitted=True,zero=True)
    del samples

    # Group ids are slot-local; nonexistent groups remain NaN, never zero-filled.
    serving_rows=[r[0] for r in db.execute("SELECT DISTINCT satellite_id FROM c WHERE policy=? ORDER BY satellite_id",(POLICIES[0],))]
    fig,axes=plt.subplots(len(serving_rows),1,figsize=(11,4*len(serving_rows)),squeeze=False)
    for ax,serving in zip(axes[:,0],serving_rows):
        n=int(db.execute('SELECT MAX(slot)+1 FROM c WHERE policy=?',(POLICIES[0],)).fetchone()[0])
        maxa=int(max(config.get('algorithm.action_set')));m=int(config.get('services.messages_per_active_satellite'))
        heat=np.full((m*maxa,n),np.nan)
        for message,gid,slot,aoi in db.execute('SELECT message_id,group_id,slot,weighted_post_group_aoi_slots FROM g WHERE policy=? AND satellite_id=?',(POLICIES[0],serving)):
            heat[int(message)*maxa+int(gid),int(slot)]=aoi
        image=ax.imshow(heat,aspect='auto',origin='lower',extent=(0,n*float(config.get('experiment.slot_dt_s')),0,m*maxa),interpolation='nearest')
        ax.set(title=f'MIRA serving {serving}: message/group positions; membership can change',xlabel='Time (s)',ylabel='message * max_action + group_id')
        fig.colorbar(image,ax=ax,label='Weighted post-group AoI (slots)')
    save(fig,'06_top_aoi_groups_over_time.png')
    fig,axes=panels()
    for ax,p,label in zip(axes[0],POLICIES,LABELS):
        for message in range(int(config.get('services.messages_per_active_satellite'))):
            a=_array(db,'SELECT MIN(time_s),SUM(weighted_post_group_aoi_slots) FROM g WHERE policy=? AND message_id=? GROUP BY slot ORDER BY slot',(p,message))
            ax.plot(a[:,0],a[:,1],label=f'message {message}',lw=.7)
        ax.set(title=label,xlabel='Time (s)',ylabel='Weighted AoI (slots)');ax.legend()
    save(fig,'07_weighted_aoi_by_message_over_time.png')

    expressions=','.join(f"SUM(outcome_category='{c}')" for c in ALLOWED_OUTCOMES)
    fig,axes=panels()
    for ax,p,label in zip(axes[0],POLICIES,LABELS):
        a=_array(db,f'SELECT CAST(time_s AS INT),{expressions},COUNT(DISTINCT slot) FROM g WHERE policy=? GROUP BY CAST(time_s AS INT) ORDER BY 1',(p,),len(ALLOWED_OUTCOMES)+2)
        ax.stackplot(a[:,0],(a[:,1:-1]/a[:,-1,None]).T,labels=ALLOWED_OUTCOMES,colors=colors)
        ax.set(title=label,xlabel='Time (s), 1-second bins',ylabel='Group opportunities per slot')
    axes[0,-1].legend(fontsize=5,loc='upper right');save(fig,'08_group_outcome_counts_over_time.png')

    def category_bars(name,scheduled=False,aoi=False,edge=False):
        fig,ax=plt.subplots(figsize=(11,5));labels=[];totals=[]
        for p in POLICIES:
            for e in ((0,1) if edge else (None,)):
                where='policy=?';args=(p,)
                if e is not None: where+=' AND is_edge=?';args+=(e,)
                if scheduled: where+=' AND scheduled=1'
                metric='SUM(weighted_post_group_aoi_slots)' if aoi else 'COUNT(*)'
                counts=dict(db.execute(f'SELECT outcome_category,{metric} FROM g WHERE {where} GROUP BY outcome_category',args))
                labels.append(p+(' / edge' if e else ' / nonedge') if edge else p)
                totals.append([counts.get(c,0.) for c in ALLOWED_OUTCOMES])
        a=np.asarray(totals,float);den=a.sum(axis=1,keepdims=True)
        if not aoi: a=np.divide(a,den,out=np.full_like(a,np.nan),where=den!=0)
        bottom=np.zeros(len(labels))
        for i,c in enumerate(ALLOWED_OUTCOMES): ax.bar(np.arange(len(labels)),a[:,i],bottom=bottom,label=c,color=colors[i]);bottom+=np.nan_to_num(a[:,i])
        ax.set(xticks=np.arange(len(labels)),xticklabels=labels,ylabel='Weighted AoI sum over trace' if aoi else 'Fraction of eligible opportunities')
        ax.tick_params(axis='x',labelrotation=18);ax.legend(fontsize=6,bbox_to_anchor=(1,1),loc='upper left');save(fig,name)
    category_bars('09_failure_cause_by_policy.png')
    category_bars('09b_failure_cause_among_scheduled_groups.png',scheduled=True)
    category_bars('10_failure_cause_edge_vs_nonedge.png',edge=True)
    category_bars('27_aoi_contribution_by_failure_cause.png',aoi=True)

    def binned(name,column,metrics,table='g'):
        fig,axes=plt.subplots(1,len(metrics),figsize=(5*len(metrics),4),squeeze=False)
        labels=[r[0] for r in db.execute(f'SELECT DISTINCT {column} FROM {table} ORDER BY {column}')]
        if column=='elevation_bin':
            order=['below_10','10-15','15-20','20-25','25-30','30-40','40-50','50-60','60-90','above_90'];labels.sort(key=lambda v:order.index(v))
        for p in POLICIES:
            for ax,(expression,ylabel) in zip(axes[0],metrics):
                values=dict(db.execute(f'SELECT {column},{expression} FROM {table} WHERE policy=? GROUP BY {column}',(p,)))
                ax.plot(range(len(labels)),[values.get(v,math.nan) for v in labels],marker='o',label=p)
                tick_indices=np.unique(np.r_[np.arange(0,len(labels),max(1,math.ceil(len(labels)/12))),len(labels)-1]).astype(int)
                ax.set(xlabel=column,ylabel=ylabel,xticks=tick_indices,xticklabels=[labels[i] for i in tick_indices]);ax.tick_params(axis='x',labelrotation=45);ax.legend(fontsize=6)
        save(fig,name)
    binned('11_outcome_rate_vs_elevation_bin.png','elevation_bin',[
        ('AVG(success)','Success / candidates'),("AVG(outcome_category='POWER_ALLOCATION_INFEASIBLE_NO_TX')",'Power no-TX / candidates'),('AVG(weighted_post_group_aoi_slots)','Mean weighted group AoI')])
    binned('12b_aoi_by_elevation_bin.png','elevation_bin',[('AVG(weighted_post_group_aoi_slots)','Mean weighted group AoI'),('q50(weighted_post_group_aoi_slots)','Median weighted group AoI')])
    binned('13b_aoi_by_csi_quantile.png','gain_db_bin',[('AVG(weighted_post_group_aoi_slots)','Mean weighted group AoI'),('q50(weighted_post_group_aoi_slots)','Median weighted group AoI'),('q95(weighted_post_group_aoi_slots)','p95 weighted group AoI')])
    binned('18_power_infeasibility_vs_class_switches.png','csi_order_class_switch_count',[('AVG(power_infeasible)','Power infeasible / serving-slots')],table='c')
    binned('29_policy_aoi_conditioned_on_elevation.png','elevation_bin',[('AVG(weighted_post_group_aoi_slots)','Mean weighted group AoI')])
    binned('30_policy_success_conditioned_on_csi.png','gain_db_bin',[('AVG(success)','Success / candidate groups')])
    fig,axes=panels()
    for ax,p,label in zip(axes[0],POLICIES,LABELS):
        rows=db.execute('SELECT scheduled_nonedge_groups,scheduled_edge_groups,COUNT(*),AVG(power_infeasible) FROM c WHERE policy=? GROUP BY scheduled_nonedge_groups,scheduled_edge_groups',(p,)).fetchall()
        a=np.full((int(max(r[1] for r in rows))+1,int(max(r[0] for r in rows))+1),np.nan)
        for nonedge,edge,n,rate in rows:
            a[int(edge),int(nonedge)]=rate
            ax.text(nonedge,edge,str(n),ha='center',va='center',fontsize=5)
        im=ax.imshow(a,origin='lower',aspect='auto',vmin=0,vmax=1)
        ax.set(title=label+' (cell text = sample count)',xlabel='Scheduled nonedge',ylabel='Scheduled edge');fig.colorbar(im,ax=ax,label='Infeasible rate')
    save(fig,'16_power_infeasibility_heatmap_edge_nonedge_counts.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,x in zip(axes,['serving_elevation_mean_deg','edge_fraction']):
        scale=5. if 'elevation' in x else .1
        for p in POLICIES:
            a=_array(db,f'SELECT CAST({x}/? AS INT)*?,AVG(power_infeasible) FROM c WHERE policy=? GROUP BY CAST({x}/? AS INT) ORDER BY 1',(scale,scale,p,scale))
            ax.plot(a[:,0],a[:,1],marker='o',label=p)
        ax.set(xlabel=x+' (binned)',ylabel='Power infeasible / serving-slots');ax.legend(fontsize=7)
    save(fig,'17_power_infeasibility_vs_elevation.png')

    # Exact means, medians and p95 for action comparisons.
    with (out/'action_diagnostic_summary.csv').open(encoding='utf-8',newline='') as handle: action_rows=list(csv.DictReader(handle))
    fig,axes=panels()
    for ax,p,label in zip(axes[0],POLICIES,LABELS):
        rows=[r for r in action_rows if r['policy']==p];x=[int(r['action']) for r in rows]
        for metric in ['mean','median','p95']: ax.plot(x,[float(r[metric+'_aggregated_weighted_sum_aoi']) for r in rows],marker='o',label=metric)
        ax.set(title=label,xlabel='Held action',ylabel='Aggregated AoI at observed action slots');ax.legend()
    save(fig,'19_action_conditioned_aoi.png')
    binned('20_action_conditioned_failure_rates.png','action',[("AVG(outcome_category='POWER_ALLOCATION_INFEASIBLE_NO_TX')",'Power no-TX / candidates'),("1.0*SUM(outcome_category='TRANSMITTED_RATE_BELOW_THRESHOLD')/NULLIF(SUM(transmission_attempted),0.0)",'Own-rate failure / transmitted'),("1.0*SUM(outcome_category='SIC_PROPAGATED_FAILURE')/NULLIF(SUM(transmission_attempted),0.0)",'SIC propagation / transmitted'),('AVG(success)','Success / candidates')])
    fig,axes=plt.subplots(2,1,figsize=(11,7))
    a=_array(db,'SELECT MIN(time_s),MAX(aggregated_weighted_sum_aoi_slots),MAX(power_infeasible) FROM c WHERE policy=? GROUP BY slot ORDER BY slot',(POLICIES[0],),3)
    axes[0].plot(a[:,0],a[:,1],lw=.7);fail=a[:,2]>0;axes[0].scatter(a[fail,0],a[fail,1],s=6,color='red',label='power infeasible');axes[0].set(ylabel='Aggregated weighted AoI');axes[0].legend()
    for serving in serving_rows:
        a=_array(db,'SELECT time_s,action FROM c WHERE policy=? AND satellite_id=? AND is_regrouping_slot=1 ORDER BY slot',(POLICIES[0],serving))
        axes[1].step(a[:,0],a[:,1],where='post',label=f'serving {serving}')
    axes[1].set(xlabel='Time (s)',ylabel='Held action');axes[1].legend();save(fig,'21_mira_action_and_aoi_timeline.png')
    fig,axes=plt.subplots(5,len(serving_rows),figsize=(7*len(serving_rows),14),squeeze=False)
    for j,serving in enumerate(serving_rows):
        a=_array(db,'SELECT time_s,serving_elevation_min_deg,serving_elevation_mean_deg,action,weighted_aoi_slots,candidate_edge_groups,candidate_nonedge_groups FROM c WHERE policy=? AND satellite_id=? ORDER BY slot',(POLICIES[0],serving),7)
        axes[0,j].plot(a[:,0],a[:,1],label='min elevation');axes[0,j].plot(a[:,0],a[:,2],label='mean elevation');axes[0,j].legend();axes[0,j].set_title(f'MIRA serving {serving}')
        axes[1,j].step(a[:,0],a[:,3],where='post');axes[1,j].set_ylabel('Held action')
        axes[2,j].plot(a[:,0],a[:,4]);axes[2,j].set_ylabel('System weighted AoI')
        axes[3,j].plot(a[:,0],a[:,5],label='edge');axes[3,j].plot(a[:,0],a[:,6],label='nonedge');axes[3,j].legend()
        a=_array(db,f'SELECT MIN(time_s),{expressions} FROM g WHERE policy=? AND satellite_id=? GROUP BY slot ORDER BY slot',(POLICIES[0],serving),12)
        for code in [3,4,5,9]: axes[4,j].plot(a[:,0],a[:,code+1],label=ALLOWED_OUTCOMES[code],lw=.8)
        axes[4,j].set_xlabel('Time (s)');axes[4,j].legend(fontsize=6)
    save(fig,'22_elevation_action_aoi_timeline.png')
    binned('23_csi_drift_within_regrouping_epoch.png','milliseconds_since_regrouping',[
        ('AVG(ABS(gain_change_from_boundary_db))','Mean abs CSI drift (dB)'),('AVG(csi_rank_changed_from_boundary)','Fraction with rank changed')])
    binned('24_aoi_vs_csi_rank_drift.png','group_order_inversion_count',[
        ('AVG(weighted_aoi_slots)','Mean serving weighted AoI'),('AVG(power_infeasible)','Power infeasible / serving-slots')],table='c')
    fig,axes=plt.subplots(1,5,figsize=(19,4),squeeze=False)
    worst=db.execute('SELECT slot,MAX(aggregated_weighted_sum_aoi_slots) a FROM c WHERE policy=? GROUP BY slot ORDER BY a DESC,slot LIMIT 5',(POLICIES[0],)).fetchall()
    for ax in axes[0][len(worst):]: empty(ax)
    for ax,(slot,total) in zip(axes[0],worst):
        rows=db.execute('SELECT satellite_id,message_id,group_id,weighted_post_group_aoi_slots,outcome_category,is_edge,max_elevation_deg,gain_db,power FROM g WHERE policy=? AND slot=? ORDER BY weighted_post_group_aoi_slots DESC LIMIT 5',(POLICIES[0],slot)).fetchall()
        ax.bar(range(len(rows)),[r[3] for r in rows],color=[colors[ALLOWED_OUTCOMES.index(r[4])] for r in rows])
        labels=[f'C{r[0]} M{r[1]} G{r[2]}\n{"E" if r[5] else "N"} {r[6]:.1f}deg\n{r[7]:.1f}dB p={r[8]:.3g}' for r in rows]
        ax.set(xticks=range(len(rows)),xticklabels=labels,title=f'slot {slot}, AoI {total:.2f}',ylabel='Weighted group AoI');ax.tick_params(axis='x',labelrotation=65,labelsize=6)
    save(fig,'25_worst_aoi_episode_panels.png')
    fig,ax=plt.subplots(figsize=(11,4))
    rows=db.execute("SELECT sic_first_failed_predecessor_message_id,sic_first_failed_predecessor_group_id,COUNT(*) n FROM g WHERE outcome_category='SIC_PROPAGATED_FAILURE' GROUP BY sic_first_failed_predecessor_message_id,sic_first_failed_predecessor_group_id ORDER BY n DESC LIMIT 20").fetchall()
    if rows: ax.bar(range(len(rows)),[r[2] for r in rows]);ax.set(xticks=range(len(rows)),xticklabels=[f'M{r[0]} G{r[1]}' for r in rows])
    else: empty(ax)
    ax.set(ylabel='Propagated failures',xlabel='First own-rate-failed predecessor (slot-local ids, pooled policies)');save(fig,'26_sic_failure_source_counts.png')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,metric in zip(axes,['SUM(weighted_post_group_aoi_slots)','AVG(weighted_post_group_aoi_slots)','p95']):
        for i,p in enumerate(POLICIES):
            values=[]
            for e in (0,1):
                if metric=='p95':
                    v=np.fromiter((r[0] for r in db.execute('SELECT weighted_post_group_aoi_slots FROM g WHERE policy=? AND is_edge=?',(p,e))),float);value=float(np.quantile(v,.95)) if len(v) else math.nan
                else: value=db.execute(f'SELECT {metric} FROM g WHERE policy=? AND is_edge=?',(p,e)).fetchone()[0]
                values.append(math.nan if value is None else value)
            ax.bar(np.arange(2)+i*.22,values,width=.22,label=p)
        ax.set(xticks=[.22,1.22],xticklabels=['nonedge','edge'],ylabel=metric);ax.legend(fontsize=6)
    save(fig,'28_aoi_contribution_edge_vs_nonedge.png')
    return names
