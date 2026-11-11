"""Traceable geographic snapshots and per-slot occupancy for the phi=15 main run."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from .geometry import radial_project_to_sphere, ecef_to_spherical, aeqd_forward
from .storage import atomic_write_json, write_csv


def group_spatial_statistics(east_km, north_km):
    points = np.column_stack((east_km, north_km)).astype(float)
    if not len(points) or not np.all(np.isfinite(points)):
        raise ValueError("spatial groups require finite member coordinates")
    center = points.mean(axis=0)
    radii = np.linalg.norm(points-center, axis=1)
    if len(points) > 1:
        pairwise = np.linalg.norm(points[:, None]-points[None, :], axis=2)[np.triu_indices(len(points), 1)]
        pair = dict(pairwise_mean_km=float(pairwise.mean()), pairwise_median_km=float(np.median(pairwise)),
                    pairwise_p90_km=float(np.quantile(pairwise,.9)), pairwise_max_km=float(pairwise.max()))
    else:
        pair = dict.fromkeys(("pairwise_mean_km","pairwise_median_km","pairwise_p90_km","pairwise_max_km"))
    area = None
    if len(points) >= 3 and np.linalg.matrix_rank(points-center) == 2:
        from scipy.spatial import ConvexHull, QhullError
        try:
            area = float(ConvexHull(points).volume)
        except QhullError:
            pass
    return dict(geographic_centroid_east_km=float(center[0]), geographic_centroid_north_km=float(center[1]),
                centroid_p90_radius_km=float(np.quantile(radii,.9)), centroid_max_radius_km=float(radii.max()),
                convex_hull_area_km2=area, **pair)


def snapshot_group_rows(config, scenario, groups, *, slot, elevation, amplitudes, satellite_ecef):
    sat = groups[0].satellite_index
    satellite_id = scenario.constellation.elements[sat].satellite_id
    radius = float(config.get("geometry.earth_radius_km"))
    lat0, lon0 = ecef_to_spherical(radial_project_to_sphere(config.get("geometry.center_raw_ecef_km"), radius))
    sub_lat, sub_lon = ecef_to_spherical(satellite_ecef[sat])
    sub_east, sub_north = aeqd_forward(sub_lat, sub_lon, float(lat0), float(lon0), radius)
    rows, members = [], []
    for group in groups:
        indices = group.member_indices
        ids = [scenario.devices.ids[int(i)] for i in indices]
        identity = hashlib.sha256(np.sort(indices).astype("<i4").tobytes()).hexdigest()
        common = dict(slot=slot, epoch=slot//int(config.get("derived.regrouping_interval_slots")),
                      time_s=slot*float(config.get("experiment.slot_dt_s")), serving_satellite_id=satellite_id,
                      message_id=group.message_id, group_id=group.group_id,
                      group_key=f"{group.message_id}:{group.group_id}", membership_hash=identity,
                      geo_edge=int(group.is_edge), scheduled=int(group.scheduled))
        rows.append(dict(**common, group_size=len(indices), max_elevation_deg=group.max_elevation_deg,
                         min_elevation_deg=float(np.min(elevation[sat,indices])), min_gain=group.gain,
                         member_ids=json.dumps(ids), power_pool=group.metadata.get("power_pool","unscheduled"),
                         promoted=int(group.metadata.get("power_pool_promoted",False)),
                         sub_satellite_east_km=float(sub_east), sub_satellite_north_km=float(sub_north),
                         sub_satellite_latitude_deg=math.degrees(float(sub_lat)),
                         sub_satellite_longitude_deg=math.degrees(float(sub_lon)),
                         **group_spatial_statistics(scenario.devices.east_km[indices],scenario.devices.north_km[indices])))
        for index in indices:
            members.append(dict(**common, ue_id=scenario.devices.ids[int(index)],
                                east_km=float(scenario.devices.east_km[index]), north_km=float(scenario.devices.north_km[index]),
                                latitude_deg=math.degrees(float(scenario.devices.latitude_rad[index])),
                                longitude_deg=math.degrees(float(scenario.devices.longitude_rad[index])),
                                elevation_deg=float(elevation[sat,index]), gain=float(amplitudes[sat,index]**2)))
    return rows, members


def geo_occupancy_row(groups, *, slot, dt_s, satellite_id, handover):
    edge = [g for g in groups if g.is_edge]
    return dict(slot=slot,time_s=slot*dt_s,serving_satellite_id=satellite_id,handover=int(handover),
                geo_edge_candidate_count=len(edge),geo_edge_scheduled_count=sum(g.scheduled for g in edge),
                geo_edge_ue_count=sum(len(g.member_indices) for g in edge),
                promoted_count=sum(g.metadata.get("power_pool_promoted",False) for g in groups))


def _read(path):
    with Path(path).open(encoding="utf-8",newline="") as handle:
        return list(csv.DictReader(handle))


def analyze_geo_edge(config, source_directory, output_directory, figures_directory, *, expected_slots):
    """Analyze saved real snapshots; never start or repair a missing simulation."""
    source, output, figures = Path(source_directory),Path(output_directory),Path(figures_directory)
    paths = {name:source/name for name in ("edge_group_snapshots.csv","edge_member_positions.csv","geo_edge_time_distribution.csv")}
    if any(not p.exists() for p in paths.values()):
        raise ValueError("missing phi15 geographic traces; rerun with save_geo_edge_trace=True")
    output.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    import shutil
    for name,path in paths.items():
        if path.resolve() != (output/name).resolve(): shutil.copyfile(path,output/name)
    snapshots=_read(paths["edge_group_snapshots.csv"])
    occupancy=_read(paths["geo_edge_time_distribution.csv"])
    if [int(r["slot"]) for r in occupancy] != list(range(expected_slots)):
        raise ValueError("geo occupancy must contain every physical slot exactly once")
    epoch_slots=sorted({int(r["slot"]) for r in snapshots})
    interval=int(config.get("derived.regrouping_interval_slots"))
    if epoch_slots != list(range(0,expected_slots,interval)):
        raise ValueError("geo snapshot cadence mismatch")
    spatial=[]
    for flag in (0,1):
        subset=[r for r in snapshots if int(r["geo_edge"])==flag]
        row=dict(geo_edge=flag,snapshot_group_observations=len(subset),sampling="every regrouping boundary; repeated UE/time observations")
        for key in ("pairwise_mean_km","pairwise_median_km","pairwise_p90_km","pairwise_max_km","centroid_p90_radius_km","centroid_max_radius_km","convex_hull_area_km2"):
            values=np.array([float(r[key]) for r in subset if r[key] != ""])
            row[key+"_sample_count"]=len(values)
            row[key+"_median"]=float(np.median(values)) if len(values) else None
            row[key+"_p90"]=float(np.quantile(values,.9)) if len(values) else None
        spatial.append(row)
    # Compare edge-group centroids ONLY within the same time and satellite.
    pairs=[]
    for slot in epoch_slots:
        edge=[r for r in snapshots if int(r["slot"])==slot and int(r["geo_edge"])==1]
        for i,left in enumerate(edge):
            for right in edge[i+1:]:
                if left["serving_satellite_id"] != right["serving_satellite_id"]:raise ValueError("multiple serving satellites in snapshot")
                distance=math.hypot(float(left["geographic_centroid_east_km"])-float(right["geographic_centroid_east_km"]),float(left["geographic_centroid_north_km"])-float(right["geographic_centroid_north_km"]))
                pairs.append(dict(slot=slot,time_s=left["time_s"],serving_satellite_id=left["serving_satellite_id"],left_group=left["group_key"],right_group=right["group_key"],centroid_distance_km=distance))
    write_csv(output/"geo_edge_spatial_summary.csv",spatial)
    write_csv(output/"geo_edge_centroid_distances.csv",pairs,fieldnames=["slot","time_s","serving_satellite_id","left_group","right_group","centroid_distance_km"])
    # Bound member-trace memory to the three requested map snapshots.
    selected=sorted(set(epoch_slots[i] for i in (0,len(epoch_slots)//2,len(epoch_slots)-1)))
    member_rows=[]
    with paths["edge_member_positions.csv"].open(encoding="utf-8",newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["slot"]) in selected:member_rows.append(row)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(selected),figsize=(7*len(selected),6),squeeze=False)
    for ax,slot in zip(axes[0],selected):
        current=[r for r in snapshots if int(r["slot"])==slot]
        edge=[r for r in current if int(r["geo_edge"])==1]
        all_members={r["ue_id"]:r for r in member_rows if int(r["slot"])==slot}
        if len(all_members)!=120:
            raise ValueError("geographic snapshot must contain all 120 UE positions")
        ax.scatter([float(r["east_km"]) for r in all_members.values()],
                   [float(r["north_km"]) for r in all_members.values()],c=".8",s=12,label="All 120 stationary UE")
        for index,row in enumerate(edge):
            members=[r for r in member_rows if int(r["slot"])==slot and r["group_key"]==row["group_key"]]
            color=plt.get_cmap("tab20")(index%20)
            ax.scatter([float(r["east_km"]) for r in members],[float(r["north_km"]) for r in members],color=color,s=22)
            ax.scatter(float(row["geographic_centroid_east_km"]),float(row["geographic_centroid_north_km"]),color=color,marker="x",s=70)
        ref=current[0]
        ax.scatter(float(ref["sub_satellite_east_km"]),float(ref["sub_satellite_north_km"]),marker="*",color="black",s=110,label="Serving sub-satellite point")
        ax.set(title=f"t={float(ref['time_s']):g}s, {ref['serving_satellite_id']}\n{len(edge)} geographic edge groups",xlabel="East (km)",ylabel="North (km)",aspect="equal")
        if not edge:ax.text(.5,.9,"No geographic edge groups",ha="center",transform=ax.transAxes)
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(figures/"geo_edge_phi15_map.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,key in zip(axes,("pairwise_mean_km","centroid_max_radius_km","pairwise_max_km")):
        for flag,label in ((0,"Non-edge"),(1,"Geo-edge")):
            values=np.sort([float(r[key]) for r in snapshots if int(r["geo_edge"])==flag and r[key]!=""])
            if len(values):ax.step(values,np.arange(1,len(values)+1)/len(values),where="post",label=f"{label}, n={len(values)}")
        ax.set(xlabel=key+" (km)",ylabel="ECDF",title="All 1-second boundary snapshots");ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(figures/"geo_edge_phi15_spread_distribution.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    times=[float(r["time_s"]) for r in occupancy]
    axes[0].plot(times,[int(r["geo_edge_candidate_count"]) for r in occupancy],lw=.7)
    axes[0].set_ylabel("Geo-edge candidate groups")
    axes[1].step(times,[int(r["serving_satellite_id"].split('_')[-1]) for r in occupancy],where="post")
    axes[1].set(xlabel="Time (s)",ylabel="Serving satellite ID")
    fig.tight_layout();fig.savefig(figures/"geo_edge_phi15_time_occupancy.png",dpi=150);plt.close(fig)
    values=[r["centroid_distance_km"] for r in pairs]
    summary=dict(snapshot_slots=len(epoch_slots),edge_group_snapshot_observations=spatial[1]["snapshot_group_observations"],
                 observed_start_time_s=float(occupancy[0]["time_s"]),
                 observed_end_time_s_exclusive=expected_slots*float(config.get("experiment.slot_dt_s")),
                 serving_satellite_ids=sorted({r["serving_satellite_id"] for r in occupancy}),
                 representative_map_slots=selected,
                 same_snapshot_edge_centroid_pair_count=len(values),
                 same_snapshot_edge_centroid_distance_median_km=float(np.median(values)) if values else None,
                 same_snapshot_edge_centroid_distance_p90_km=float(np.quantile(values,.9)) if values else None,
                 slots_with_geo_edge=sum(int(r["geo_edge_candidate_count"])>0 for r in occupancy),
                 sampling="per-slot occupancy; all candidate groups at every 1s boundary; early/middle/late maps",
                 spatial_metrics=spatial)
    atomic_write_json(output/"summary.json",summary)
    report=["# phi=15° 地理 Edge Groups 分析","",f"共 {len(epoch_slots)} 個 boundary snapshots；地理 edge group-time observations={spatial[1]['snapshot_group_observations']}。",
            f"全 slot 有 edge 的時槽數={summary['slots_with_geo_edge']} / {expected_slots}。",
            f"適用時段=[{summary['observed_start_time_s']}, {summary['observed_end_time_s_exclusive']}) 秒；服務衛星={summary['serving_satellite_ids']}；地圖 snapshot slots={selected}。",
            "群內是否靠近：依下列群內距離與半徑統計衡量；未預設 cluster 距離門檻。",
            json.dumps(spatial,ensure_ascii=False,indent=2),
            "所有 edge groups 是否集中在同一區塊：以同一時刻、同一 serving satellite 的 centroid 間距回答；不同時間的點不合併成一個地理群組。",
            f"同 snapshot centroid pairs={len(values)}，median={summary['same_snapshot_edge_centroid_distance_median_km']} km，p90={summary['same_snapshot_edge_centroid_distance_p90_km']} km。",
            "沒有 edge 或無多組 pair 時，此項不可判定；不以空資料推論集中。單一固定 layout seed 與有限時段限制推論範圍。",
            "低仰角的定義不蘊含地面距離接近；環帶、多方向分布或單一 cluster 必須依這些圖和統計檢視。",
            "同一 UE 可於不同 epoch 重複出現；統計的權重為 group-time observations。Singleton pairwise distance 為 NA。",
            "地理 Edge 不保證 instantaneous CSI 較差；promoted weak non-edge 是獨立的功率標籤。"]
    (output/"report_geo_edge_phi15.md").write_text("\n\n".join(report)+"\n",encoding="utf-8")
    return summary
