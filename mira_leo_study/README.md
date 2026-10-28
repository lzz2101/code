# MIRA-LEO: phi=20、1 秒分組、功率無解計為更新失敗

目前正式設定為 phi_theta_deg=20、B=0.05、compact actions=[1,2,4,5,8,10]、Fixed a=4、每秒 regrouping、10 ms physical slot、10 秒 / 10 epoch SW-UCB window。Algorithm 2 維持使用者指定論文版本的嚴格排程規則；Algorithm 3 維持 TWC/AIM staircase 與 global-SIC power-only projection。

## 新增的失敗處理

當 LP 明確回報 infeasible（solver status=2），且沒有 floor/configuration 錯誤時：

- 保留 Algorithm-2 scheduled flags，沒有刪除群組。
- 該衛星本 slot 實際功率為 0，所有排程更新失敗；其他衛星照常運作。
- 不把不可行的 target power 用於干擾、SINR 或 decoding。
- AoI 沿用失敗遞迴，包含適用的 handover penalty。
- 該 slot 的 reward 照常計入，pilot 也保留失敗樣本，三方法採相同規則。
- 記錄 power_infeasible_cohort_slots / power_outage_slots，不混成一般 channel decoding failure。

這是使用者新增的 power-outage 模型假設。純功率 helper 的 valid 仍為 false，表示 proposed plan 真正不可行；actual_total_power=0。Scheduled groups 與真正 transmitted groups 分開輸出。數值求解錯誤（例如 status=4）、floor/configuration 錯誤、scheduled set 被修改等仍停止，不會被掩蓋成傳輸失敗。

有 outage 的完整 run 顯示 COMPLETED_WITH_POWER_OUTAGES；clipping 問題仍保留 REWARD_RANGE_MISMATCH。invalid_power_or_sic_count 現在用來標記未處理的約束錯誤，已處理的 power outages 由專用欄位計數。

## 兩個時間尺度與演算法

| 模組 | 更新 |
|---|---|
| Action selection / subscriber CSI sorting / membership rebuild | 每 100 slots = 1 秒 |
| Geometry / current CSI / group gain / elevation edge label / scheduling / power / SIC / AoI | 每 10 ms |
| SW-UCB observation | 每 epoch 一筆，平均該 epoch 的 per-slot normalized rewards |

Membership 僅 cache immutable tuples；gain / is_edge / power / decoding 不 cache。Handover 不觸發額外 regrouping，該 epoch reward 仍給原 action owner。Global epoch index 使 inactive agent 的舊 observations 正常過期。

Strict Algorithm 2：U=62。若 edge count > U，取 priority 最高的 U 個 edge；否則排全部 edge（包含非正 priority），再用剩餘容量排 positive-priority nonedge。Scheduler 不做 power admission。Compact 最多每 satellite 30 groups。

Power mixed pools=0.7/0.3；single-class 全部 1.0。Current CSI weak-to-strong staircase floor=V_phi*(N-r+1)，continuous residual 平均加回。Global SIC repair 保留 class sums、floors 與 schedule。沒有新增 water filling 或修正 reference TWC 的 off-by-one helper。

其他物理設定保持原值：480 s、carrier 20 GHz、bandwidth 50 MHz、SCS 60 kHz、phi_min=10 deg、P_sat=1、V_phi=.0005、R_theta=1e-5、weights=[.7,.2,.1]、xi=1/sqrt(2)、HO=25 ms、ISL hop=5 ms。

## 安裝與不執行模擬的驗證

在 mira_leo_study 上一層執行，使用 Python 3.11 以上：

```powershell
Set-Location 'C:\Users\lzz\OneDrive\ドキュメント\我的論文code'
.\.venv\Scripts\python.exe -m pip install -r .\mira_leo_study\requirements.txt
.\.venv\Scripts\python.exe -m compileall -q .\mira_leo_study
.\.venv\Scripts\python.exe -m mira_leo_study validate-config
.\.venv\Scripts\python.exe -m unittest discover -s .\mira_leo_study\tests -p 'test_*.py' -v
```

最新非模擬測試：87 tests，82 passed，5 skipped。實際 simulation integration tests 仍需明確設定 MIRA_LEO_RUN_SIMULATION_TESTS=1；本次沒有啟用。人工 counterfactual component tests 使用自行建立的 arrays，不跑真實軌道/通道/AoI study。SciPy、pyarrow、matplotlib 等完整 requirements 都應安裝。

## 在自己的 PC 跑新版

新預設輸出目錄為 outputs/fixed_phi20_B005_rg1s_compact_outage，避免與 20 度或 fail-stop 模型結果混用。

```powershell
.\.venv\Scripts\python.exe -m mira_leo_study paper-results --slots 1000 --seed 0 --outputs .\outputs\fixed_phi20_B005_rg1s_compact_outage
```

沒有新 bounds 時會先自動 pilot。30 度或舊 fail-stop policy 的 bounds context 不相容，需重算；這次沒有執行 reward pilot。也可先單獨執行：

```powershell
.\.venv\Scripts\python.exe -m mira_leo_study calibrate-reward-bounds --outputs .\outputs\fixed_phi20_B005_rg1s_compact_outage --seed 0
```

完整三方法 48,000 slots：

```powershell
.\.venv\Scripts\python.exe -m mira_leo_study paper-results --seed 0 --outputs .\outputs\fixed_phi20_B005_rg1s_compact_outage --resume
```

`--slots` 只縮短 evaluation，不縮短 pilot。短版存於 debug/slots_1000_seed_0/，正式版在 root。Checkpoint schema=4，拒絕 schema 2 / 3。Epoch 中途暫停保存 held partition、owner、reward sum/count，不提前 finalize partial reward。`--save-group-trace` 可保存較大的逐群組資料。`--tau-s 5 10 20` 是 observation-window sensitivity，regrouping 仍為 1 秒。

## 看結果

Root：config.json、parameter_snapshot.json、summary.json、report.md。

- formal_results/convergence_long_run_aoi：raw / running AoI CSV、comparison_summary.csv/json、convergence_long_run_aoi.png/pdf、raw trace 與 scheduler diagnostic 圖。
- formal_results/regrouping_overhead：事件 CSV、normalization summary 與 Markdown table。
- runs/<policy>：slot_metrics.csv/parquet、convergence_slot_metrics.csv、bandit_observations.csv、選用 group_trace。

新 slot 欄位包括 power_plan_valid、transmission_failure_power、transmission_attempted、transmitted_groups、power_solver_status、planned_total_power、實際 total_transmitted_power。Solver status=-1 代表未使用 LP，0 代表成功，2 代表 infeasible。Cohort / slot 計數及真正發射的 SIC violation 分開保存；power outages 不列入一般 rate / SIC decoding-failure counters。

AoI 是全 constellation weighted-sum，單位 slots，running mean 保留整段含 exploration 與失敗 slots，不用 reward 代替 AoI。Comparison summary 另列每方法 power_infeasible_cohort_slots（失敗 cohort-slot 次數）、power_outage_slots（至少一個 cohort outage 的 physical slots 數）。不預設三方法排名。

Overhead 排除 initial assignment、coverage exit、handover；epoch 內 membership 不變。Action occupancy 每 slot 計數，decision count 只在 boundary 計數。AIM-static、GA-grouping、PPO-multicast 仍 NOT_IMPLEMENTED。歷史 phi/B API 保留但不屬於正式 CLI。

## 歷史功率診斷（20° / 30°）

目前正式設定已改回 20°，保留 zero-power outage；下列是先前完成的診斷，本次沒有重跑。


本次只額外執行固定 pilot snapshot grid 的 power-only scan：90 個時刻 x 2 cohorts x 6 actions=1080 組，假設所有 candidate scheduled。沒有計算 AoI、learning 或正式 policy failure frequency。

- 20 deg 的上一輪診斷：8 組 LP-infeasible。
- 30 deg 本輪診斷：170 組 LP-infeasible，910 組 feasible，其他 invalid=0。
- 170 組全部驗證會轉成零功率 transmission failure。
- 分 action：a=2:3、a=4:19、a=5:20、a=8:52、a=10:76；a=1:0。

因此 30 度仍有同類衝突，且此取樣範圍內增加。這不是實際完整 simulation 的失敗率；Algorithm 2 可能略過非正收益 nonedge。資料位於 mira_leo_study/diagnostics/pilot_power_phi30_seed0_all_candidates/。

可重現相同診斷：

```powershell
.\.venv\Scripts\python.exe -m mira_leo_study.power_diagnostics
```

目前修改見 PHI20_POWER_OUTAGE_MODIFICATION_REPORT.md；PHI30_POWER_OUTAGE_MODIFICATION_REPORT.md 為上一版歷史紀錄。前一版 1 秒 compact 報告保留為歷史紀錄。
