# 120-UE single-serving 修改報告

日期：2026-11-03（Asia/Taipei）。

## 完成範圍

依使用者指定的 MD 修改程式、測試與文件；使用者明確要求無須實際模擬，因此此次沒有啟動 smoke、reward pilot、paper-results 或模擬 integration tests。驗收範圍是語法、匯入、呼叫介面、設定與人工 component tests；沒有宣稱完成正式數值驗證或得到新研究結果。

指定專案：`C:\Users\lzz\OneDrive\ドキュメント\我的論文code\mira_leo_study`。

編輯及驗證副本：`C:\Users\lzz\Documents\我的論文code\mira_leo_study`。開始時以指定 OneDrive 專案為基準同步 source/config/tests/docs；原始檔備份與 SHA-256 manifest 保存在 `C:\Users\lzz\Documents\我的論文code\backups\120ue_single_serving_20261103`。僅回寫本報告列出的 source/config/tests/docs，不回寫或覆蓋舊 outputs、diagnostics 資料或整個專案目錄。

## 新場景與保留設定

- 保留 8 顆 LEO、800 km、高度/5+3 軌道面/原地理位置/epoch；120 stationary UE 全數 active。
- 固定 3 個互斥 messages，各 40 人；subscription seed 控制指派，位置與訂閱 deterministic，更新 schema tag 的 subscription hash 與新的 device count 不會混同舊模型。
- 全部 120 人共用單一 serving satellite；association 回傳 scalar satellite ID、handover、initial acquisition 與 outage count。Coverage-first / minimum elevation / hysteresis / TTT 保留，fallback 不假定全員成功。
- 無任何鄰星 UE、messages、group、scheduler、AoI 或 NOMA/SIC 模擬。Per-satellite bandit agent 保留，只在該星成為 action owner 時選擇或更新。
- Membership/action 每 100 slots 更新；CSI、geometry、edge、scheduling、power、SINR、SIC 與 AoI 每 slot 更新。Handover 不提早重建 partition，epoch reward 仍給原 owner。
- 比對原 base.yaml，僅改 experiment name、UE count、association mode/objective、external-interference mode，移除舊 population-count key 並新增 interferer_power_mode。其餘數值保持原值：10 ms、480 s、phi_min=10°、phi_theta=20°、compact=[1,2,4,5,8,10]、V_c/V_nc=0.7/0.3、V_phi=0.0005、R_theta=1e-5、B=0.05、weights=[0.7,0.2,0.1]、20 GHz/60 kHz/50 MHz、seed=0/0、HO=25 ms、原 propagation/ISL/SIC/AoI 定義。

## 確切干擾模型

對當前 serving LEO l、UE u、非服務衛星 s：

```text
chi[s,u,t] = 1[elevation[s,u,t] >= 10 degrees]
E[|h_SR|^2] = 2*b0 + Omega
zeta[s,u,t] = (c / (4*pi*d[s,u,t]*fc))^2
psi[l,u,t] = sum(s != l) chi[s,u,t] * zeta[s,u,t]
                          * (2*b0 + Omega) * P_interferer
P_interferer = channel.normalized_satellite_power = 1.0
psi[group,t] = max(u in group) psi[l,u,t]
```

先逐 UE 加總可見鄰星功率，再取群組最大值；不是 sum(max)，也不使用 edge-group count 換算干擾。不重新抽鄰星 instantaneous SR fading，不額外乘 bandwidth、beam gain、方向圖或隔離係數。使用原 path_gain 與正規化單位。Group worst-interference UE 可以不同於 weakest-desired-channel UE，保留原保守近似。

Serving actual TX power 只加總本研究 scheduled group powers。LP 證明無解時仍保留 schedule、actual TX=0、更新失敗；外部功率向量始終為七個 1 與當前 serving index 的 0。Handover 即時移動排除位置；舊 serving 若覆蓋 UE 便貢獻干擾。外部 interference 在 decoding 前寫入 group；intra-NOMA/SIC 不改。

Group trace 的 `external_interference_power` 對所有 candidate groups 記錄該幾何下的 expected interference，包括未排程/功率 outage；`interferer_count_at_worst_ue` 是在最大干擾 UE 上可見且滿載的鄰星數，tie 取 member order 第一個 UE，沒有假造量測值。Summary 原 `mean_interference_power` 仍以實際有發射的 group opportunities 統計。

## AoI、輸出與相容性

- 一個 UE 每 slot 只更新一次；weighted sum 是全 120 人的加權和。主要報表不除以 8；per-satellite contribution 只表示當時 serving 的這批 120 人。
- Subscription schema 為 message_by_device、active_device_ids、message_members[int]、subscription_hash；Group.key=(message_id,group_id)，另保留 satellite_index。正式 API、config、CSV/Parquet 不再以 cohort 分割；程式只保留舊 key 的明確拒絕訊息及 regression assertions。
- Slot trace 每 policy/slot 僅一列；diagnostic SQL 強制此 identity。Group trace 以 policy/slot/message/group 唯一，另記當前 serving satellite ID。
- `power_infeasible_serving_slots`、`serving_edge_groups`、`aoi_rank_in_system` 等取代舊欄位。新增 serving_actual_tx_power、external power vector、external_interference_power、worst interference UE 及 interferer count。
- Config snapshot、summary、reward-pilot metadata 明確記錄 total_ue=120、message_count=3、subscribers_per_message=40、serving_satellite_count=1、interference_mode=fixed_full_load、interferer_power_normalized=1.0。Environment hash 也納入 external-interference 假設。
- Checkpoint schema 從指定專案原本的 5 升為 **6**，diagnostic schema=**2**，拒絕舊 schema。Reward protocol 改為 `120ue_single_serving_full_load_regroup1s_v6`，檔案及直接 API 都驗證 context；舊 300-UE bounds 不能沿用。Fixed-a=4 pilot 不依賴 bandit window，因此 tau sensitivity 共用新 bounds，不更動 B。
- 新預設輸出為 `outputs/120ue_single_serving_full_load_interference/`。舊 outputs 未作為新結果，也未覆蓋。
- README 更新模型、可直接放論文的英文 scenario、PC 驗證與新 bounds/模擬指令；舊修改報告加上歷史標記。
- 已檢查 analysis.py、study.py、phi_study.py、storage.py；它們未殘留舊 population 依賴或硬編碼 UE 分母，因此不為改名而修改。現有 CSV/Parquet helper 接收新 schema，checkpoint validator 在 simulation.py 版本化。

## 重要程式碼位置

- 設定與驗證：`config.py:101`。
- 固定訂閱：`scenario.py:12`。
- 單值關聯：`association.py:8`。
- Message-only partition：`algorithms.py:22`。
- 單一 population 的 epoch state：`regrouping.py:11`。
- 固定滿載外部功率：`channel.py:53`。
- 逐 UE 加總與驗證：`channel.py:65`。
- 群組最大值：`channel.py:105`。
- 服務與外部功率分離：`simulation.py:515`。
- 舊 checkpoint 拒絕：`simulation.py:294`。
- 新的 reward identity：`reward_pilot.py:22`。
- Counterfactual 同步外部干擾：`calibration.py:107`。
- 新診斷 schema：`diagnostic_trace.py:9`。
- 單一 serving slot identity：`diagnostics.py:231`。
- 核心人工測試：`tests/test_single_serving_interference.py:201`。

## 驗證命令與結果

在套件上一層目錄，以既有 `.venv\Scripts\python.exe` 執行（本機 Python 3.14）；requirements 的 numpy、scipy、pyarrow、matplotlib 均可匯入，沒有缺少依賴。維持專案 Python >=3.11 的語法。

```powershell
.\.venv\Scripts\python.exe -m compileall -q mira_leo_study
.\.venv\Scripts\python.exe -m mira_leo_study validate-config
$env:MIRA_LEO_RUN_SIMULATION_TESTS = '0'
.\.venv\Scripts\python.exe -m unittest discover -s mira_leo_study/tests -v
.\.venv\Scripts\python.exe .verification_120ue\static_check.py
```

- compileall：通過。
- validate-config：VALID；120 UE、3 messages、40 subscribers/message、1 serving、fixed_full_load、compact actions、100-slot regrouping；reward_bounds_available=false（刻意沒有執行 pilot）。
- unittest：**129 tests，123 passed、6 skipped，0 failures/errors**。6 個 opt-in 實際 simulation tests 按使用者要求跳過。
- 靜態檢查：所有 source/test modules 可匯入，**942** 個直接 function/constructor call signatures 可綁定，沒有 unresolved global。額外檢查常數字典鍵，未留下重複 metadata keys。
- 人工干擾測試涵蓋單/雙可見鄰星、全不可見、強服務星排除、HO 排除位置、worst UE=0.08、max(sum)≠sum(max)、功率 outage 獨立、策略無關輸入、shape/finite/nonnegative；另有 120 人訂閱/seed/hash、coverage fallback/TTT、全部不可見、部分成員不可見整組失敗、三策略初始化、epoch 人工 HO/owner reward、state pickle continuation、功率/SIC、report 包裝及 36 張人工診斷圖。
- 第一輪遇到 Windows sandbox 預設 Temp 寫入/刪除限制，改設 TMP/TEMP 為工作區 `.verification_120ue\tmp` 後完成上述測試。最終完整 log 位於工作區 `.verification_120ue\unit_verified.log`；靜態 checker 為工作區 `.verification_120ue\static_check.py`。這些是驗證材料，不是論文模擬輸出。

未執行：smoke、短跑/正式 multi-policy simulation、新 reward-bounds pilot、真正 simulation checkpoint/resume/on-off parity、480 s 正式數值驗證。人工 epoch/pickle 測試已通過，但不取代上述 integration tests。沒有生成校準數字或任何效能排名。

## 在 PC 上的後續工作與模型限制

依 README 先用此新場景執行 calibrate-reward-bounds，再用新的 bounds 執行 MIRA/Random/Fixed；不能使用舊 bounds。B=0.05 不重新估計。單位/物理近似沒有另行重設；不增加延遲或鄰星負載模型。

本模型採 cochannel/full-load 及原 directional-coupling 簡化，鄰星未 joint optimize；只能稱為 single-serving-LEO multicast optimization under dynamic multi-LEO interference and handover。若無衛星完整覆蓋，採明確 outage fallback；全員可見仍是 group success 的必要條件。

## 所有回寫檔案

- `DIAGNOSTIC_MODIFICATION_REPORT.md`
- `IMPLEMENTATION_STATUS.md`
- `MODIFICATION_REPORT.md`
- `MODIFICATION_REPORT_120UE_SINGLE_SERVING.md`
- `PHI20_POWER_OUTAGE_MODIFICATION_REPORT.md`
- `PHI30_POWER_OUTAGE_MODIFICATION_REPORT.md`
- `README.md`
- `REGROUPING_1S_COMPACT_MODIFICATION_REPORT.md`
- `__main__.py`
- `algorithms.py`
- `association.py`
- `calibration.py`
- `channel.py`
- `config.py`
- `configs/base.yaml`
- `diagnostic_plots.py`
- `diagnostic_trace.py`
- `diagnostics.py`
- `geometry_export.py`
- `metrics.py`
- `models.py`
- `paper_results.py`
- `plotting.py`
- `power_diagnostics.py`
- `regrouping.py`
- `reward_pilot.py`
- `scenario.py`
- `simulation.py`
- `sources.py`
- `tests/test_algorithms.py`
- `tests/test_association.py`
- `tests/test_channel_time.py`
- `tests/test_diagnostics.py`
- `tests/test_fixed_results.py`
- `tests/test_geometry_orbit.py`
- `tests/test_phi_study.py`
- `tests/test_power_outage.py`
- `tests/test_protocol_and_simulation.py`
- `tests/test_regrouping_strict.py`
- `tests/test_single_serving_interference.py`
