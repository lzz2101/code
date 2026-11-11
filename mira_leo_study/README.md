# MIRA-LEO P1：13°、15°、20°、25° 六組實驗

主要設定為 φ=15°，正式功率模式為 `csi_expanded_protected_pool`。唯一正式矩陣如下：

| run_id | policy | φ | 分組方式 |
|---|---|---:|---|
| mira_phi13 | mira_sw_ucb | 13° | SW-UCB compact |
| mira_phi15 | mira_sw_ucb | 15° | SW-UCB compact（主要設定） |
| mira_phi20 | mira_sw_ucb | 20° | SW-UCB compact |
| mira_phi25 | mira_sw_ucb | 25° | SW-UCB compact |
| random_phi15 | random_grouping | 15° | 每 epoch 從 compact actions 抽樣 |
| fixed_phi15 | fixed_grouping | 15° | a=4，仍每秒更新成員 |

保留 8 顆 synthetic LEO（800 km）、120 位靜止 UE、3 messages×40 UE、單一 serving LEO。其他 7 顆使用原本可見度控制的 fixed-full-load expected cross-link interference，不建立它們的用戶與排程。Coverage minimum 保持 10°。正式每組 480 秒／48,000 個 10 ms simulation slots；10 秒 warm-up、1 秒 regrouping、SW-UCB window 10 個 epochs。Layout / subscriptions / channel / policy seeds 固定為 42 / 1042 / 0 / 0。

## 功率規則與 P1

地理 edge 仍是 `max(member elevations) < phi`，等於門檻不算 edge。Scheduler 維持 Algorithm 2：edge-first 與 weighted marginal AoI reduction。

只對 scheduled groups，以當前原始線性 power gain 排序。若有 scheduled geo-edge，取其最大 gain 作 `g_ref`；非地理 edge 且 `gain < g_ref` 的群組晉升到 protected pool。晉升只改功率池，不改 `is_edge`、排程優先權或地理 edge penalty。Protected / ordinary 共用 70% / 30%，單池非空時獨享 100%，無 geo-edge 時全部 ordinary。

各池重算 `V_phi*(n-r+1)` floors，再均分剩餘功率。Target 若不符全域弱 CSI→強 CSI 的 nonincreasing power ordering，使用 L1 LP 修正；同時維持池預算 equality、floors 與總功率。Floors 過大、池預算與 SIC 不相容、LP 數值失敗均保留 schedule、實際功率為零、更新失敗 AoI；原因分別記錄。

`P1_slot = C - weighted-sum AoI`，AoI 採每 slot 更新後的 **slots** 單位。Bandit raw reward 額外減 `B*M*N*geo_edge_count`，B 固定 0.05。正式 P1 不含 penalty，也不先將 AoI 換成 ms。報告提供 full-horizon 與 post-warmup sums、means、running means。P1 越大越好，僅比較 achieved objective，不宣稱 global optimum。

## 安裝與程式驗證（不執行模擬）

在套件的上一層執行；Python ≥3.11。Windows PowerShell：

```powershell
Set-Location -LiteralPath 'C:\Users\lzz\Documents\我的論文code'
python -m pip install -r .\mira_leo_study\requirements.txt
python -m compileall -q .\mira_leo_study
python -m mira_leo_study validate-config
python -m mira_leo_study p1-ablation --help
$env:MIRA_LEO_RUN_SIMULATION_TESTS = '0'
python -m unittest discover -s .\mira_leo_study\tests -v
```

依這次使用者要求，交付時未執行 smoke、reward pilot、debug 或正式模擬。單元測試以人工群組、矩陣、CSV、圖表封裝與 checkpoint component 驗證。需要啟動實際模擬的 6 項 integration tests 預設停用。詳細清單見 [修改報告](MODIFICATION_REPORT_P1_ABLATION.md)。

## 在自己的 PC 執行

以下兩個命令都會真正執行校準與模擬，由使用者自行啟動。執行位置同上；`--config` 已有套件內預設值。

先短跑六組，所有資料放在獨立 debug 目錄：

```powershell
python -m mira_leo_study p1-ablation --debug-slots 1000 --seed 0 --outputs .\mira_leo_study\outputs\p1_ablation_13_15_20_25 --resume
```

正式六組，每組 48,000 slots：

```powershell
python -m mira_leo_study p1-ablation --seed 0 --outputs .\mira_leo_study\outputs\p1_ablation_13_15_20_25 --resume
```

Runner 會為四個 φ 各自產生／驗證新 allocator 的 reward bounds。Random/Fixed 共用 MIRA(15) 的 φ15 bounds。B 不重新校準。Debug 的短視窗 bounds 與正式 bounds 分開保存；不接受舊 φ20 allocator 的 cache/checkpoint。首次正式執行的成本包含四份 reward pilot 加六組模擬，因此不只六個 480 秒模擬視窗的運算量。

`--resume` 會接續相同 identity 的 checkpoint，已完成且通過 CSV 核對的 runs 可重用。更換 config、schema 或輸出追蹤選項時請使用新的輸出資料夾。不要同時在相同輸出路徑啟動兩個程序。可選 `--save-group-trace` 匯出六組完整逐群組 traces；檔案會明顯增大，預設只保留必要 slot metrics 與 φ15 地理資料。

## 輸出與地理分析

`outputs/p1_ablation_13_15_20_25/` 下各 run 有獨立目錄。`comparison/` 含六組 CSV、四個 MIRA 的 ablation、φ15 三種 policies、P1 decomposition、running objective、功率可行性、paired absolute differences 與 `environment_identity_check.json`。六組的 layout、subscriptions、trajectory、channel/environment trace hashes 與 slots 必須相同；config_id 可隨 φ 改變。

`mira_phi15` 每秒保存全部 candidate groups 與每名 UE 的原始位置、仰角、gain、membership、服務衛星 ground point；每 slot 記錄 geo-edge occupancy。`geo_edge_phi15/` 提供群內 all-pairs 距離、centroid radius、凸包面積，以及**同一 snapshot / serving satellite**的 edge-group centroid 間距。Singleton 的 pairwise distance 為 NA。

地圖使用保存的 UE 座標，預設 early/middle/late 三個 snapshots，不挑選「剛好有 edge」的時間。低仰角不直接代表群內位置接近，也不表示不同 edge groups 聚在同一地區。重複 UE / time 觀測的權重與時間、衛星適用範圍會在地理報告中標明。

`figures/` 的比較圖和三張地理圖只在讀入完整、核對過的真實 run traces 後產生；`manifest.json` 記錄圖與來源 CSV。交付時只放 `NOT_RUN` 報告、PENDING 矩陣與 provenance，沒有模擬數值或正式圖表。

比較階段逐列讀 CSV，只保留六組緊湊 numeric arrays；模擬 traces 分 chunk 寫入。地理 member traces 只讀三個地圖 snapshots，避免一次載入每秒的全部 member dictionaries。

## 相容性

Checkpoint schema 7、diagnostic schema 3、reward protocol `p1_csi_expanded_protected_pool_v7`。舊 schema/bounds 不可續用。CSV 在 checkpoint 恢復時會裁掉未提交 slot，並支援完成檔案改名途中中斷的恢復。

`phi_study.py` 的歷史 full-action φ/B APIs 與部分純舊功率 helper fixtures 標為 `legacy_archived`，用於既有相容性回歸測試；新正式 runner 不引用它們，正式 simulator 直接拒絕舊 allocator。`paper-results` 保留 φ15 三種 policies 的相容入口；本次六組實驗請使用 `p1-ablation`。沒有新增舊 allocator 第七組、其他角度的 Random/Fixed，PPO/GA/AIM-static 仍為 NOT_IMPLEMENTED。

## 15° MIRA 與 baseline 的 AoI 圖

完成六組模擬後，`figures/baseline_phi15_weighted_aoi.png` 比較 MIRA / Random / Fixed 的完整時段平均 weighted-sum AoI（ms）；`figures/baseline_phi15_convergence_aoi.png` 顯示三種 policies 的 running-mean weighted-sum AoI（slots）。兩者都只使用 φ15 三組資料，來源為 `comparison/baselines_phi15.csv` 與 `comparison/p1_running_objective.csv`，可透過 manifest 追溯。

本次 Documents 專案修改尚未執行模擬，所以圖檔將於你在自己的 PC 跑完後產生。

## 16 分鐘「僅服務衛星」CSI 診斷：一次輸出三張圖

新增獨立程式 `csi_elevation_serving_satellite.py`。每個 0.01 秒時槽沿用目前的 coverage-first、hysteresis、TTT 與換星規則，選出全部 UE 共用的一顆 serving satellite，再只統計該星仰角 ≥10° 的 UE 鏈路。使用原始 CSI power gain，不做干擾調整，也不執行 multicast 分群、排程或 AoI。

從專案上一層，在 PowerShell 執行：

```powershell
Set-Location -LiteralPath 'C:\Users\lzz\Documents\我的論文code'
& '.\.venv\Scripts\python.exe' -m mira_leo_study.csi_elevation_serving_satellite
```

預設正式時長為 960 秒／96,000 slots，一次產生三張圖，放在 `C:\Users\lzz\Documents\我的論文code\mira_leo_study\outputs\csi_elevation_16min_serving_only`：

- `01_elevation_vs_gain.png`：僅服務星的仰角 vs 原始 CSI 散點，以及對應的期望增益曲線。
- `02_bottom20_fraction_by_elevation.png`：各 1° 仰角區間中，落入服務星全域最弱 20% CSI 的比例。
- `03_bottom30_fraction_by_elevation.png`：各 1° 仰角區間中，落入服務星全域最弱 30% CSI 的比例。

兩個全域門檻都以同一批「可見服務星鏈路」重新計算，不沿用全部八星版本的門檻。曲線的分母是該仰角區間內的服務星觀測數，不是在各角度分別取 20%／30%。沒有樣本的角度留空；cutoff ties 全數納入並在 summary 記錄實際比例。

另保存 `association_trace.csv`（每 slot 的服務星／換星／可見 UE 數）、`sampled_scatter_points.csv`（散點的時間、衛星與 UE ID）、`elevation_bin_statistics.csv`（20%／30% 共用的完整統計）及 `run_summary.json`。散點最多抽樣 100,000 點；門檻、比例與期望曲線使用全部服務星樣本。每個點是一筆 UE–衛星–時間觀測，不是一名獨立 UE。

如要在自己的 PC 先短跑 1,000 slots，可加 `--smoke-slots 1000`，資料會另外放在 `debug\slots_1000`。`--keep-raw` 可保留完整服務星 gain/bin 二進位檔；預設完成後清除暫存。原本全部八顆衛星的兩支 CSI 程式、資料夾與 480 秒 P1 設定均保留。

本次只執行人工資料測試與語法／匯入／CLI 驗證，沒有執行新的真實 CSI 診斷。詳細內容見 `SERVING_CSI_DIAGNOSTIC_REPORT.md`。
