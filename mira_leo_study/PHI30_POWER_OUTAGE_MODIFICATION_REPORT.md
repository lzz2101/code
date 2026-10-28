> 歷史版本：使用者已將門檻改回 20°，並保留 zero-power transmission failure。現行設定請見 PHI20_POWER_OUTAGE_MODIFICATION_REPORT.md。

# phi=30 與功率無解視為傳輸失敗：修改及診斷報告

## 使用者本次要求

1. 將正式 edge threshold 從 20 度改為 30 度。
2. 將真正無解的 power/SIC 配置當作該衛星本 slot 更新失敗，繼續執行。
3. 檢查 30 度是否還有類似情況。

使用者這次的指示覆蓋上一版 MD 的「phi=20、功率無解即停止」設定。Strict Algorithm 2、compact action set、1-second frozen membership、current-slot CSI、fixed class pools 與 global-SIC repair 都保留。

## 修改內容

| 檔案 | 重點 |
|---|---|
| configs/base.yaml | phi_theta_deg=30、infeasible_power_handling=transmission_failure、新 experiment identity；reward bounds 清為待重新校準。 |
| config.py | 固定 30 度 / B=0.05 常數、outage policy validation；其他物理參數不變。 |
| algorithms.py | 保留 LP status/message；只有 solver 明確 status=2 且沒有 floor error 時轉成 outage。純 power-plan valid 仍為 false；allocate_power 保留 schedule、actual_total_power=0、success=false，不傳送不可行功率。 |
| simulation.py | 分開記錄 power-outage cohort-slots / distinct physical slots、受影響 scheduled groups / devices；不將已處理 outage 當作 unhandled constraint error。AoI/reward 正常反映失敗；未發射群組不進一般 rate / SIC decoding-failure counters。Schema=4。 |
| calibration.py | Pilot counterfactual 接受已處理 outage，跳過其 decoding，actual transmitted power=0；其他衛星的干擾計算也使用這個零功率。 |
| reward_pilot.py | 失敗樣本保留在 reward bounds samples，不跳過；payload 記失敗 sample 數。Phi=30 與 outage policy 進入新的 context / protocol version。 |
| paper_results.py | 接受 COMPLETED_WITH_POWER_OUTAGES，逐列驗證失敗時 zero power / no attempted transmission / zero successful groups / LP status=2，並核對 outage counters。比較摘要與 report 保存各方法失敗次數。 |
| __main__.py | 新預設輸出目錄 outputs/fixed_phi30_B005_rg1s_compact_outage，validate-config 顯示 30 度與新 policy。 |
| sources.py | 更新正式 phi 與 outage 模型的設定來源說明。 |
| power_diagnostics.py（新增） | 可重現的 fixed pilot snapshot-grid power-only scan，不跑 AoI / learning policy。 |
| tests/test_power_outage.py（新增） | No-transmission 行為、原 schedule 保留、數值/floor 錯誤不掩蓋、AoI/reward 失敗規則、pilot 另一衛星照常傳送、zero interference、formal validator 稽核。 |
| tests/test_fixed_results.py / test_regrouping_strict.py | 更新 production 30 度與 output fixture，保留歷史研究 fixture。 |
| README.md / IMPLEMENTATION_STATUS.md | 新指令、輸出欄位、模型語意、診斷結果。上一版 report 標記歷史。 |

## 功率失敗的確切定義

原流程仍是 current CSI → frozen partition materialization → strict Algorithm 2 → TWC/AIM target → global-SIC power-only repair。

如果 repair LP 明確回報 infeasible（status=2）：

- scheduled set 不變，removed_groups=[]。
- 該 serving satellite 本 physical slot 不發射，所有其 scheduled group updates 失敗。
- 所有實際 group.power=0；actual_total_power=0；原 target / floor / ordering diagnostics 保留供 audit。
- 不將 infeasible target 代入 SINR，不讓無效功率干擾其他衛星。
- AoI 使用既有 unsuccessful recursion，包括適用的 handover penalty。
- Slot reward 以 success_count=0 與真實 post-failure AoI 計算；candidate edge-count penalty 不變。
- MIRA / Random / Fixed 與 reward pilot 使用同一 allocate_power 與 failure 語意。

這個狀態是新增的 power-outage 模型假設。不能說 proposed plan 變可行，也不能把失敗樣本從統計刪掉。

LP status=4 等數值問題、missing dependency、floor/configuration error、scheduled set mutation 等仍屬錯誤，保持停止。infeasible_power_handling=error 仍可供非正式 / 歷史 diagnostic 使用，但正式 validator 要求 transmission_failure。

## 指標與資料語意

- power_plan_valid：proposed power plan 是否可行，outage 時為 0。
- transmission_failure_power：已確認 LP infeasible 並視為更新失敗。
- transmission_attempted / transmitted_groups：是否真正發射；outage 均為 0。
- planned_total_power 與實際 total_transmitted_power 分開。Target 通常加總 1，outage 實際為 0。
- power_infeasible_cohort_slots：累計失敗 cohort-slot 數。
- power_outage_slots：至少一個 cohort 發生 outage 的 physical slot 數。
- power_failure_scheduled_groups / power_failure_device_opportunities：因 power outage 失敗的排程機會。
- invalid_power_or_sic_count：未處理的 constraint violation，不混入已明確記錄的 zero-power outages。
- sic_violation_after_count 可保留 rejected plan 的 ordering diagnostic；transmitted_sic_violation_count 必須為 0。

完成且有 outage 的 status 為 COMPLETED_WITH_POWER_OUTAGES；reward clipping 異常仍保留 REWARD_RANGE_MISMATCH。Schema 升為 4，舊 checkpoint 無法混用。舊 phi=20 / fail-stop bounds context 不相容，需要重新 pilot。

## 30 度是否仍會無解

已執行與上一輪相同的 seed=0、fixed pilot snapshot grid：

- 窗口 [0,30)、[225,255)、[450,480) 秒。
- 每秒一個 snapshot，共 90 個時刻。
- 每個時刻 2 cohorts × 6 compact actions，共 1080 組配置。
- 幾何與帶 hysteresis / TTT 的 association 由 current config 在每個 physical slot 重建。
- 每個 snapshot 用當下 indexed CSI；這個診斷假設所有 candidates 都 scheduled。
- 沒有演進 AoI、跑學習策略或執行 reward pilot。

| Threshold | LP 無解配置 | 檢查總數 |
|---|---:|---:|
| 20 deg（上一輪保存資料） | 8 | 1080 |
| 30 deg（本輪） | 170 | 1080 |

30 度仍有同類衝突，在此取樣範圍內增加。910 組 power plan 可行；其他類型 invalid=0。170 個 LP 無解配置全部驗證會轉為零功率 failure。

| Action | 無解配置 |
|---:|---:|
| 1 | 0 |
| 2 | 3 |
| 4 | 19 |
| 5 | 20 |
| 8 | 52 |
| 10 | 76 |

Cohort index 0 有 33 組、index 1 有 137 組。門檻增加會改變 edge / nonedge 組成，並不保證固定 0.7/0.3 預算與 stochastic global CSI ordering 更相容。

這些數字是結構性功率診斷結果，不是完整 480 秒的實際 policy failure rate。真正 Algorithm 2 可能不排非正收益 nonedge，且不同 policy 的 held partition 與 action 不同，實際次數要由使用者之後執行記錄。

診斷產物：

```text
mira_leo_study/diagnostics/pilot_power_phi30_seed0_all_candidates/
  README.md
  summary.json
  power_cases.csv
  infeasible_cases.json
```

可重現命令：

```powershell
.\.venv\Scripts\python.exe -m mira_leo_study.power_diagnostics
```

## 驗證

- compileall：通過。
- validate-config：VALID，phi=30、B=0.05、compact、100 slots per epoch、10 window epochs、transmission_failure。
- unittest discovery：87 tests；82 passed，5 opt-in simulation tests skipped，0 failures/errors。
- 新增 8 個 outage component tests；人工 pilot counterfactual 檢查 failed satellite power=0、AoI 增加、另一衛星仍成功，並確認不計入其 cross-satellite interference。
- Numerical status=4 與 floor overflow 保持 unhandled error，沒有降低 assertion 或跳過 violation。
- Formal validator 驗證 zero-power outage，也拒絕宣稱 outage 卻仍有發射功率的資料。

本次沒有執行實際 reward pilot、1000-slot evaluation、480-second study 或 benchmark。不要將此診斷 CSV 當作 AoI 比較成果。所有既有 outputs 與先前診斷資料保留。

## 使用者 PC 的下一個命令

```powershell
.\.venv\Scripts\python.exe -m mira_leo_study paper-results --slots 1000 --seed 0 --outputs .\outputs\fixed_phi30_B005_rg1s_compact_outage
```

新資料夾會自動先產生新 phi=30 / outage-policy reward bounds。完整命令移除 --slots 1000，並可加 --resume。
