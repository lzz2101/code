> 歷史紀錄：以下內容屬於 120-UE single-serving 重構前的版本，不能作為新模型的結果或目前 API 規格。最新狀態見 [MODIFICATION_REPORT_120UE_SINGLE_SERVING.md](MODIFICATION_REPORT_120UE_SINGLE_SERVING.md)。

> 歷史版本：本檔描述 20 度、功率無解即停止的設定。現行 30 度與 zero-power outage 行為請見 PHI30_POWER_OUTAGE_MODIFICATION_REPORT.md。

# 1 秒 Regrouping / Compact / Strict Algorithm 2 修改報告

## 來源與本次驗證範圍

規格為 `CODEX_MIRA_LEO_1S_COMPACT_STRICT_ALG2_TWC_POWER_CSI_NO_OFFBYONE_FIX.md`。使用者補充指定 Algorithm 2 的論文為 `C:/Users/lzz/Downloads/Copy_AoI_LEO (3).pdf`，已核對第 6 頁 Algorithm 2、第 7 頁 Algorithm 3、第 3 頁 CSI/edge 定義、第 5 頁式 (17c) 及第 8 頁 Theorem 3。另讀取 `codes/IEEE_TWC/Environment.py::PowerAllication()` 與 `CalculateMaxScheduleGroups()` 及 Train.py 的 power 參數。

依使用者直接要求，本次僅修改程式並執行語法、匯入、CLI、靜態及非模擬單元檢查。沒有執行 run_simulation、smoke、reward pilot 或 formal experiment；MD 要求的實際 350-slot / 480-second 數值不在本次執行範圍，沒有製造替代結果。

## 修改檔案

| 檔案 | 重點 |
|---|---|
| configs/base.yaml | compact actions、regrouping_interval_s=1、twc_aim_staircase、新 protocol experiment name；待新 pilot 產生 frozen bounds。 |
| config.py | finite positive interval、interval/slot 與 window/interval 整除檢查；100-slot interval、10-epoch window；正式 validator 要求 compact / 1 s / TWC mode；tau list 也需整 epoch。U=62 恢復作 Algorithm-2 cap，52/34 仍只作 diagnostic。 |
| models.py | 新增 immutable GroupPartitionEntry，用 tuple 保存 membership，支援 pickle。 |
| algorithms.py | 拆 build_group_partition / materialize_groups_from_partition；make_groups 保留 wrapper。Strict Algorithm 2 取代 power admission。TWC targets / continuous residual / power-only projection / no-drop allocator；UCB 支援 window_steps，保留舊 window_slots API。 |
| regrouping.py（新增） | held action/partition/owner、global epoch decision、整秒 rebuild、per-slot dynamic materialization、reward aggregation / finalize、decision histogram；state 全部可 checkpoint。 |
| simulation.py | 每 slot current CSI → held partition materialization → strict Algorithm 2 → TWC power → interference / SIC / AoI；mid-epoch handover 保留 partition / owner；新增 trace、projection diagnostic、bandit observation CSV；schema=3；真正不可行寫 error artifact 並停止。 |
| calibration.py | shared counterfactual evaluator 支援傳入 held partitions/current channel；正式 TWC invalid sample 明確報錯，不校準假值；歷史 B API 保留。 |
| reward_pilot.py | compact-only samples，a=4 baseline membership 每秒更新，snapshot/windows 對齊 epochs；context 含 protocol version / cadence / compact / power mode，拒用舊 bounds。 |
| paper_results.py | legal actions 從 config 取得；驗證 epoch held action/partition/owner、boundary events、current CSI indices、Algorithm-2 counts、decision / observation counts；schema=3 cache；failed model 不標為 completed comparison；新報告語意。 |
| __main__.py | 新預設 output root；validate-config 顯示 compact / 100 slots / 10 epochs / scheduler / allocator；正式 CLI 保持無 phi/B 搜尋。 |
| tests/test_algorithms.py | literal 歷史測試明確使用 legacy mode；projection 無解要求 schedule 不變、removed=[]，不再期待 drop。 |
| tests/test_fixed_results.py | 依新的 strict scheduling 要求替換上一版 admission assertions；更新人工 output fixtures / context；baseline 共用 pipeline 檢查要求不呼叫 admission helper。 |
| tests/test_phi_study.py | 歷史 fixture 顯式 full action set、per-slot interval、literal mode，保留歷史 protocol；更新 cap snapshot 語意。 |
| tests/test_regrouping_strict.py（新增） | config、350 artificial component ticks、frozen partition、dynamic CSI/edge、epoch reward、handover owner、pickle/resume、UCB window/exploration、TWC exact floor/residual、feasible/infeasible SIC repair、current/future CSI isolation、formal failure handling、opt-in 150/300-slot resume test。 |
| README.md / IMPLEMENTATION_STATUS.md | 現行設定、指令、依賴、模型限制、驗證範圍與輸出欄位。 |
| MODIFICATION_REPORT.md | 加上上一版歷史報告標記，保留原紀錄。 |
| REGROUPING_1S_COMPACT_MODIFICATION_REPORT.md | 本次交付與驗證報告。 |

metrics.py 的 AssignmentTracker、原 orbit/channel/AoI 公式及 reference TWC 原始檔沒有修改。

## 最終正式設定

| 設定 | 值 |
|---|---|
| Action set | compact = [1,2,4,5,8,10] |
| 每 message subscribers | 40 |
| 每 satellite messages | 3 |
| 每 serving satellite 最大 candidate groups | 30 |
| Physical slot | 0.01 s |
| Duration | 480 s / 48,000 slots |
| Regrouping interval | 1.0 s / 100 slots |
| SW-UCB window | 10.0 s / 10 global regrouping epochs |
| phi / B | 20 deg / 0.05 |
| Power mode | twc_aim_staircase |
| Power | P_sat=1.0, V_c=0.7, V_nc=0.3, V_phi=0.0005 |
| Fixed baseline | a=4 |

Carrier=20 GHz、bandwidth=50 MHz、SCS=60 kHz、R_theta=1e-5、weights=[0.7,0.2,0.1]、xi=1/sqrt(2)、HO=25 ms、fixed ISL hop=5 ms、iid-per-link-per-slot fading 均保留。

## Algorithm 2 的實作

位置：algorithms.py::schedule_groups（約第 91 行）。原 weighted marginal AoI benefit 公式保留，scheduler 內已沒有 check_power_feasibility、evaluate_power_plan 或 allocate_power 呼叫。

```python
U = floor(-0.5 + sqrt(0.25 + 2 * P_sat / V_phi))  # 62
if len(edge) > U:
    selected = sorted(edge, key=(-priority, message_id, group_id))[:U]
else:
    selected = list(edge)  # 包含 priority <= 0 的 edge
    positive = sorted([g for g in nonedge if g.priority > 0], key=...)
    selected += positive[:U - len(edge)]
```

Top-U edge 不做 positive filter。Nonedge 才需正收益。人工測試包含 cap=3 / edge=5 的負 priority edge case、cap=5 / edge=2 的 all-edge case，以及 nonedge priorities 10、5、1、0、-1。

單一 class cap 62 取自原 paper formula，沒有用 mixed pool 的 52 / 34 修改 scheduling；compact 30 groups 仍不代表 nonedge 無條件全排。

## Membership / CSI / Reward 時序

- algorithms.py::build_group_partition（約第 22 行）只產生 message/group/member tuples，無 gain/power/success state。
- materialize_groups_from_partition（約第 47 行）每 slot 由 held membership 與該 slot channel/elevation 重新建立 Group。
- regrouping.py::groups_for_slot（約第 33 行）只在 slots 0、100、200… select action / sort / build partition，其他 slots 直接使用 held membership。
- simulation.py 約第 400 行產生 `serving_channel_amplitude(..., slot, distance)`；約第 420 行 materialize；約第 464 行 allocate。Power 讀取 current group.gain，沒有舊 CSI cache 或 future stochastic CSI。
- `csi_slot_index` 與 `power_csi_slot_index` 都記 current slot，formal validator 要求與實際 row slot 相同。

Handover 不觸發 off-grid rebuild。Serving satellite 可以改變，epoch owner 仍是 boundary 選 action 的原 agent，reward 回傳原 owner。下一個 boundary 才由新 serving agent 選擇。

Reward 每 slot 正常計算並用原 owner 的 frozen normalizer 轉成 [0,1]；epochReward = sum(normalized slot rewards) / sample count。整秒 epoch 只 update 一次，真正最後不足 100 slots 時可對 partial epoch update 一次。stop_after_slots 暫停不構成 true final slot，不提前 finalize。

SW-UCB 使用 global epoch index，window_steps=10。保留舊 constructor window_slots alias 供歷史程式使用；production 的 learning step 不再是 10-ms physical slot。最近 10 秒只有最近 10 個 epoch observations，inactive agent 的 observations 仍會隨 global time 過期。

Theorem 1 的正文 slot notation 需要與實作的 epoch index 對齊；本次沒有宣稱原 slot-based statement 完全不變。

## TWC / Algorithm 3 Power 轉譯

位置：algorithms.py::twc_aim_class_budget、twc_aim_staircase_target、evaluate_twc_aim_power_plan（約第 280 行）、allocate_power（約第 351 行）。

1. 只使用 Algorithm 2 的 scheduled set。
2. Mixed class pools 為 0.7 / 0.3；single edge 或 single nonedge 使用完整 1.0。
3. 每 class 依 current slot gain weak → strong 排序，floor = V_phi × (N-r+1)。
4. 每組再分到 (class budget - sum floors) / N 的 continuous float residual。
5. 如 target 違反 global SIC，使用既有 LP 的 L1 power-only projection，保留全部 groups、class equalities、per-group floors、total power / global ordering。
6. 無解時 return invalid plan；正式 simulator 寫 MODEL_CONSISTENCY_ERROR 並 raise，沒有 drop loop，也不更改 scheduled flags。

Reference TWC 的 `PowerAllication()` 是 class split / staircase / residual 的結構參考；fractional residual 的 math.floor 沒有移植，因 paper Algorithm 3 指定平均分配 continuous residual。Reference `CalculateMaxScheduleGroups()` 的 indexing / off-by-one 完全未修改，也未新增修正 helper。原專案 paper staircase 使用 N(N+1)/2 的 floor 語意保留。

人工數值驗證：

- N=3，V_phi=0.0005 → floors=[0.0015,0.0010,0.0005]。
- Budget=0.3 → residual=0.297，每組 share=0.099，確實加回 target。
- N=30 → floor_total=0.2325 < 0.3，也小於 0.7。
- Mixed target budgets 與 single-class full power 均通過單元檢查。
- 可行的 edge / nonedge / edge 交錯案例通過 power-only projection，before violations > 0、after violations=0、class sums 不變、removed=[]。
- 不可行的 weak nonedge / strong edge 各一組案例，確認 MODEL_CONSISTENCY_ERROR、scheduled flags 不變、removed=[]，沒有 fake valid 結果。

實際 formal 的 projection 次數尚未量測；執行後由 sic_projection_count 與逐 slot sic_projection_used 記錄。Allocator 所有模式都不再有 defensive-drop loop。

## 尚存的模型一致性限制

Compact set 能保證 class floor 容量充足，不能保證固定 class budget 和 global SIC 在所有 current CSI order 下同時可行。例如 weak nonedge 與 strong edge 各一組的 class sums 強制為 0.3 / 0.7，但 global SIC 要求 weak power >= strong power，這個實例數學上無解。

依本次規格，這時必須明確停止，不可藉由改 scheduler、budget 或刪群組掩蓋。這是研究模型的限制，並非單元測試忽略的 violation。Formal output 若失敗，run directory 與 root summary/report 都標記 model consistency failure。是否在真實整段 realization 遇到此情況尚未實測。

## Checkpoint / Output

Schema 已升為 3，舊 schema=2 直接拒絕 resume。所有 held partitions/actions、owner、reward sum/count/finalized、last boundary、decision histogram 都在 state，pickle round-trip 能保存不可變會員 tuple。Resume identity 檢查先於寫入 config/snapshot。

每 slot trace 保存：epoch、boundary、held action、owner、membership_hash、interval、window epochs、CSI indices、strict scheduler / allocator、U、candidate / scheduled edge/nonedge、nonedge positive count、class budgets / floors / residuals、target/final power（group trace）、projection before/after counts、removed/drop diagnostic。

Action occupancy 每 physical slot 計一次；action decision 只在 boundary 計一次。新增 bandit_observations.csv，保存每個 epoch 的 owner、action、sample_count、平均 normalized reward。舊 rejected-by-power 欄位保留為零；改用 Algorithm-2 cap 與非正 nonedge 區分未排原因。

Overhead tracker 保持 stable logical assignments。人工 component 檢查確認非 boundary regrouping events=0；initial / coverage exit / handover 分開。AIM-static、GA、PPO 未實作的狀態保留，不產生曲線或假值。

## 測試與驗證結果

已執行：

```powershell
.\.venv\Scripts\python.exe -m compileall -q .\mira_leo_study
.\.venv\Scripts\python.exe -m mira_leo_study --help
.\.venv\Scripts\python.exe -m mira_leo_study validate-config
.\.venv\Scripts\python.exe -m mira_leo_study paper-results --help
```

另透過 unittest discovery runner 執行 `mira_leo_study/tests/test_*.py`，將 tempfile / matplotlib cache 導到 workspace `.verification_tmp`，以符合本環境 filesystem sandbox。逐一 import 全部頂層 modules 與 AST state-key audit、scheduler power-independence audit 也通過。

結果：79 tests，74 passed，5 skipped，0 failed / errors。五個 skipped 都是明確 opt-in 的實際 simulation integration tests，沿用 MIRA_LEO_RUN_SIMULATION_TESTS=1 啟用機制；沒有為掩蓋失敗而降低 assertion 或刪除測試。

新增及更新測試覆蓋：

- Invalid interval / fractional epoch window / tau list 拒絕。
- 350 artificial controller ticks：boundary calls=4，action sequence starts [1,2,4,5]，每 epoch action/hash 固定，observation count=4（最後 partial 50 slots 一筆）。
- Random 只在 boundary 抽樣並可重現；Fixed 全部為 4 且 250 ticks 只 build 三次。
- 人工 slot 50 handover：不重新選 action / membership，handover assignments 增加，reward 給原 owner。
- 原 owner 每 epoch 平均 reward、初始 compact exploration [1,2,4,5,8,10]、global epoch pruning。
- Tick 150 pickle 時只完成一個 epoch；第二個 epoch count=50，resume 後與不中斷的 component bookkeeping 相同。
- 固定 membership 下 current CSI 改變立即改變 power order；改 future array 不改 current scheduled/power/rate/success。
- Algorithm 2 兩分支、負 priority edge、positive nonedge、cap / deterministic ties、scheduler 禁止呼叫 power helper。
- TWC exact floors / continuous residual / 30-group invariant / mixed/full budgets / feasible and infeasible SIC projection / no drops。
- Formal validator 拒絕 action 20；stubbed failure 不產生 completed comparison。
- Output packaging 使用暫存人工資料檢查三 policy + tau API 與 CSV / JSON / PNG / PDF / Markdown artifacts，完成後清除，不放進 outputs。

原有 full-action archived phi fixture 已明確 override 回 full / per-slot / literal 設定，沒有改成 compact 來讓歷史 protocol 測試過關。上一版 power-admission tests 改寫為本次明確要求的 strict Algorithm-2 assertions。

真正的 350-slot simulation、300-slot / pause150 的 full simulator resume、三方法實際 environment trace 比對及 reward pilot 都未執行。人工 bookkeeping / partition / allocator 測試不能當作論文模擬數據。

## 在使用者 PC 執行

新預設 root：outputs/fixed_phi20_B005_rg1s_compact。舊 full-action reward bounds 不相容，需新 pilot；本次未重新產生。既有 outputs 保留。

```powershell
.\.venv\Scripts\python.exe -m mira_leo_study calibrate-reward-bounds --outputs .\outputs\fixed_phi20_B005_rg1s_compact --seed 0
.\.venv\Scripts\python.exe -m mira_leo_study paper-results --slots 350 --seed 0 --outputs .\outputs\fixed_phi20_B005_rg1s_compact
.\.venv\Scripts\python.exe -m mira_leo_study paper-results --seed 0 --outputs .\outputs\fixed_phi20_B005_rg1s_compact --resume
```

上述指令會真正執行 pilot / simulation，這次交付沒有執行。完整參數與輸出目錄見 README.md。
