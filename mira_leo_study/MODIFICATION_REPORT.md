> 歷史版本報告：本檔描述上一版 per-slot / power-feasible-admission 實作。現行版本請見 REGROUPING_1S_COMPACT_MODIFICATION_REPORT.md；以下內容不代表目前正式設定。

# 修改交付報告

本次依 CODEX_MIRA_LEO_POWER_FEASIBLE_PAPER_RESULTS_RANDOM_FIXED.md 修改既有 simulator。依使用者的直接要求，只驗證程式，不執行任何實際 simulation、reward pilot 或 paper-results experiment；Markdown 中要求提供短模擬數值的部分由此限制取代。

## 修改檔案與重點

| 檔案 | 修改重點 |
|---|---|
| algorithms.py | 保留 AoI priority 與 edge-first；正 priority 才進候選；逐組完整功率 admission。新增純 evaluate_power_plan / check_power_feasibility，allocator 共用計算。保留 projection 的 defensive fallback。 |
| config.py | 固定研究 validation；正功率與 split 一致性；62/52/34 floor-only diagnostics。舊角度 validation / constant 只供歷史 API。 |
| configs/base.yaml | phi=20、B=0.05，移除 search、beta_B、B0 參數；保留 pilot_fixed_clip；新增正式 output / tau list / pilot windows 設定。 |
| simulation.py | 三方法共用完整 pipeline；抽出 action selector；逐 cohort/slot 功率拒絕原因、raw/running AoI、環境 hash、device re-signalling、action change；CSV 與 Parquet 並存；更新 checkpoint identity。 |
| metrics.py（新增） | running mean、overhead normalization、穩定 logical assignment tracker、可 checkpoint 的 trace hash。 |
| reward_pilot.py（新增） | 固定 phi/B 的 reward bounds pilot、sample CSV、frozen identity 與 config/seed 校驗。完全不呼叫 B estimator。 |
| paper_results.py（新增） | paper-results 正式 workflow、三方法 CSV/JSON、PNG/PDF、scheduler diagnostic、tau list overhead、環境/功率/slot consistency checks、實測排名與 unavailable baseline 報告。 |
| __main__.py | 新增 calibrate-reward-bounds、paper-results、validate-config；移除所有角度/B 搜尋與舊參數搜尋入口；smoke 不再覆寫 B 或 reward normalization。 |
| storage.py | 新 run 的 CSV writer 不會 append 前一次結果，保留 resume 行為。 |
| calibration.py | 標明舊 B API 為 archived；保留共同 counterfactual evaluator 供新 reward pilot 使用。 |
| phi_study.py、study.py | 明確標示 archived，不由正式 CLI 呼叫。 |
| tests/test_fixed_results.py（新增） | power admission、純檢查、mixed/single-class、非正收益、SIC、baseline selector、RNG、normalization、overhead exclusions、人工資料 output packaging、可選真實整合測試。 |
| tests/test_phi_study.py | 舊研究測試使用明確 historical fixture，更新 snapshot 欄位；會實際模擬的方法改成明確 opt-in。 |
| tests/test_protocol_and_simulation.py | 短模擬設定不再覆寫 B=1；要求零 invalid/drop；實際 simulation 測試 opt-in。 |
| tests/__init__.py | MIRA_LEO_RUN_SIMULATION_TESTS=1 才啟用實際模擬整合測試。 |
| README.md、IMPLEMENTATION_STATUS.md | 新流程、完整輸出路徑、單位、overhead 定義、tau 語意、安裝與驗證方式。 |
| MODIFICATION_REPORT.md（本檔） | 交付範圍、驗證證據與尚未產生的實測結果。 |

## 新排程的實際流程

1. 原公式計算 w * Gamma，不改 propagation、handover 或 ISL 的物理意義。
2. 所有 groups 先取消 scheduled，power 歸零。
3. edge 與 non-edge 都只接受 priority > 0；各 class 依 priority/message/group 排序，保留 edge-first。
4. 對 selected + candidate 呼叫 check_power_feasibility；不讀 capacity 作 admission cap，也不為排滿而選組。
5. 共用 evaluate_power_plan 計算 class budgets、V_phi rank floors、原 literal residual power、global SIC ordering、class/total power。
6. 只有 valid 才加入；其餘記錄 floor / SIC / total-power rejection。
7. allocate_power 對最終 scheduled set 使用同一個 helper 實際寫入 power；後續 decoding / throughput / AoI 使用實際 scheduled flags。

single-class 仍可使用整個 P_sat=1；mixed class 仍為 0.7 / 0.3。62、52、34 只是 floor-only diagnostics。Reward edge penalty 仍使用所有 edge candidate groups，分類仍為 max(group elevation) < 20 度。

literal mode 沒有 allocator drop；projection 對繞過 admission 的外部呼叫保留 defensive drop。Simulator 記 post_schedule_power_drop counter；paper-results 若有任何 drop 或 invalid allocation 就停止報告。純檢查不改 scheduled、power、success、decoding 或 metadata。

## 固定參數、停用入口與 baseline

固定值位於 configs/base.yaml，且 paper-results / reward pilot 會使用 validate_fixed_study_config 檢查。正式預設仍為 literal_with_checks、pilot_fixed_clip，SW-UCB window=10 s。

舊 CLI 的 pilot-b、phi-sweep、phi-report、calibrate（B0）、screen、joint、evaluate、robustness、舊 report 搜尋流程已移除。正式流程不輸出 B_by_phi、selected_phi、by-phi comparison 或 B-star selection。歷史 Python API 與既有 outputs 保留，不刪除使用者資料。

Random 以獨立 indexed policy RNG 均勻選合法 action。Fixed selector 在 100 個純函式測試 decision 中全部為 4。純 action-selector 單元測試樣本（policy_seed=0、satellite index=3、slot=0..9）為：

- Random: [1, 8, 10, 8, 8, 8, 20, 5, 40, 8]
- Fixed: [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

以上只是 selector 單元測試樣本，並非實際衛星服務或 simulation action trace。

AIM-static、GA-grouping、PPO-multicast 尚未實作；正式輸出標為 NOT_IMPLEMENTED，不生成替代曲線。MIRA / Random / Fixed 的環境 hash 比較已實作，indexed channel 不受額外 policy RNG draw 影響的單元測試已通過；尚未執行三方法真實 trace 的實測比對。

## 驗證與結果

使用工作區 .venv/Scripts/python.exe（Python 3.14.6）執行：

- compileall：全部 Python 語法編譯成功。
- 逐一 import 套件頂層所有 Python modules：成功。
- python -m mira_leo_study --help：成功，新正式 CLI 正確列出。
- python -m mira_leo_study validate-config：VALID，phi=20、B=0.05；reward bounds 尚未實測校準。
- 靜態 AST audit：simulation 的 state 欄位均有初始化或為 checkpoint identity。
- unittest discover：61 tests，57 passed，4 skipped，0 failures、0 errors。

4 個略過項目為：短版實際模擬、checkpoint/resume 實際模擬、SW-UCB 實際模擬、三方法共同環境/功率 pipeline 實際模擬。它們仍保留並可透過 MIRA_LEO_RUN_SIMULATION_TESTS=1 在使用者 PC 執行；這不是因測試失敗而略過。

本環境的系統暫存目錄受 sandbox 限制，測試執行器將 tempfile.tempdir 與 matplotlib cache 改放工作區 .verification_tmp，全部重新通過；沒有降低 assertions。Output packaging 測試只將人工 fixture 寫至臨時資料夾，檢查 CSV、JSON、PNG、PDF、Markdown table，完成後清除。沒有將人工資料放入 outputs。

本機 .venv 未安裝 pyarrow / SciPy；頂層 import 與正式 literal mode 的單元檢查不依賴它們。使用者 PC 正式執行前應依 requirements.txt 安裝依賴；Parquet 需要 pyarrow，選用 projection 需要 SciPy。既有 projection 測試也檢查了缺少 SciPy 時會明確回報 dependency error。

## 模擬後才會有的數值與路徑

以下項目依使用者限制尚未產生，不能宣稱已實測：每 slot candidate/scheduled/rejected/edge counts、normalized transmitted power trace、實際 violations、三 policy final/mean/tail AoI、實際 environment/channel hash、device re-signalling event totals、duration-normalized overhead。

程式將保存這些欄位並在正式運行後檢查，不補入假值。正式 output root 預設為 outputs/fixed_phi20_B005：

- formal_results/convergence_long_run_aoi/：三 policy raw / running AoI CSV、comparison_summary.csv/json、convergence_long_run_aoi.png/pdf、weighted_sum_aoi_raw_trace.png、candidate_vs_scheduled_groups.png。
- formal_results/regrouping_overhead/：regrouping_events.csv、regrouping_overhead_summary.csv、regrouping_overhead_table.md。
- runs/<policy>/：完整 cohort/slot power/action diagnostics、snapshot、summary，以及選用 group trace。

Overhead 保留 total_regrouping_events、eligible_device_service_count、evaluated_duration_s/min、events_per_device_per_min 和 action_change_events。第一個可見 assignment、離開 coverage、handover 與未訂閱裝置不算 regrouping。tau=10 s 一定包含，也支援 list 的額外 sensitivity runs。

實際執行命令與完整參數請見 README.md。可直接執行：

```powershell
python -m mira_leo_study paper-results --outputs .\outputs\fixed_phi20_B005 --seed 0 --resume
```

此命令會真正執行 pilot（若缺 bounds）與三方法模擬，本次未執行。
