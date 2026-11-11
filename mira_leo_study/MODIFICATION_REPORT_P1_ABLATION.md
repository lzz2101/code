# P1 六組實驗程式修改／驗證報告

狀態：程式已實作並完成不執行模擬的驗證；正式六組完成數 **0/6**。這次使用者明確要求只確認語法與可執行性，因此未依 MD 啟動 smoke、reward pilot、debug 1000 slots 或正式 48,000×6 slots。沒有產生或編造研究數值、最佳仰角與正式圖表。

## 與 MD 的對應

- Main φ=15°，MIRA 僅 13/15/20/25°，Random/Fixed 只在 15°，總共六組。
- 新唯一正式 allocator：scheduled geographic edges + 嚴格弱於 strongest scheduled geo-edge 的 non-edges 共用 protected pool。Geographic flag 與 Algorithm-2 schedule 不改。
- Protected/ordinary =70/30；單池 100%；新池 floors、target、全域 SIC、strict pool equality、EIRP 完整檢查；必要時使用 L1 LP。
- 無共同可行解、floors 過大或數值失敗時 actual power=0、success=0，保留 scheduled decisions，按原因記錄 outage。
- 保留原軌道、SR channel、單服務星 association/handover、外部滿載干擾、AoI recursion、480 秒、10 ms、每秒 regrouping、10 秒 warm-up、compact actions、B=.05、seeds 42/1042/0/0。
- P1 以 post-update `C-weighted_AoI_slots` 計算，與 bandit penalty 分開；full/warm-up sums、means、running means 可由 CSV 回算。
- 每 φ 各自校準 reward bounds，baseline 共用 φ15，cache identity 納入新模式、φ、分池 version；schema 升級，拒絕舊 resume。
- 六組比較逐列核對 traces 與六種 environment identities；四 MIRA 與 φ15 三 policies 分開畫圖；只使用實際 CSV。
- Main φ15 提供全部 candidate groups 的每秒地理/member snapshots、每 slot occupancy、群內距離／半徑／凸包、同時同衛星 centroid 距離、三個代表時刻地圖與地理報告。
- Fixed(a=4) 固定每 message 的組數，仍每秒更新 membership。低仰角不等於地面集中，achieved P1 不宣稱 global optimum。

## 驗證結果

原始基線：129 項測試，6 項需模擬的 integration tests 未啟用。修改後：152 項測試，146 項通過、6 項需模擬的 integration tests 未啟用，沒有 failure/error。

新測試覆蓋 strongest-edge promotion、strict ties、unscheduled 排除、單池／空池、70/30 可行解、L1 修復、本質無解、floor 超額、LP numeric error/exception/nonfinite 結果、非法 gain/phi/key/empty members、四角度邊界與每 slot CSI、P1 分離、地理距離、checkpoint schema/CSV 恢復。

人工 CSV packaging 測試以 stub 取代 simulator 與 calibration：驗證六組、各角度 bounds 路徑、CSV/P1/warm-up 回算、19 張圖、geo pipeline、完成 cache 與 CHECKPOINTED 分派。人工檔案只寫 temporary directory，測試後刪除，沒有放入正式 outputs。

另外檢查 Python syntax/imports、直接函式呼叫 signature、`validate-config`、`p1-ablation --help`。實際 simulator loop 的完整／resume equality integration test 依使用者要求未執行，因此不宣稱已驗證真實模擬數值或完整長跑記憶體峰值。

## 保留的 legacy_archived 測試／API

- `tests/test_phi_study.py`：舊 full-action φ/B API 的參數驗證與 stub packaging 回歸；整個 module 標註 legacy_archived，新六組 runner 不呼叫它。
- `tests/test_regrouping_strict.py` 的 `test_legacy_archived_class_budgets_and_continuous_residual_exact_values`、`test_legacy_archived_compact_30_group_floor_invariant`：純舊 helper 的數學 fixture；`test_infeasible_projection_reports_error_without_dropping` 也保留一段純舊 plan fixture，正式 allocate 部分已換新池案例。
- `tests/test_power_outage.py::test_legacy_error_policy_still_returns_an_unhandled_invalid_plan`：明確指定舊模式的純 helper 相容性檢查。
- `tests/test_fixed_results.py` 正式預設與 allocator assertions 已換 φ15／新模式；Algorithm-2 admission 測試保留。
- 六項 opt-in integration tests 是因「使用者要求無須實際模擬」而未執行，不是為規避新模式失敗而新增 skip；原基線同樣預設停用。正式 runner/simulator 拒絕舊 allocator 的新測試有執行且通過。

## 尚待使用者執行

尚未執行 reward pilot、debug、正式六組。因此尚無最佳 φ、實測 P1/AoI/goodput/outage、地理集中性結論、正式比較 CSV/PNG。請依 README 在自己的 PC 啟動 `p1-ablation`。Runner 完成後會產生與實際 traces 一致的 report、environment check、comparison tables、figures 與地理分析。

若出現 `REWARD_RANGE_MISMATCH`，會保留 clipping 診斷並在報告說明，不會偷偷調 B 或將失敗剔除。正式與 debug 輸出、四份 bounds 和六組目錄均獨立。

## Documents 版既有 CSI 程式相容性

這份 Documents 專案與先前修改前的 55 個共用原檔逐一雜湊相同，另有兩個 CSI 診斷程式與一份路徑遷移報告。既有 CSI 程式原樣保留；已檢查其 help 與設定建立：診斷仍為 960 秒，P1 正式設定維持 480 秒。版本 A 的無干擾 bottom20 與歷史檔名版本 B 的無干擾 bottom30 沒有變更。這些檢查沒有執行任何診斷 slot loop。

已檢查 1164 個直接函式呼叫 signatures，並確認所有 modules 的語法及 imports。正式輸出目錄的 manifest 僅為 NOT_RUN/PENDING，沒有複製先前 OneDrive 的模擬結果作為本次資料。

## 原始碼盤點與交付

這次以使用者指定的 Documents 現存專案為基礎，沒有收到或假稱使用 MD 文字中的 ZIP。原始檔 hashes、project version、Python 版本、config_id、四組 seeds 與 MD hash 都存於 `outputs/p1_ablation_13_15_20_25/provenance.json`；來源 metadata 另隨套件保存在 `configs/p1_source_provenance.yaml`。

原始碼備份：`C:\Users\lzz\Documents\我的論文code\.p1_documents_20261108\original`。

新增／修改檔案：

- `__main__.py`
- `algorithms.py`
- `calibration.py`
- `config.py`
- `configs/base.yaml`
- `configs/p1_source_provenance.yaml`
- `diagnostic_trace.py`
- `geo_edge.py`
- `outputs/p1_ablation_13_15_20_25/implementation_verification.json`
- `outputs/p1_ablation_13_15_20_25/manifest.json`
- `outputs/p1_ablation_13_15_20_25/provenance.json`
- `outputs/p1_ablation_13_15_20_25/report.md`
- `p1_ablation.py`
- `p1_metrics.py`
- `paper_results.py`
- `phi_study.py`
- `power_diagnostics.py`
- `protected_power.py`
- `README.md`
- `reward_pilot.py`
- `simulation.py`
- `sources.py`
- `storage.py`
- `tests/test_diagnostics.py`
- `tests/test_fixed_results.py`
- `tests/test_new_power_pools.py`
- `tests/test_p1_ablation.py`
- `tests/test_phi_study.py`
- `tests/test_power_outage.py`
- `tests/test_regrouping_strict.py`
- `tests/test_reward_bounds_io.py`
- `tests/test_single_serving_interference.py`
- `MODIFICATION_REPORT_P1_ABLATION.md`（本報告）
