> 歷史紀錄：以下內容屬於 120-UE single-serving 重構前的版本，不能作為新模型的結果或目前 API 規格。最新狀態見 [MODIFICATION_REPORT_120UE_SINGLE_SERVING.md](MODIFICATION_REPORT_120UE_SINGLE_SERVING.md)。

# 現行設定：20 度 + 功率無解視為更新失敗

依使用者最新要求，正式 phi_theta_deg 已從 30.0 改回 20.0，infeasible_power_handling 保持 transmission_failure。

## 行為

- LP 明確回報 infeasible（status=2）且無 floor/configuration 錯誤：該衛星本 slot 不發射，actual power=0，所有已排程更新失敗，AoI/reward 包含這次失敗。
- Algorithm-2 scheduled flags 保留，removed_groups=[]；其他衛星照常傳輸，干擾使用實際零功率。
- 數值問題、floor/configuration 錯誤、schedule mutation 等仍屬錯誤，不假裝成傳輸失敗。
- Pilot / MIRA / Random / Fixed 使用相同規則。
- Strict Algorithm 2、compact actions、1 秒 frozen membership、current-slot CSI、10 ms slot、10 秒 UCB window、B=0.05 與其餘 power/physical 參數均保持。

## 修改檔案

- configs/base.yaml / config.py：正式 20 度、experiment identity、validator。
- reward_pilot.py：context 的 protocol label 改為角度中立名稱；實際 phi 仍納入 hash，30 度 artifact 不能混用。
- paper_results.py：report 使用正式參數常數產生文字。
- __main__.py / sources.py：預設路徑與設定來源改回 20 度。
- power_diagnostics.py：預設診斷路徑由目前 phi / seed 產生，避免檔名仍錯寫 30 度。
- tests/test_fixed_results.py、test_regrouping_strict.py、test_power_outage.py：更新正式 fixture；確認 30 度設定/bounds 被拒絕，保留 zero-power failure 測試。
- README.md / IMPLEMENTATION_STATUS.md：更新目前設定。30 度報告與資料保留為歷史紀錄。

## 驗證

compileall、全部頂層 modules import、validate-config 均通過。Unittest discovery 共 87 tests：82 passed、5 simulation integration tests skipped、0 failures/errors。沒有執行 reward pilot、實際 simulation 或重新做 power scan。

先前 20 / 30 度的 power-only 診斷保留；本次不將其當作實際 outage frequency，也沒有生成新的模擬數值。

Checkpoint schema 保持 4；config identity 包含 phi，因此不會接續 30 度 checkpoint。舊 30 度及 fail-stop reward bounds 不相容，需在現行設定下重新 pilot。

## 執行

新預設目錄：outputs/fixed_phi20_B005_rg1s_compact_outage。

```powershell
.\.venv\Scripts\python.exe -m mira_leo_study paper-results --slots 1000 --seed 0 --outputs .\outputs\fixed_phi20_B005_rg1s_compact_outage
```

沒有相符 bounds 時會先自動重新校準。完整模擬移除 --slots 1000；本次沒有執行上述命令。
