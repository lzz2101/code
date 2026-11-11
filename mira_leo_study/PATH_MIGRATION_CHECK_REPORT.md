# 路徑搬移後的完整性與 PowerShell 核對（2026-10-30）

目前使用位置：`C:\Users\lzz\Documents\我的論文code`。

## 完整性

- 修改說明文件前，`mira_leo_study` 的 72 個檔案與搬移前 OneDrive 最新副本逐一比較，檔案清單及 SHA-256 全部相同，沒有缺失或較舊版本替代。
- 43 個 Python 原始碼檔案均可讀且通過語法檢查；目前診斷模組、36 圖流程、CLI、group trace 與最新 reward-bounds I/O 錯誤處理都保留。
- 歷史合併清單中的 18,208 個檔案在新根目錄皆存在，沒有缺檔。這是存在性／大小盤點，不宣稱 18,208 個檔案全部重新計算 SHA-256。
- 與較早歷史合併清單相比，4 個 Python 檔案（__main__.py、errors.py、paper_results.py、reward_pilot.py）及其 4 個 pyc 大小不同；對應後來新增的讀檔錯誤處理。這些原始碼與搬移前最新副本一致，沒有回退成較早版本。
- 原有 output inventory 的 397 個檔案皆存在，大小一致。
- 已發布快照中的 85 個完整／debug 結果檔案另以 SHA-256 核對，全部一致。
- 本次沒有發現需要補回的檔案。歷史 folder_merge_audit_20261030 的報告仍記錄當時位置，保留其原始稽核內容；它不是目前的執行路徑說明。

## 執行環境

- 專案 `.venv\Scripts\python.exe`：Python 3.14.6，可在新路徑使用，sys.prefix 與套件來源皆指向新路徑。
- 原套件環境仍可載入 numpy 2.5.0、scipy 1.18.1、pyarrow 26.0.0、PyYAML 6.0.3、matplotlib 3.11.0。
- 另有 `C:\venvs\mira-leo-study\Scripts\python.exe`（Python 3.12.10），也通過相同非模擬測試。其套件版本與專案 .venv 不完全相同；沿用搬移前的套件環境請使用專案 .venv 的 Python。
- 兩個環境各自執行 109 項測試：103 passed、6 個實際模擬測試 skipped，無失敗。
- 模組實際來源確認為 `C:\Users\lzz\Documents\我的論文code\mira_leo_study`；沒有載入舊 OneDrive 位置或 github_publish_code 的版本。
- validate-config 回報 VALID；正式 phi、B、action、時間尺度及功率失敗處理設定保持不變。
- 新位置 outputs 的建立、寫入、讀回及移除暫存檔檢查通過，沒有啟動模擬。

## 校準檔與來源

- 新位置 `outputs/diagnostic_phi20_B005/reward_bounds/reward_bounds.json` 與先前完整結果的校準檔均可正常讀取。
- 兩者都通過 seed=0 與目前 config 的 reward-context 驗證；reward identity 均為 `716a1a0690420569a4822c9627855dc855b146dcbfbc8a40cc63cbd46ac50def`。
- 先前 OneDrive 的 OSError 22 沒有在此讀檔檢查中出現。這不等於執行過新的長期模擬；僅確認目前檔案可讀且校準設定相符。

## 本次變更

只更新 README 的現行 PowerShell 路徑、最新驗證數量與使用方式，更新 IMPLEMENTATION_STATUS，新增本核對報告。沒有修改 Python 原始碼、演算法參數、虛擬環境、既有校準值或結果。

沒有執行實際 reward pilot、短期／長期 AoI 模擬，也沒有推送 GitHub。完整新 run 仍由使用者在自己的 PowerShell 執行。
