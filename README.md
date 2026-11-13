# MIRA-LEO 論文程式

目前版本的原始碼位於 [`mira_leo_study/`](mira_leo_study/README.md)，包含 P1 六組仰角消融實驗、地理分析與 CSI／仰角診斷。

## 目前實驗設定

- 8 LEO、120 個靜止 UE、3 messages×40 UE；每個時槽由一顆共同服務衛星負責。
- P1 主設定 φ=15°；MIRA 使用 13°／15°／20°／25°，Random／Fixed(a=4) 僅在 15°，共六組。
- P1 每組 480 秒／48,000 個 10 ms slots；每秒 regrouping；SW-UCB 使用 compact actions。
- 正式功率配置為 CSI-expanded protected pool：scheduled geo-edge 與弱 CSI promoted non-edge 共用 70%，ordinary 共用 30%；單池 100%，並檢查全域 SIC、floors 與功率預算。無解時保留排程並零傳輸。
- P1 指標為 `C - post-update weighted-sum AoI (slots)`；bandit penalty 另行計算。
- 保留原 association／handover、滿載非服務星干擾與固定 seeds。

## 安裝與驗證

在 repository 根目錄使用 Python 3.11 以上。PowerShell：

```powershell
py -3 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r '.\mira_leo_study\requirements.txt'
& '.\.venv\Scripts\python.exe' -m mira_leo_study validate-config
$env:MIRA_LEO_RUN_SIMULATION_TESTS = '0'
& '.\.venv\Scripts\python.exe' -m unittest discover -s '.\mira_leo_study\tests' -v
```

目前共 161 項測試：155 項通過，6 項需實際模擬的 integration tests 預設停用。程式發布驗證不會啟動正式模擬。

## P1 六組實驗

```powershell
& '.\.venv\Scripts\python.exe' -m mira_leo_study p1-ablation --seed 0 --outputs '.\outputs\p1_ablation_13_15_20_25' --resume
```

Runner 會為四個 φ 各自校準 reward bounds，再執行六組。若先短跑，在命令末尾加上 `--debug-slots 1000`；debug 資料與正式結果分開保存。可用 `--help` 查看選項。

## 16 分鐘 CSI／仰角診斷

僅服務衛星版本沿用既有選星／換星規則，只統計當時服務星的可見 UE 鏈路。一次輸出仰角 vs 原始 CSI、全域最弱20%比例、全域最弱30%比例三張圖：

```powershell
& '.\.venv\Scripts\python.exe' -m mira_leo_study.csi_elevation_serving_satellite --outputs '.\outputs\csi_elevation_16min_serving_only'
```

全域門檻與角度比例使用全部服務星觀測；散點圖最多均勻抽樣100,000點。分位數以磁碟 memmap 計算，另輸出各角度統計、服務星紀錄與散點 UE／時間身分。

原本全部8顆衛星的無干擾版本也保留：

```powershell
& '.\.venv\Scripts\python.exe' -m mira_leo_study.csi_elevation_no_interference
& '.\.venv\Scripts\python.exe' -m mira_leo_study.csi_elevation_with_interference
```

第二個歷史檔名目前依使用者設定，仍為「無干擾、global bottom30」版本；第一個為無干擾 bottom20。兩支都統計全部可見衛星–UE鏈路，與 serving-only 版本有不同取樣範圍。

## 原始碼與歷史資料

- [ZIP 版程式與元件完整導讀（φ15／固定 ingress 六組版）](docs/MIRA_LEO_ZIP_CODE_GUIDE_ZH_TW.md)：逐檔用途、資料流程、指標、環境、測試及附圖來源；以原 ZIP 核對，非後續 random-ingress 新版。

- [目前來源檔 SHA-256 清單](CURRENT_SOURCE_MANIFEST.json)
- [P1 修改及驗證報告](mira_leo_study/MODIFICATION_REPORT_P1_ABLATION.md)
- [服務星 CSI 診斷說明](mira_leo_study/SERVING_CSI_DIAGNOSTIC_REPORT.md)

Repository 根目錄舊程式與 `mira_leo_v1/`、`mira_leo_v2/`、`mira_leo_v2a3_verified/` 保留為歷史版本。既有 `outputs/`、`results/` 與 `SNAPSHOT_MANIFEST.json` 屬於較早的研究快照；目前程式的結果請依自己的執行輸出判讀。這次更新只發布目前原始碼、測試與說明，新增的模擬資料、虛擬環境與本機驗證目錄由 `.gitignore` 排除。
