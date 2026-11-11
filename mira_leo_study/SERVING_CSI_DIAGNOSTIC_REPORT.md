# 僅服務衛星的 16 分鐘 CSI 診斷

新增 `csi_elevation_serving_satellite.py`，依使用者要求一次輸出仰角 vs CSI、全域 bottom20、全域 bottom30 共三張圖。原本 `csi_elevation_no_interference.py` 與 `csi_elevation_with_interference.py` 保留原樣；後者仍是使用者先前指定的無干擾 bottom30 版本。

## 資料定義

每個時槽仍以原函式產生完整 `(8,120)` 通道與幾何，再由原 `AssociationTracker` 依 coverage-first、hysteresis=1°、TTT=1s 選出一顆全體 UE 共用的服務星。只保留此星仰角 ≥10° 的鏈路。保留原本完整維度的 fading 呼叫，避免因只抽一列通道而改變相同 slot/UE/衛星的 CSI。

沿用 120 個固定 UE、layout seed 42、subscription seed 1042、channel/policy seeds 0。只有新診斷的記憶體內 config 延長到 960 秒；base config 的 480 秒不變。不做 CSI 的干擾調整，不呼叫 multicast scheduler、power allocator 或 AoI simulation。

## 三張圖與可追溯輸出

1. `01_elevation_vs_gain.png`：X 為當時服務星仰角，Y 為原始 power gain 的 dB 值。灰點是最多 100,000 個均勻抽樣觀測，橘線是該選擇條件下全部觀測的期望 power gain（先平均線性 gain 再轉 dB）。
2. `02_bottom20_fraction_by_elevation.png`：以全期所有可見服務星樣本求 exact global q20，統計各仰角區間有多少比例落在該門檻以下。
3. `03_bottom30_fraction_by_elevation.png`：同一批完整服務星樣本的 exact global q30。20% 與 30% 不是每個仰角各自定義的 percentile。

全域 q20/q30 使用磁碟 memmap 副本的 order statistics；原始 gain/bin 成對檔案不重排。兩份 quantile working copies 依序建立並清除。最多 11,520,000 個可見服務星觀測；gain/bin 約 104 MB，計入一份 quantile working copy 的暫存約 196 MB（十進位）。記錄緩衝每 1,000 slots 寫出，避免累積全部 Python dictionaries。

空仰角區間在 CSV 留空、曲線斷開，不填入假的 0%。若量化／重複值在 cutoff 形成 ties，採 `gain_db <= threshold` 全數納入並記錄 realized fraction；不偷偷刪樣本湊滿指定比例。正常連續通道無 threshold ties 時，比例誤差限制為一筆樣本。

`association_trace.csv` 保存全部 slots 的 serving satellite、handover、initial acquisition、reason、fallback、outage receivers 及可見樣本數。`sampled_scatter_points.csv` 可由 observation ID 回推 slot、服務星與 UE。兩份全域門檻、散點 hash、原始全八星通道／幾何 hash、layout hash、seeds 和樣本量均在 `run_summary.json`。

正式輸出為獨立目錄 `mira_leo_study/outputs/csi_elevation_16min_serving_only`；debug 位於其 `debug/slots_N`。只有成功完成統計與三張圖後，暫存結果才改名交付至最終檔名。失敗會保留中間檔並在 stderr 顯示位置。

## 驗證範圍

新增 9 項人工資料測試，涵蓋：只選服務星、排除非服務星、10° 可見度邊界、當前資料對齊、hysteresis/TTT 與換星、空可見樣本、兩個精確分位數、空角度 bins、cutoff ties、三圖與 CSV 封裝、散點身分回推及 CLI／參數驗證。圖表封裝測試以人工通道陣列取代真實 CSI 計算，檔案僅存在 temporary directory，測試後刪除。

本次整體驗證為 161 項測試：155 項通過、6 項需實際模擬的 integration tests 依既有設定未執行，沒有 failure/error。所有 modules 的語法與 imports、1,234 個直接函式呼叫 signatures、新模組 `--help` 與原 `validate-config` 均通過。原兩支 CSI 程式、base config、association/channel/geometry/orbit/scenario 逐檔雜湊與修改前一致。測試紀錄保存於 `C:\Users\lzz\Documents\我的論文code\.verification_serving_csi_20261112\tests.log`。

沒有執行 96,000 slots 的正式診斷，也沒有執行真實的 short/smoke CSI 診斷；因此本次不提供或宣稱新的實驗數值與正式圖檔。使用者可依 README 在自己的 PC 執行。
