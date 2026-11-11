> 歷史紀錄：以下內容屬於 120-UE single-serving 重構前的版本，不能作為新模型的結果或目前 API 規格。最新狀態見 [MODIFICATION_REPORT_120UE_SINGLE_SERVING.md](MODIFICATION_REPORT_120UE_SINGLE_SERVING.md)。

# Diagnostic-only modification report

本次依 `CODEX_MIRA_LEO_POOR_RESULT_DIAGNOSTICS.md` 加入追蹤、CSV、事後分析與繪圖。使用者明確要求不實際模擬，故文件建議的 3,000-slot / 48,000-slot 驗收未執行；沒有重新抽 channel、執行 reward pilot 或更新既有結果。

## 修改檔案

- `diagnostic_trace.py`：互斥 outcome、pre/post AoI attribution、current CSI / elevation、boundary CSI drift、class interleaving、群組 AoI 排名、nullable Parquet 型別。
- `diagnostics.py`：讀取保存結果，分批建 SQLite 索引，精確統計、共同 CSI quintile、CSV、環境 hash / AoI 加總驗證與資料驅動報告。
- `diagnostic_plots.py`：36 張診斷圖；scatter 使用獨立 post-run sampling。
- `algorithms.py`：保存既有 scheduler 計算中的 pre / predicted AoI；decode 記錄第一個 own-rate-failed predecessor。排程、功率與 decode 數學運算保留。
- `regrouping.py`：診斷開啟時保存 boundary gain / rank，checkpoint 可攜帶；沒有更改 immutable partition。
- `simulation.py`：`--save-group-trace` 保存擴充 group trace、cohort power / elevation diagnostics；實際 delivery gate 記錄 handover blackout。Checkpoint schema 5。
- `storage.py`：可顯式指定 Arrow 欄位型別，避免首批全空的 predecessor / class CSI / solver message 在後續批次推斷型別失敗。
- `paper_results.py`、`__main__.py`：新增 `diagnose-results` 與 `paper-results --diagnostics`。
- `tests/test_diagnostics.py`：人工 component / trace pipeline tests；實際 on/off parity test 為 opt-in。
- `README.md`、`IMPLEMENTATION_STATUS.md`：操作方式、輸出、限制與驗證說明。

## 保留的設定

20°、B=0.05、actions=[1,2,4,5,8,10]、M=3、N=40、weights=[.7,.2,.1]、10 ms physical slot、1 秒 regrouping、10 秒 / 10 epoch SW-UCB、xi=1/sqrt(2)、Vc/Vnc=.7/.3、Vphi=.0005、Rtheta=1e-5、Psat=1、iid-per-link-per-slot fading、strict Algorithm 2、TWC/AIM power 與 proven infeasible → no-TX failure 全部保留。`configs/base.yaml` 未修改；reward-bounds context 不變。

## 新增 trace 資料

- Pre AoI：`pre_group_aoi_sum_slots`、成功 / 失敗預測 next AoI、`gamma_slots`、`message_weight` 與各自加權版本。
- Post AoI：sum / mean / max / p95、weighted sum、pre/post change、cohort fraction、rank、top1 / top3 / top5。各 cohort 的群組 weighted post AoI 必須加總等於其 `weighted_aoi_slots`。
- CSI：worst-user amplitude、power gain、gain dB、member amplitude mean/max/std、worst device index / elevation；`gain_db=10log10(max(gain,float.tiny))` 僅用於診斷。
- Elevation：min/mean/max/std、fraction below phi、elevation bin。Edge 仍只用 `max_elevation_deg <20`；nonedge 可能包含很多低仰角成員。
- Power / PHY：target、floor、實際 power、margin / ratio、budget fraction、transmission flags、rate margin、own rate / SIC flags、route / visibility / blackout、first-failed SIC predecessor 的 message/group/gain/power/rate。
- Drift：boundary gain/rank、current gain/rank、gain delta、rank changed、elapsed ms、cohort inversion count。跨 handover 的 current gain 仍與該 epoch 起點 serving satellite 的 boundary CSI 比較。
- Cohort：LP status/message、failure reason、兩類 weakest/strongest gain、全域 CSI class sequence / switch count、cohort 仰角與 edge fraction。

無實際 transmission 時 `rate_margin_bps_hz` 為 NaN；不可把未傳輸的 0 rate 誤當 own-rate failure。Power fraction of satellite 使用既有 Psat 為分母；group / class fraction 使用該 class budget。

## 互斥 outcome 的順序

1. 未排的 nonedge 且 priority<=0：`UNSCHEDULED_NONPOSITIVE_NONEDGE`。
2. 未排的 edge 且 edge 數>U：`UNSCHEDULED_ALGORITHM2_CAP_EDGE`；若違反此 invariant，回報 OTHER。
3. 未排且正 priority 的 nonedge：`UNSCHEDULED_ALGORITHM2_CAP_NONEDGE`。
4. 已排且 power diagnostic 為 transmission failure：`POWER_ALLOCATION_INFEASIBLE_NO_TX`。
5. 真正 transmit 且 own rate 未達門檻：`TRANSMITTED_RATE_BELOW_THRESHOLD`。
6. 真正 transmit、own rate 過但 SIC chain 未過：`SIC_PROPAGATED_FAILURE`。
7. Delivery gates 依序為 `ROUTE_UNREACHABLE`、`NOT_VISIBLE_OR_COVERAGE_FAILURE`、`HANDOVER_BLACKOUT`。
8. `SUCCESS`；其餘為 defensive `OTHER_FAILURE`。

重疊的條件以上述順序決定唯一類別；原始 rate、SIC、route、visibility 等 flags 仍全部保存，沒有隱藏同時發生的問題。OTHER 非零時，診斷 status 為 `WARNING_OTHER_FAILURE`，不宣稱 attribution 完成。

## 圖表與用途

| 檔案 | 回答的問題 |
|---|---|
| 01_edge_nonedge_counts_over_time.png | MIRA 兩類 candidate / scheduled 數如何變化？ |
| 01b_edge_nonedge_counts_policy_comparison.png | 三方法的群組數是否不同？ |
| 02_edge_nonedge_scheduling_rate.png | Edge / nonedge 排程率是否不同？0 denominator 保留 NaN。 |
| 03_group_csi_over_time_edge_vs_nonedge.png | 各類 current gain median、p10/p90 如何變化？ |
| 04_csi_distribution_edge_vs_nonedge.png | 兩類 worst-user gain 的完整 ECDF 是否不同？ |
| 05_elevation_vs_csi_scatter.png | Group max elevation 與 gain 的關係？ |
| 05b_worst_user_elevation_vs_csi.png | Worst-user elevation 是否更能對應 gain？ |
| 06_top_aoi_groups_over_time.png | 哪些 message/group 位置貢獻高 AoI？不存在的位置為 NaN，ID 不代表永久會員。 |
| 07_weighted_aoi_by_message_over_time.png | 高 AoI 主要由哪個 message 貢獻？ |
| 08_group_outcome_counts_over_time.png | 失敗機制如何隨時間變化？ |
| 09_failure_cause_by_policy.png | 占全部 candidate opportunities 的 outcome 比例？ |
| 09b_failure_cause_among_scheduled_groups.png | 只在 scheduled opportunities 中如何分布？ |
| 10_failure_cause_edge_vs_nonedge.png | 失敗集中在 edge 還是 nonedge？ |
| 11_outcome_rate_vs_elevation_bin.png | Elevation bin 與成功、功率無解、AoI 的關係？ |
| 12_aoi_vs_elevation.png | 不同 outcome 的群組 AoI / elevation 散點？ |
| 12b_aoi_by_elevation_bin.png | 各 elevation bin 的 AoI mean / median？ |
| 13_aoi_vs_csi.png | CSI、edge label、outcome 與 AoI 的關係？ |
| 13b_aoi_by_csi_quantile.png | 共同 CSI quintile 的 AoI mean / median / p95？ |
| 14_power_vs_csi.png | 真正 transmit 的弱 CSI 群組分到多少 power？顏色分 class，marker 分 outcome。 |
| 15_rate_margin_vs_csi.png | Own rate 距離門檻多遠，與 CSI 有何關係？ |
| 15b_rate_margin_vs_power.png | Own rate margin 與實際 power 有何關係？ |
| 16_power_infeasibility_heatmap_edge_nonedge_counts.png | 哪些 mixed-class 組合常失敗？Cell 標示樣本數。 |
| 17_power_infeasibility_vs_elevation.png | Cohort elevation / edge fraction 與 power failure 的關係？ |
| 18_power_infeasibility_vs_class_switches.png | 全域 CSI class 交錯與失敗率的關係？ |
| 19_action_conditioned_aoi.png | 各實際 action 時段的 aggregated AoI mean / median / p95？ |
| 20_action_conditioned_failure_rates.png | 各 action 的 no-TX、own-rate、SIC、成功率？ |
| 21_mira_action_and_aoi_timeline.png | Held action 與 outage / AoI spike 是否同時出現？ |
| 22_elevation_action_aoi_timeline.png | 各 cohort 的仰角、action、AoI、class 與 outcome timeline？ |
| 23_csi_drift_within_regrouping_epoch.png | Boundary CSI 隨 epoch 內 elapsed ms 如何變化？ |
| 24_aoi_vs_csi_rank_drift.png | CSI order inversion 較多的 cohort-slot 是否 AoI 較高？ |
| 25_worst_aoi_episode_panels.png | 最差五個 MIRA slots 由哪些群組貢獻？ |
| 26_sic_failure_source_counts.png | 哪些 slot-local predecessor IDs 引發 SIC propagation？ |
| 27_aoi_contribution_by_failure_cause.png | 各 outcome 的 weighted AoI 總貢獻？ |
| 28_aoi_contribution_edge_vs_nonedge.png | 兩類 AoI 總量、每群組 mean / p95？ |
| 29_policy_aoi_conditioned_on_elevation.png | 同 elevation bin 下三方法 AoI 是否不同？ |
| 30_policy_success_conditioned_on_csi.png | 同 pooled CSI quintile 下三方法成功率？ |

## 驗證範圍與尚待使用者執行的驗收

語法、匯入、固定設定、人工 component / trace pipeline、CSV / Arrow nullable 欄位、36 圖及報告輸出均檢查。人工 fixture 僅用於測試，不是論文 simulation result。

驗證結果：42 個 Python 檔案語法正確；101 項測試中 95 passed、6 個實際模擬測試 skipped，無失敗。人工 pipeline 的 36 張圖與所有必要 CSV / report 可生成。另驗證 base config 與已發布快照逐位元一致，先前 85 個完整 / debug 結果檔案的 SHA-256 仍相同；新 CLI 對舊的無 group trace 結果會以明確訊息要求重跑。

實際 diagnostics on/off parity regression 已加入，會比較 action、membership、逐群組 scheduled set / power / success / rate / SINR、AoI、reward 與 environment/channel hash；必須明確設定 `MIRA_LEO_RUN_SIMULATION_TESTS=1` 才執行。本次沒有啟用，所以不能宣稱已實測整段 simulation 數值完全一致。

人工 pipeline 的 OTHER_FAILURE=0；另測試 OTHER 非零時會標示 warning。人工三方法 hash 相同的檢查與 hash 不同時拒絕輸出的檢查均納入；新版真實 run 的 hash、OTHER count、短 / 長 run 統計仍待使用者執行。

本次沒有產生真實 short/full run 的 edge 數、failure 比例或 top-AoI events，也沒有重新判定 MIRA 較差的機制。新 `diagnostic_report.md` 會依使用者跑出的資料描述當次 realization 的 observation / association，不把相關性當成 controlled causal proof。

既有論文結果、GitHub 發布副本與 `configs/base.yaml` 保持原樣。本次只修改使用者指定的工作專案。
