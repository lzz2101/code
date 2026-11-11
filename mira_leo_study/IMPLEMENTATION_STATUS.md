> 歷史紀錄：以下內容屬於 120-UE single-serving 重構前的版本，不能作為新模型的結果或目前 API 規格。最新狀態見 [MODIFICATION_REPORT_120UE_SINGLE_SERVING.md](MODIFICATION_REPORT_120UE_SINGLE_SERVING.md)。

# Current Implementation Status (2026-10-30)

- Fixed phi=20 degrees, B=.05, compact actions=[1,2,4,5,8,10], Fixed a=4, 1-second regrouping, 10-ms physical slots, SW-UCB 10-second / 10-epoch window.
- Strict Algorithm 2 and TWC/AIM staircase plus global-SIC power-only repair remain. No schedule drops, budget relaxation or reward redesign.
- Proven LP status=2 with no floor/configuration error means zero actual transmission power for that serving satellite-slot. All planned updates fail and remain in AoI/reward statistics. Other satellites continue normally.
- Numerical / configuration / schedule-mutation errors remain errors. Power outages are separate from transmitted PHY rate and propagated SIC failures.
- Added diagnostic-only group AoI / CSI / elevation / power / exclusive outcome / boundary-drift traces and the first own-rate-failed SIC predecessor.
- Added diagnose-results (saved traces only) and paper-results --diagnostics (save traces, then analyze).
- Full-data chunked CSV / Parquet + disk-backed SQLite analysis produces conditional summaries, worst-event CSVs, a data-driven report and 36 plots. Scatter sampling is independent and post-run.
- Missing/old/incomplete traces request a new user-run comparison; analysis never starts a simulator or fabricates missing fields.
- Checkpoint schema=5; the reward-bounds protocol / physical model parameters are unchanged. Old interrupted schema-4 checkpoints are rejected.
- Verification: 101 tests, 95 passed, 6 actual simulation tests skipped; 42 Python source files parse successfully. Artificial trace tests cover CSVs, Arrow types, all 36 plots and the report. Real on/off simulation parity is opt-in and was not run.
- This modification did not execute a real reward pilot, short run, full run or power scan. Existing result files and the published GitHub checkout were preserved.
- Details and PowerShell commands: README.md and DIAGNOSTIC_MODIFICATION_REPORT.md.

## Historical power-only audits

- Seed 0, 90 snapshot times x 2 cohorts x 6 actions=1080 configurations, assuming all candidates scheduled; these are not policy failure frequencies.
- At 20 degrees: 8 LP-infeasible configurations. At 30 degrees: 170 LP-infeasible, 910 feasible, 0 other invalid; all infeasible cases were handled as zero-power failed transmissions.
- Current formal setting remains 20 degrees; historical 30-degree bounds are incompatible.
- AIM-static / GA-grouping / PPO-multicast remain NOT_IMPLEMENTED.
