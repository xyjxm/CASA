# VLN DMPS MPC-CBF Progress Replan 20260611

This lightweight artifact records the locked online evaluation for `vln_dmps_mpc_cbf_progress` / `vln_ppsr`. Large videos, frames, full logs, and large CSV/JSONL files remain outside git under the run directory.

- Final status: `PASS_IMPLEMENTED_BUT_NO_IMPROVEMENT`
- Run dir: `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246`
- Report: `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/dmps_mpc_cbf_progress_report.md`
- Videos: `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos`
- Frames: `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/frames`
- Real NaVid/Uni-NaVid used: `True`
- Policy backend: `real_navid_visual_adapter`
- Methods: `vln_only, vln_casa_reject_only, vln_casa_replan, vln_dmps_mpc_cbf_progress`
- Locked episodes per method: `20`

## Main Result

`vln_dmps_mpc_cbf_progress` restored safe success to `0.30` while keeping unsafe violation rate at `0.00`, but strict pass was not claimed because positive post-reject progress did not improve over naive replan.

See `result_summary.json`, `method_summary.csv`, and `dmps_mpc_cbf_progress_report.md` for details.
