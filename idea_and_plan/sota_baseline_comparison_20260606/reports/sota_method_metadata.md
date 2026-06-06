# SOTA Method Metadata

- verified_at: `2026-06-06`
- scope: title, authors, venue/status, code URL, license where public repository was found
- number policy: No external result numbers are copied; all reported CASA metrics must be rerun metrics.

| method | source method | venue/status | code status | fidelity label | implemented |
|---|---|---|---|---|---:|
| `clbf_lbac_adapted` | Reinforcement Learning for Safe Robot Control using Control Lyapunov Barrier Functions | IEEE International Conference on Robotics and Automation (ICRA) 2023 | no_confirmed_official_repo | `paper_faithful_proxy` | 0 |
| `safedpa_adapted` | Safe Deep Policy Adaptation | IEEE International Conference on Robotics and Automation (ICRA) 2024 | official_repository_verified_not_integrated | `paper_faithful_proxy` | 0 |
| `pcbf_adapted` | Reinforcement Learning with Probabilistically Safe Control Barrier Functions for Ramp Merging | IEEE International Conference on Robotics and Automation (ICRA) 2023 | no_confirmed_official_repo | `paper_faithful_proxy` | 1 |
| `safer_splat_cbf_adapted` | SAFER-Splat: A Control Barrier Function for Safe Navigation with Online Gaussian Splatting Maps | ICRA accepted per project page / arXiv status 2025 | official_repository_verified_not_integrated | `paper_faithful_proxy` | 0 |
| `crc_cbf_adapted` | Safe Probabilistic Planning for Human-Robot Interaction using Conformal Risk Control | IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025 | project_site_claims_code_but_no_repository_url_verified | `paper_faithful_proxy` | 1 |
| `mpc_cbf_humanoid_adapted` | Geometry-Aware Predictive Safety Filters on Humanoids: From Poisson Safety Functions to CBF Constrained MPC | IEEE-RAS International Conference on Humanoid Robots 2025 | no_confirmed_official_repo | `lightweight_proxy` | 1 |

## Source Links

### clbf_lbac_adapted

- authors: Desong Du, Shaohang Han, Naiming Qi, Haitham Bou Ammar, Jun Wang, Wei Pan
- doi: `10.1109/ICRA48891.2023.10160991`
- arxiv: `2305.09793`
- CASA adaptation: learned CLBF-style score gate over CASA skill candidates
- note: Venue/title/authors verified; no official code URL was verified in this run.
- source: https://research.tudelft.nl/en/publications/reinforcement-learning-for-safe-robot-control-using-control-lyapu
- source: https://arxiv.org/abs/2305.09793

### safedpa_adapted

- authors: Wenli Xiao, Tairan He, John M. Dolan, Guanya Shi
- arxiv: `2310.08602`
- official_code_url: https://github.com/LeCAR-Lab/SafeDPA
- official_code_commit: `414f57303bcc0681e0301e17422cbbe04b4e6424`
- license: `MIT`
- CASA adaptation: dynamics-adaptation CBF gate/filter over frozen CASA skill commands
- note: Official repo and MIT license verified, but this run does not integrate or execute that code.
- source: https://www.ri.cmu.edu/publications/safe-deep-policy-adaptation/
- source: https://arxiv.org/abs/2310.08602
- source: https://github.com/LeCAR-Lab/SafeDPA

### pcbf_adapted

- authors: Soumith Udatha, Yiwei Lyu, John M. Dolan
- doi: `10.1109/ICRA48891.2023.10161418`
- arxiv: `2212.00618`
- CASA adaptation: chance-constrained CBF gate using calibrated uncertainty over CASA skill candidates
- note: Adapted into CASA as a gate-only probabilistic CBF proxy; no external paper numbers are used.
- source: https://publications.ri.cmu.edu/reinforcement-learning-with-probabilistically-safe-control-barrier-functions-for-ramp-merging
- source: https://dblp.org/rec/conf/icra/UdathaLD23
- source: https://arxiv.org/abs/2212.00618

### safer_splat_cbf_adapted

- authors: Timothy Chen, Aiden Swann, Javier Yu, Ola Shorinwa, Riku Murai, Monroe Kennedy III, Mac Schwager
- arxiv: `2409.09868`
- official_code_url: https://github.com/chengine/safer-splat
- official_code_commit: `adfeba258f34aa949011638b54243cfb595568d2`
- license: `MIT`
- CASA adaptation: MuJoCo obstacle/user proxy map to Gaussian/ellipsoid CBF gate
- note: Official repo verified, but CASA adaptation would use MuJoCo map proxies unless exact code is integrated later.
- source: https://chengine.github.io/safer-splat/
- source: https://arxiv.org/abs/2409.09868
- source: https://github.com/chengine/safer-splat

### crc_cbf_adapted

- authors: Jake Gonzales, Kazuki Mizuta, Karen Leung, Lillian J. Ratliff
- doi: `10.1109/IROS60139.2025.11247339`
- arxiv: `2603.10392`
- CASA adaptation: conformal-risk-calibrated CBF gate over CASA skill candidates
- note: Project/paper metadata verified; no repository URL was verified, so not exact_code.
- source: https://jakeagonzales.github.io/crc-cbf-website/
- source: https://arxiv.org/abs/2603.10392
- source: https://eurekamag.com/research/103/587/103587483.php

### mpc_cbf_humanoid_adapted

- authors: Ryan M. Bena, Gilbert Bahati, Blake Werner, Ryan K. Cosner, Lizhi Yang, Aaron D. Ames
- doi: `10.1109/Humanoids65713.2025.11203169`
- arxiv: `2508.11129`
- CASA adaptation: reduced-order skill-level MPC-CBF feasibility gate
- note: This run implements a gate-only feasibility surrogate, not the full geometry-aware Poisson MPC solver.
- source: https://arxiv.org/abs/2508.11129
- source: https://eurekamag.com/research/100/409/100409769.php
- source: https://dblp.org/rec/journals/corr/abs-2508-11129
