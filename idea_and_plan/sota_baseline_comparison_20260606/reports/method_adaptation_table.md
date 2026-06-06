# Method Adaptation Table

| method | original domain | safety mechanism | CASA adaptation | main-table fidelity | implemented |
|---|---|---|---|---|---:|
| `clbf_lbac_adapted` | 2D quadrotor / robot control | Control Lyapunov barrier function with Lyapunov barrier actor-critic | learned CLBF-style score gate over CASA skill candidates | `paper_faithful_proxy` | 0 |
| `safedpa_adapted` | inverted pendulum, Safety Gym, RC car | RL policy adaptation plus CBF safety filter | dynamics-adaptation CBF gate/filter over frozen CASA skill commands | `paper_faithful_proxy` | 0 |
| `pcbf_adapted` | autonomous driving / ramp merging | probabilistic CBF constraints under model uncertainty with RL | chance-constrained CBF gate using calibrated uncertainty over CASA skill candidates | `paper_faithful_proxy` | 1 |
| `safer_splat_cbf_adapted` | robot navigation with online Gaussian Splatting maps | CBF action filter over online GSplat map hazards | MuJoCo obstacle/user proxy map to Gaussian/ellipsoid CBF gate | `paper_faithful_proxy` | 0 |
| `crc_cbf_adapted` | human-robot interaction planning | conformal risk control over CBF safety margins | conformal-risk-calibrated CBF gate over CASA skill candidates | `paper_faithful_proxy` | 1 |
| `mpc_cbf_humanoid_adapted` | humanoid and quadruped predictive safety filtering | geometry-aware Poisson safety functions with CBF-constrained nonlinear MPC | reduced-order skill-level MPC-CBF feasibility gate | `lightweight_proxy` | 1 |

Main-table wording must state that these are adapted implementations rerun in CASA/SONIC Phase5.
