# Implementation Fidelity Table

| method | code status | official code | commit | license | fidelity label | caveat |
|---|---|---|---|---|---|---|
| `clbf_lbac_adapted` | no_confirmed_official_repo | not verified | `n/a` | n/a | `paper_faithful_proxy` | Venue/title/authors verified; no official code URL was verified in this run. |
| `safedpa_adapted` | official_repository_verified_not_integrated | https://github.com/LeCAR-Lab/SafeDPA | `414f57303bcc0681e0301e17422cbbe04b4e6424` | MIT | `paper_faithful_proxy` | Official repo and MIT license verified, but this run does not integrate or execute that code. |
| `pcbf_adapted` | no_confirmed_official_repo | not verified | `n/a` | n/a | `paper_faithful_proxy` | Adapted into CASA as a gate-only probabilistic CBF proxy; no external paper numbers are used. |
| `safer_splat_cbf_adapted` | official_repository_verified_not_integrated | https://github.com/chengine/safer-splat | `adfeba258f34aa949011638b54243cfb595568d2` | MIT | `paper_faithful_proxy` | Official repo verified, but CASA adaptation would use MuJoCo map proxies unless exact code is integrated later. |
| `crc_cbf_adapted` | project_site_claims_code_but_no_repository_url_verified | not verified | `n/a` | n/a | `paper_faithful_proxy` | Project/paper metadata verified; no repository URL was verified, so not exact_code. |
| `mpc_cbf_humanoid_adapted` | no_confirmed_official_repo | not verified | `n/a` | n/a | `lightweight_proxy` | This run implements a gate-only feasibility surrogate, not the full geometry-aware Poisson MPC solver. |
