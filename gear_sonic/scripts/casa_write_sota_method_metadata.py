"""Write verified SOTA method metadata for CASA baseline comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


VERIFIED_AT = "2026-06-06"


METHODS: list[dict[str, Any]] = [
    {
        "method_name": "clbf_lbac_adapted",
        "paper_short_name": "CLBF-LBAC-adapted",
        "source_method": "Reinforcement Learning for Safe Robot Control using Control Lyapunov Barrier Functions",
        "authors": ["Desong Du", "Shaohang Han", "Naiming Qi", "Haitham Bou Ammar", "Jun Wang", "Wei Pan"],
        "venue": "IEEE International Conference on Robotics and Automation (ICRA)",
        "year": 2023,
        "pages": "9442-9448",
        "doi": "10.1109/ICRA48891.2023.10160991",
        "arxiv": "2305.09793",
        "original_domain": "2D quadrotor / robot control",
        "original_safety_mechanism": "Control Lyapunov barrier function with Lyapunov barrier actor-critic",
        "official_code_url": None,
        "official_code_commit": None,
        "license": None,
        "code_status": "no_confirmed_official_repo",
        "metadata_sources": [
            "https://research.tudelft.nl/en/publications/reinforcement-learning-for-safe-robot-control-using-control-lyapu",
            "https://arxiv.org/abs/2305.09793",
        ],
        "implemented_in_this_run": False,
        "main_table_fidelity_label": "paper_faithful_proxy",
        "casa_adaptation_route": "learned CLBF-style score gate over CASA skill candidates",
        "notes": "Venue/title/authors verified; no official code URL was verified in this run.",
    },
    {
        "method_name": "safedpa_adapted",
        "paper_short_name": "SafeDPA-adapted",
        "source_method": "Safe Deep Policy Adaptation",
        "authors": ["Wenli Xiao", "Tairan He", "John M. Dolan", "Guanya Shi"],
        "venue": "IEEE International Conference on Robotics and Automation (ICRA)",
        "year": 2024,
        "pages": "17286-17292",
        "doi": None,
        "arxiv": "2310.08602",
        "original_domain": "inverted pendulum, Safety Gym, RC car",
        "original_safety_mechanism": "RL policy adaptation plus CBF safety filter",
        "official_code_url": "https://github.com/LeCAR-Lab/SafeDPA",
        "official_code_commit": "414f57303bcc0681e0301e17422cbbe04b4e6424",
        "license": "MIT",
        "code_status": "official_repository_verified_not_integrated",
        "metadata_sources": [
            "https://www.ri.cmu.edu/publications/safe-deep-policy-adaptation/",
            "https://arxiv.org/abs/2310.08602",
            "https://github.com/LeCAR-Lab/SafeDPA",
        ],
        "implemented_in_this_run": False,
        "main_table_fidelity_label": "paper_faithful_proxy",
        "casa_adaptation_route": "dynamics-adaptation CBF gate/filter over frozen CASA skill commands",
        "notes": "Official repo and MIT license verified, but this run does not integrate or execute that code.",
    },
    {
        "method_name": "pcbf_adapted",
        "paper_short_name": "PCBF-adapted",
        "source_method": (
            "Reinforcement Learning with Probabilistically Safe Control Barrier Functions "
            "for Ramp Merging"
        ),
        "authors": ["Soumith Udatha", "Yiwei Lyu", "John M. Dolan"],
        "venue": "IEEE International Conference on Robotics and Automation (ICRA)",
        "year": 2023,
        "pages": "5625-5630",
        "doi": "10.1109/ICRA48891.2023.10161418",
        "arxiv": "2212.00618",
        "original_domain": "autonomous driving / ramp merging",
        "original_safety_mechanism": "probabilistic CBF constraints under model uncertainty with RL",
        "official_code_url": None,
        "official_code_commit": None,
        "license": None,
        "code_status": "no_confirmed_official_repo",
        "metadata_sources": [
            "https://publications.ri.cmu.edu/reinforcement-learning-with-probabilistically-safe-control-barrier-functions-for-ramp-merging",
            "https://dblp.org/rec/conf/icra/UdathaLD23",
            "https://arxiv.org/abs/2212.00618",
        ],
        "implemented_in_this_run": True,
        "main_table_fidelity_label": "paper_faithful_proxy",
        "casa_adaptation_route": (
            "chance-constrained CBF gate using calibrated uncertainty over CASA skill candidates"
        ),
        "notes": "Adapted into CASA as a gate-only probabilistic CBF proxy; no external paper numbers are used.",
    },
    {
        "method_name": "safer_splat_cbf_adapted",
        "paper_short_name": "SAFER-Splat-CBF-adapted",
        "source_method": (
            "SAFER-Splat: A Control Barrier Function for Safe Navigation with Online "
            "Gaussian Splatting Maps"
        ),
        "authors": [
            "Timothy Chen",
            "Aiden Swann",
            "Javier Yu",
            "Ola Shorinwa",
            "Riku Murai",
            "Monroe Kennedy III",
            "Mac Schwager",
        ],
        "venue": "ICRA accepted per project page / arXiv status",
        "year": 2025,
        "pages": None,
        "doi": None,
        "arxiv": "2409.09868",
        "original_domain": "robot navigation with online Gaussian Splatting maps",
        "original_safety_mechanism": "CBF action filter over online GSplat map hazards",
        "official_code_url": "https://github.com/chengine/safer-splat",
        "official_code_commit": "adfeba258f34aa949011638b54243cfb595568d2",
        "license": "MIT",
        "code_status": "official_repository_verified_not_integrated",
        "metadata_sources": [
            "https://chengine.github.io/safer-splat/",
            "https://arxiv.org/abs/2409.09868",
            "https://github.com/chengine/safer-splat",
        ],
        "implemented_in_this_run": False,
        "main_table_fidelity_label": "paper_faithful_proxy",
        "casa_adaptation_route": "MuJoCo obstacle/user proxy map to Gaussian/ellipsoid CBF gate",
        "notes": (
            "Official repo verified, but CASA adaptation would use MuJoCo map proxies "
            "unless exact code is integrated later."
        ),
    },
    {
        "method_name": "crc_cbf_adapted",
        "paper_short_name": "CRC-CBF-adapted",
        "source_method": "Safe Probabilistic Planning for Human-Robot Interaction using Conformal Risk Control",
        "authors": ["Jake Gonzales", "Kazuki Mizuta", "Karen Leung", "Lillian J. Ratliff"],
        "venue": "IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)",
        "year": 2025,
        "pages": "18676-18683",
        "doi": "10.1109/IROS60139.2025.11247339",
        "arxiv": "2603.10392",
        "original_domain": "human-robot interaction planning",
        "original_safety_mechanism": "conformal risk control over CBF safety margins",
        "official_code_url": None,
        "official_code_commit": None,
        "license": None,
        "code_status": "project_site_claims_code_but_no_repository_url_verified",
        "metadata_sources": [
            "https://jakeagonzales.github.io/crc-cbf-website/",
            "https://arxiv.org/abs/2603.10392",
            "https://eurekamag.com/research/103/587/103587483.php",
        ],
        "implemented_in_this_run": True,
        "main_table_fidelity_label": "paper_faithful_proxy",
        "casa_adaptation_route": "conformal-risk-calibrated CBF gate over CASA skill candidates",
        "notes": "Project/paper metadata verified; no repository URL was verified, so not exact_code.",
    },
    {
        "method_name": "mpc_cbf_humanoid_adapted",
        "paper_short_name": "MPC-CBF-Humanoid-adapted",
        "source_method": (
            "Geometry-Aware Predictive Safety Filters on Humanoids: From Poisson Safety Functions "
            "to CBF Constrained MPC"
        ),
        "authors": [
            "Ryan M. Bena",
            "Gilbert Bahati",
            "Blake Werner",
            "Ryan K. Cosner",
            "Lizhi Yang",
            "Aaron D. Ames",
        ],
        "venue": "IEEE-RAS International Conference on Humanoid Robots",
        "year": 2025,
        "pages": "669-676",
        "doi": "10.1109/Humanoids65713.2025.11203169",
        "arxiv": "2508.11129",
        "original_domain": "humanoid and quadruped predictive safety filtering",
        "original_safety_mechanism": "geometry-aware Poisson safety functions with CBF-constrained nonlinear MPC",
        "official_code_url": None,
        "official_code_commit": None,
        "license": None,
        "code_status": "no_confirmed_official_repo",
        "metadata_sources": [
            "https://arxiv.org/abs/2508.11129",
            "https://eurekamag.com/research/100/409/100409769.php",
            "https://dblp.org/rec/journals/corr/abs-2508-11129",
        ],
        "implemented_in_this_run": True,
        "main_table_fidelity_label": "lightweight_proxy",
        "casa_adaptation_route": "reduced-order skill-level MPC-CBF feasibility gate",
        "notes": (
            "This run implements a gate-only feasibility surrogate, not the full "
            "geometry-aware Poisson MPC solver."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.output_root / "data"
    reports_dir = args.output_root / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "verified_at": VERIFIED_AT,
        "verification_scope": "title, authors, venue/status, code URL, license where public repository was found",
        "paper_number_policy": (
            "No external result numbers are copied; all reported CASA metrics must be rerun metrics."
        ),
        "methods": METHODS,
    }
    (data_dir / "sota_method_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (reports_dir / "sota_method_metadata.md").write_text(_markdown(metadata) + "\n")
    print(json.dumps({"metadata_json": str(data_dir / "sota_method_metadata.json")}, indent=2))


def _markdown(metadata: dict[str, Any]) -> str:
    lines = [
        "# SOTA Method Metadata",
        "",
        f"- verified_at: `{metadata['verified_at']}`",
        f"- scope: {metadata['verification_scope']}",
        f"- number policy: {metadata['paper_number_policy']}",
        "",
        "| method | source method | venue/status | code status | fidelity label | implemented |",
        "|---|---|---|---|---|---:|",
    ]
    for item in metadata["methods"]:
        venue = f"{item['venue']} {item['year']}"
        lines.append(
            f"| `{item['method_name']}` | {item['source_method']} | {venue} | "
            f"{item['code_status']} | `{item['main_table_fidelity_label']}` | "
            f"{int(bool(item['implemented_in_this_run']))} |"
        )
    lines.extend(["", "## Source Links", ""])
    for item in metadata["methods"]:
        lines.append(f"### {item['method_name']}")
        lines.append("")
        lines.append(f"- authors: {', '.join(item['authors'])}")
        if item.get("doi"):
            lines.append(f"- doi: `{item['doi']}`")
        if item.get("arxiv"):
            lines.append(f"- arxiv: `{item['arxiv']}`")
        if item.get("official_code_url"):
            lines.append(f"- official_code_url: {item['official_code_url']}")
            lines.append(f"- official_code_commit: `{item['official_code_commit']}`")
            lines.append(f"- license: `{item['license']}`")
        lines.append(f"- CASA adaptation: {item['casa_adaptation_route']}")
        lines.append(f"- note: {item['notes']}")
        for source in item["metadata_sources"]:
            lines.append(f"- source: {source}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
