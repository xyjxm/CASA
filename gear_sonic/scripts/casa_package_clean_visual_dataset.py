from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy clean CASA videos/labels into a download folder and zip it.")
    parser.add_argument("--clean-csv", type=Path, required=True)
    parser.add_argument("--dataset-summary", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--package-name", default="phase2_clean_visual_1000_v2_videos_labels_20260519")
    parser.add_argument("--include-contact-sheets", action="store_true", default=True)
    parser.add_argument("--include-vlm-json", action="store_true", default=True)
    parser.add_argument("--include-source-logs", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.clean_csv.expanduser().resolve().open(newline="", encoding="utf-8")))
    package_dir = args.output_root / args.package_name
    zip_path = args.output_root / f"{args.package_name}.zip"
    if package_dir.exists():
        if not args.force:
            raise FileExistsError(f"{package_dir} exists; pass --force to replace it")
        shutil.rmtree(package_dir)
    if zip_path.exists() and args.force:
        zip_path.unlink()
    package_dir.mkdir(parents=True, exist_ok=True)

    video_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        video_src = resolve(row["video_path"])
        video_name = f"{index:04d}__{row['visual_bucket']}__{row['scene_complexity']}__{row['video_id']}.mp4"
        video_dst = package_dir / video_name
        shutil.copy2(video_src, video_dst)
        out_row = dict(row)
        out_row["packaged_video"] = video_name

        if args.include_contact_sheets and row.get("contact_sheet_path"):
            contact_src = resolve(row["contact_sheet_path"])
            if contact_src.exists():
                contact_dir = package_dir / "contact_sheets"
                contact_dir.mkdir(exist_ok=True)
                contact_name = video_name.replace(".mp4", "_contact.jpg")
                shutil.copy2(contact_src, contact_dir / contact_name)
                out_row["packaged_contact_sheet"] = f"contact_sheets/{contact_name}"

        if args.include_vlm_json and row.get("result_json"):
            json_src = resolve(row["result_json"])
            if json_src.exists():
                json_dir = package_dir / "vlm_json"
                json_dir.mkdir(exist_ok=True)
                json_name = video_name.replace(".mp4", ".json")
                shutil.copy2(json_src, json_dir / json_name)
                out_row["packaged_vlm_json"] = f"vlm_json/{json_name}"

        if args.include_source_logs:
            log_dir = package_dir / "source_logs" / f"{index:04d}__{row['video_id']}"
            log_dir.mkdir(parents=True, exist_ok=True)
            if row.get("source_summary_path"):
                summary_src = resolve(row["source_summary_path"])
                if summary_src.exists():
                    shutil.copy2(summary_src, log_dir / "rollout_summary.json")
                    out_row["packaged_rollout_summary"] = str((log_dir / "rollout_summary.json").relative_to(package_dir))
            if row.get("source_sim_state_csv"):
                sim_state_src = resolve(row["source_sim_state_csv"])
                if sim_state_src.exists():
                    shutil.copy2(sim_state_src, log_dir / "sim_state.csv")
                    out_row["packaged_sim_state_csv"] = str((log_dir / "sim_state.csv").relative_to(package_dir))

        video_rows.append(out_row)

    write_csv(package_dir / "labels_clean1000_v2.csv", video_rows)
    shutil.copy2(args.clean_csv, package_dir / "rollouts_clean_v2.full.csv")
    if args.dataset_summary and args.dataset_summary.exists():
        shutil.copy2(args.dataset_summary, package_dir / "dataset_summary_v2.json")
    readme = {
        "package": args.package_name,
        "rows": len(rows),
        "labels": "labels_clean1000_v2.csv",
        "source_clean_csv": str(args.clean_csv),
    }
    (package_dir / "README.json").write_text(json.dumps(readme, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(package_dir.parent))

    print(
        json.dumps(
            {
                "package_dir": str(package_dir),
                "zip_path": str(zip_path),
                "rows": len(rows),
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


if __name__ == "__main__":
    main()
