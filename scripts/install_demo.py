import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_SOURCE = PROJECT_ROOT / "demo_data" / "datasets"
DEMO_DATASETS = {
    "demo_copenhagen_birds": "birds",
    "demo_copenhagen_bats": "bats",
}
COPENHAGEN = {
    "location_name": "Copenhagen",
    "latitude": "55.6761",
    "longitude": "12.5683",
    "period_label": "August 2026",
}


def resolve_from_root(value, data_root):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (data_root / path).resolve()


def copy_demo_datasets(data_root, overwrite=False):
    if not DEMO_SOURCE.exists():
        raise FileNotFoundError(f"Packaged demo data was not found: {DEMO_SOURCE}")

    datasets_dir = data_root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    installed = []
    skipped = []
    for dataset_id in DEMO_DATASETS:
        source = DEMO_SOURCE / dataset_id
        destination = datasets_dir / dataset_id
        if destination.exists() and not overwrite:
            skipped.append(dataset_id)
            continue
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        installed.append(dataset_id)
    return installed, skipped


def update_demo_metadata(manifest_path):
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)

    for row in rows:
        dataset_type = DEMO_DATASETS.get(row.get("dataset_id"))
        if dataset_type is None:
            continue
        row["dataset_type"] = dataset_type
        row.update(COPENHAGEN)

    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(command, cwd):
    print(f"> {' '.join(str(part) for part in command)}")
    subprocess.run(command, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Install the packaged Copenhagen bird and bat demo projects.")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT), help="Folder that contains the datasets directory.")
    parser.add_argument("--db", default="datasets/audiomoth.duckdb", help="DuckDB path, relative to the data root unless absolute.")
    parser.add_argument("--overwrite", action="store_true", help="Replace previously installed demo dataset folders.")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    manifest_path = data_root / "datasets" / "datasets_manifest.csv"
    db_path = resolve_from_root(args.db, data_root)
    installed, skipped = copy_demo_datasets(data_root, overwrite=args.overwrite)

    run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_manifest.py"),
            "--data-root",
            str(data_root),
            "--manifest",
            str(manifest_path),
        ],
        PROJECT_ROOT,
    )
    update_demo_metadata(manifest_path)

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "build_db.py"),
        "--data-root",
        str(data_root),
        "--manifest",
        str(manifest_path),
        "--db",
        str(db_path),
    ]
    for dataset_id in DEMO_DATASETS:
        command.extend(["--dataset-id", dataset_id])
    run(command, PROJECT_ROOT)

    print("\nDemo projects ready:")
    for dataset_id in DEMO_DATASETS:
        print(f"- {dataset_id}")
    if installed:
        print(f"Copied: {', '.join(installed)}")
    if skipped:
        print(f"Kept existing folders: {', '.join(skipped)}")
    print(f"Database: {db_path}")


if __name__ == "__main__":
    main()
