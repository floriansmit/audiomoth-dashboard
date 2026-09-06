import argparse
import csv
import re
from datetime import datetime
from pathlib import Path


MANIFEST_COLUMNS = [
    "dataset_id",
    "dataset_type",
    "location_name",
    "latitude",
    "longitude",
    "period_label",
    "birdnet_dir",
    "batdetect2_dir",
    "birda_dir",
    "audio_dir",
]

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3"}
BIRDNET_SUFFIXES = (
    ".BirdNET.selection.table.txt",
    ".BirdNET.selection.table.csv",
    ".selection.table.txt",
    ".selection.table.csv",
)


def parse_recording_datetime(name):
    candidates = [
        (r"\d{8}_\d{6}", "%Y%m%d_%H%M%S"),
        (r"\d{8}-\d{6}", "%Y%m%d-%H%M%S"),
        (r"\d{4}-\d{2}-\d{2}[_-]\d{2}-\d{2}-\d{2}", None),
        (r"\d{8}", "%Y%m%d"),
    ]

    for pattern, date_format in candidates:
        match = re.search(pattern, name)
        if not match:
            continue

        stamp = match.group(0)
        formats = [date_format] if date_format else ["%Y-%m-%d_%H-%M-%S", "%Y-%m-%d-%H-%M-%S"]
        for fmt in formats:
            try:
                return datetime.strptime(stamp, fmt)
            except ValueError:
                pass

    return None


def dataset_type_from_folder(folder_name):
    return "bats" if "bat" in folder_name.lower() else "birds"


def split_dataset_ids(folder_name):
    match = re.match(r"^(?P<number>\d+)(?P<rest>[_\-\s].*)$", folder_name)
    if not match:
        return f"{folder_name}_birds", f"{folder_name}_bats"

    number = match.group("number")
    rest = match.group("rest")
    bird_rest = re.sub("bat", "Bird", rest, flags=re.IGNORECASE)
    bat_rest = re.sub("bird", "Bat", rest, flags=re.IGNORECASE)
    return f"{number}a{bird_rest}", f"{number}b{bat_rest}"


def period_label_from_audio(audio_dir):
    if not audio_dir.exists():
        return ""

    dated_files = []
    for path in audio_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        recording_dt = parse_recording_datetime(path.name)
        if recording_dt is not None:
            dated_files.append(recording_dt)

    if not dated_files:
        return ""

    first_recording = min(dated_files)
    return first_recording.strftime("%B %Y")


def relative_path(path, root):
    return path.resolve().relative_to(root.resolve()).as_posix()


def find_audio_dir(dataset_dir):
    preferred_names = ("audio", "wav", "wavs", "recordings", "audiomoth")
    child_dirs = [path for path in dataset_dir.iterdir() if path.is_dir()]

    for preferred in preferred_names:
        for child in child_dirs:
            if child.name.lower() == preferred:
                return child

    scored = []
    for child in child_dirs:
        count = sum(1 for path in child.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS)
        if count:
            scored.append((count, child))

    if scored:
        return max(scored, key=lambda item: item[0])[1]

    return dataset_dir


def find_birdnet_dir(dataset_dir):
    child_dirs = [path for path in dataset_dir.iterdir() if path.is_dir()]

    for child in child_dirs:
        if child.name.lower() == "birdnet_birdfreq":
            return child

    for child in child_dirs:
        if "birdnet" in child.name.lower() and child.name.lower() != "birdnet":
            return child

    for child in child_dirs:
        if child.name.lower() == "birdnet":
            return child

    scored = []
    for child in child_dirs:
        count = sum(1 for path in child.rglob("*") if path.is_file() and is_birdnet_table(path))
        if count:
            scored.append((count, child))

    if scored:
        return max(scored, key=lambda item: item[0])[1]

    root_tables = [path for path in dataset_dir.iterdir() if path.is_file() and is_birdnet_table(path)]
    if root_tables:
        return dataset_dir

    return None


def find_batdetect2_dir(dataset_dir):
    child_dirs = [path for path in dataset_dir.iterdir() if path.is_dir()]

    for child in child_dirs:
        if child.name.lower() in {"batdetect2", "batdetect2_outputs"}:
            return child

    for child in child_dirs:
        if "batdetect" in child.name.lower():
            return child

    return None


def find_birda_dir(dataset_dir):
    child_dirs = [path for path in dataset_dir.iterdir() if path.is_dir()]

    for child in child_dirs:
        if child.name.lower() == "birda":
            return child

    for child in child_dirs:
        if "birda" in child.name.lower():
            return child

    return None


def is_birdnet_table(path):
    return any(path.name.endswith(suffix) for suffix in BIRDNET_SUFFIXES)


def read_existing_manifest(manifest_path):
    if not manifest_path.exists():
        return {}

    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = {}
        for row in reader:
            if not row.get("dataset_id"):
                continue
            if not row.get("dataset_type") and row.get("dataset_name"):
                row["dataset_type"] = dataset_type_from_folder(row["dataset_id"])
            rows[row["dataset_id"]] = row
        return rows


def write_manifest(manifest_path, rows):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def build_manifest_rows(data_root, datasets_dir, existing_rows):
    rows = []

    for dataset_dir in sorted(path for path in datasets_dir.iterdir() if path.is_dir()):
        base_dataset_id = dataset_dir.name
        existing_base = existing_rows.get(base_dataset_id, {})
        audio_dir = find_audio_dir(dataset_dir)
        birdnet_dir = find_birdnet_dir(dataset_dir)
        batdetect2_dir = find_batdetect2_dir(dataset_dir)
        birda_dir = find_birda_dir(dataset_dir)

        if birdnet_dir and batdetect2_dir:
            bird_dataset_id, bat_dataset_id = split_dataset_ids(base_dataset_id)
            auto_rows = [
                {
                    "dataset_id": bird_dataset_id,
                    "dataset_type": "birds",
                    "location_name": "",
                    "latitude": "",
                    "longitude": "",
                    "period_label": period_label_from_audio(audio_dir),
                    "birdnet_dir": relative_path(birdnet_dir, data_root),
                    "batdetect2_dir": "",
                    "birda_dir": relative_path(birda_dir, data_root) if birda_dir else "",
                    "audio_dir": relative_path(audio_dir, data_root),
                },
                {
                    "dataset_id": bat_dataset_id,
                    "dataset_type": "bats",
                    "location_name": "",
                    "latitude": "",
                    "longitude": "",
                    "period_label": period_label_from_audio(audio_dir),
                    "birdnet_dir": "",
                    "batdetect2_dir": relative_path(batdetect2_dir, data_root),
                    "birda_dir": relative_path(birda_dir, data_root) if birda_dir else "",
                    "audio_dir": relative_path(audio_dir, data_root),
                },
            ]
        else:
            auto_rows = [{
            "dataset_id": base_dataset_id,
            "dataset_type": dataset_type_from_folder(dataset_dir.name),
            "location_name": "",
            "latitude": "",
            "longitude": "",
            "period_label": period_label_from_audio(audio_dir),
            "birdnet_dir": relative_path(birdnet_dir, data_root) if birdnet_dir else "",
            "batdetect2_dir": relative_path(batdetect2_dir, data_root) if batdetect2_dir else "",
            "birda_dir": relative_path(birda_dir, data_root) if birda_dir else "",
            "audio_dir": relative_path(audio_dir, data_root),
            }]

        for auto_row in auto_rows:
            existing = existing_rows.get(auto_row["dataset_id"], existing_base)
            row = {}
            for column in MANIFEST_COLUMNS:
                if column in {"location_name", "latitude", "longitude"}:
                    row[column] = existing.get(column, auto_row[column])
                elif column == "dataset_type":
                    row[column] = existing_rows.get(auto_row["dataset_id"], {}).get(column, auto_row[column])
                else:
                    row[column] = auto_row[column]
            rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate or update an AudioMoth Dashboard datasets manifest.")
    parser.add_argument("--data-root", default=".", help="External data root that contains the datasets folder.")
    parser.add_argument("--datasets-dir", default="datasets", help="Datasets folder, relative to data root unless absolute.")
    parser.add_argument("--manifest", default="datasets/datasets_manifest.csv", help="Manifest path, relative to data root unless absolute.")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    datasets_dir = Path(args.datasets_dir)
    manifest_path = Path(args.manifest)

    if not datasets_dir.is_absolute():
        datasets_dir = data_root / datasets_dir
    if not manifest_path.is_absolute():
        manifest_path = data_root / manifest_path

    datasets_dir.mkdir(parents=True, exist_ok=True)

    existing_rows = read_existing_manifest(manifest_path)
    rows = build_manifest_rows(data_root, datasets_dir, existing_rows)
    write_manifest(manifest_path, rows)

    print(f"Updated manifest: {manifest_path}")
    print(f"Datasets found: {len(rows)}")
    for row in rows:
        print(
            f"- {row['dataset_id']}: birdnet={row['birdnet_dir']} "
            f"batdetect2={row['batdetect2_dir']} birda={row['birda_dir']} audio={row['audio_dir']}"
        )
    print("Manual fields preserved when dataset_id matched: dataset_type, location_name, latitude, longitude")


if __name__ == "__main__":
    main()
