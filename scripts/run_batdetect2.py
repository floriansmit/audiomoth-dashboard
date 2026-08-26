import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


OUTPUT_SUFFIX = ".selection.table.txt"
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3"}


def species_code(common_name):
    text = common_name.lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text[:12] or "bat"


def confidence(annotation):
    for key in ("det_prob", "class_prob", "confidence", "probability"):
        value = annotation.get(key)
        if value is not None:
            return value
    return None


def batdetect_json_to_selection_table(json_path, output_path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    annotations = data.get("annotation", [])

    rows = []
    for index, annotation in enumerate(annotations, start=1):
        common_name = annotation.get("class") or annotation.get("species") or "Unknown bat"
        start_time = annotation.get("start_time", annotation.get("Begin Time (s)"))
        end_time = annotation.get("end_time", annotation.get("End Time (s)"))

        if start_time is None or end_time is None:
            continue

        rows.append(
            {
                "Selection": index,
                "View": "Spectrogram 1",
                "Channel": 1,
                "Begin Time (s)": start_time,
                "End Time (s)": end_time,
                "Low Freq (Hz)": annotation.get("low_freq"),
                "High Freq (Hz)": annotation.get("high_freq"),
                "Species Code": species_code(common_name),
                "Common Name": common_name,
                "Confidence": confidence(annotation),
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "Selection",
            "View",
            "Channel",
            "Begin Time (s)",
            "End Time (s)",
            "Low Freq (Hz)",
            "High Freq (Hz)",
            "Species Code",
            "Common Name",
            "Confidence",
        ],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    return len(df)


def matching_output_name(json_path):
    name = json_path.name
    if name.lower().endswith(".wav.json"):
        return name[: -len(".wav.json")] + OUTPUT_SUFFIX
    if name.lower().endswith(".json"):
        return name[: -len(".json")] + OUTPUT_SUFFIX
    return json_path.stem + OUTPUT_SUFFIX


def convert_outputs(batdetect_output_dir, birdnet_output_dir):
    converted = 0
    detections = 0

    for json_path in sorted(batdetect_output_dir.rglob("*.json")):
        output_path = birdnet_output_dir / matching_output_name(json_path)
        detections += batdetect_json_to_selection_table(json_path, output_path)
        converted += 1

    return converted, detections


def run_command(command):
    return subprocess.run(command, check=True, capture_output=True, text=True)


def audio_files(audio_dir):
    return sorted(
        path
        for path in audio_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def expected_output_paths(audio_path, batdetect_output_dir):
    return [
        batdetect_output_dir / f"{audio_path.name}.csv",
        batdetect_output_dir / f"{audio_path.name}.json",
    ]


def already_analyzed(audio_path, batdetect_output_dir):
    return any(path.exists() for path in expected_output_paths(audio_path, batdetect_output_dir))


def pending_audio_files(audio_dir, batdetect_output_dir):
    return [path for path in audio_files(audio_dir) if not already_analyzed(path, batdetect_output_dir)]


def link_or_copy_audio(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def prepare_pending_audio_dir(audio_dir, batdetect_output_dir, rerun_existing):
    files = audio_files(audio_dir)
    if rerun_existing:
        print(f"BatDetect2 will analyze all files: {len(files)}")
        return audio_dir, None, files

    pending = pending_audio_files(audio_dir, batdetect_output_dir)
    skipped = len(files) - len(pending)
    print(f"BatDetect2 already analyzed: {skipped}")
    print(f"BatDetect2 pending files: {len(pending)}")

    if not pending:
        return None, None, []

    temp_dir = Path(tempfile.mkdtemp(prefix="audiomoth_batdetect2_pending_"))
    for source in pending:
        relative_path = source.relative_to(audio_dir)
        link_or_copy_audio(source, temp_dir / relative_path)

    return temp_dir, temp_dir, pending


def run_batdetect2(audio_dir, batdetect_output_dir, detection_threshold=None, model=None, rerun_existing=False):
    input_dir, temp_dir, pending = prepare_pending_audio_dir(audio_dir, batdetect_output_dir, rerun_existing)
    if input_dir is None:
        print("No new audio files need BatDetect2 analysis.")
        return

    command = [
        "batdetect2",
        "process",
        "directory",
        str(input_dir),
        str(batdetect_output_dir),
    ]

    if model:
        command.extend(["--model", str(model)])
    if detection_threshold is not None:
        command.extend(["--detection-threshold", str(detection_threshold)])

    try:
        run_command(command)
    except subprocess.CalledProcessError as exc:
        combined_output = "\n".join([exc.stdout or "", exc.stderr or ""])
        if "No such command 'process'" not in combined_output:
            print(combined_output, file=sys.stderr)
            raise

        legacy_threshold = detection_threshold if detection_threshold is not None else 0.3
        legacy_command = [
            "batdetect2",
            "detect",
            str(input_dir),
            str(batdetect_output_dir),
            str(legacy_threshold),
        ]
        print("Detected BatDetect2 1.x CLI. Falling back to: batdetect2 detect ...")
        run_command(legacy_command)
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run BatDetect2 on a WAV folder. The dashboard builder reads BatDetect2 CSV outputs directly."
    )
    parser.add_argument("--audio-dir", required=True, help="Folder containing bat WAV files.")
    parser.add_argument(
        "--batdetect-output-dir",
        help="Folder for raw BatDetect2 JSON/CSV outputs. Defaults to <audio parent>/batdetect2_outputs.",
    )
    parser.add_argument("--detection-threshold", type=float, help="Optional BatDetect2 detection threshold.")
    parser.add_argument("--model", help="Optional custom BatDetect2 model checkpoint.")
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Analyze all audio files, even when matching BatDetect2 CSV/JSON outputs already exist.",
    )
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir).resolve()
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio folder not found: {audio_dir}")

    batdetect_output_dir = Path(args.batdetect_output_dir).resolve() if args.batdetect_output_dir else audio_dir.parent / "batdetect2_outputs"

    try:
        run_batdetect2(audio_dir, batdetect_output_dir, args.detection_threshold, args.model, args.rerun_existing)
    except FileNotFoundError:
        print(
            "Could not find the 'batdetect2' command. Install it first with:\n"
            "  pip install batdetect2\n"
            "or, if using uv:\n"
            "  uv tool install batdetect2",
            file=sys.stderr,
        )
        raise

    print(f"BatDetect2 outputs: {batdetect_output_dir}")
    print("Run build_db.py afterwards; it reads BatDetect2 CSV files directly.")


if __name__ == "__main__":
    main()
