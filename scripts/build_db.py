import argparse
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import numpy as np
import soundfile as sf


DEFAULT_DB_PATH = "datasets/audiomoth.duckdb"
DEFAULT_MANIFEST_PATH = "datasets/datasets_manifest.csv"
AUDIO_EXTENSIONS = (".wav", ".WAV", ".flac", ".FLAC", ".mp3", ".MP3")
BIRDNET_SUFFIXES = (
    ".BirdNET.selection.table.txt",
    ".BirdNET.selection.table.csv",
    ".selection.table.txt",
    ".selection.table.csv",
)
BATDETECT2_SUFFIXES = (".WAV.csv", ".wav.csv", ".WAV.json", ".wav.json")
BIRDA_SUFFIXES = (
    ".BirdNET.results.csv",
    ".BirdNET.csv",
    ".Perch.results.csv",
    ".Perch.csv",
    ".csv",
)
METRIC_PADDING_SECONDS = 3.0
METRIC_NOISE_GUARD_SECONDS = 0.25
METRIC_NOISE_FRAME_SECONDS = 0.05


def clean_id(value):
    return re.sub(r"[^\w.-]+", "_", str(value).strip(), flags=re.UNICODE).strip("_")


def read_table(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


def strip_birdnet_suffix(filename):
    for suffix in BIRDNET_SUFFIXES:
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return Path(filename).stem


def strip_batdetect2_suffix(filename):
    for suffix in BATDETECT2_SUFFIXES:
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return Path(filename).stem


def strip_birda_suffix(filename):
    for suffix in BIRDA_SUFFIXES:
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return Path(filename).stem


def parse_recording_datetime(name):
    candidates = [
        r"(?P<stamp>\d{8}_\d{6})",
        r"(?P<stamp>\d{8}-\d{6})",
        r"(?P<stamp>\d{4}-\d{2}-\d{2}[_-]\d{2}-\d{2}-\d{2})",
    ]
    formats = ["%Y%m%d_%H%M%S", "%Y%m%d-%H%M%S", "%Y-%m-%d_%H-%M-%S", "%Y-%m-%d-%H-%M-%S"]

    for pattern in candidates:
        match = re.search(pattern, name)
        if not match:
            continue
        stamp = match.group("stamp")
        for fmt in formats:
            try:
                return datetime.strptime(stamp, fmt)
            except ValueError:
                pass
    return None


def relative_to_root(path, root):
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def build_audio_index(audio_dir, data_root):
    index = {}
    if not audio_dir.exists():
        return index

    for audio_path in audio_dir.rglob("*"):
        if not audio_path.is_file() or audio_path.suffix not in AUDIO_EXTENSIONS:
            continue

        keys = {
            audio_path.name.lower(),
            audio_path.stem.lower(),
        }
        timestamp = parse_recording_datetime(audio_path.name)
        if timestamp is not None:
            keys.add(timestamp.strftime("%Y%m%d_%H%M%S").lower())

        rel_path = relative_to_root(audio_path, data_root)
        for key in keys:
            index.setdefault(key, rel_path)

    return index


def infer_audio_path(table_path, df, audio_index):
    if "Begin Path" in df.columns:
        values = df["Begin Path"].dropna().astype(str)
        values = values[values.str.strip() != ""]
        if not values.empty:
            begin_path = Path(values.iloc[0])
            for key in (begin_path.name.lower(), begin_path.stem.lower()):
                if key in audio_index:
                    return audio_index[key]
            return begin_path.name

    base_name = strip_birdnet_suffix(table_path.name)
    for key in (base_name.lower(), Path(base_name).stem.lower()):
        if key in audio_index:
            return audio_index[key]

    timestamp = parse_recording_datetime(base_name)
    if timestamp is not None:
        key = timestamp.strftime("%Y%m%d_%H%M%S").lower()
        if key in audio_index:
            return audio_index[key]

    return f"UNMATCHED/{base_name}.WAV"


def normalise_detections(df):
    required = {"Common Name", "Confidence", "Begin Time (s)", "End Time (s)"}
    missing = required - set(df.columns)
    if missing:
        return pd.DataFrame()

    df = df.copy()
    df = df[df["Common Name"].notna()]
    df = df[df["Common Name"].astype(str).str.lower() != "nocall"]
    if df.empty:
        return df

    return pd.DataFrame(
        {
            "CommonName": df["Common Name"].astype(str),
            "Confidence": pd.to_numeric(df["Confidence"], errors="coerce"),
            "BeginTime": pd.to_numeric(df["Begin Time (s)"], errors="coerce"),
            "EndTime": pd.to_numeric(df["End Time (s)"], errors="coerce"),
            "SpeciesCode": df.get("Species Code", pd.Series([None] * len(df))),
            "LowFreqHz": pd.to_numeric(df.get("Low Freq (Hz)", pd.Series([None] * len(df))), errors="coerce"),
            "HighFreqHz": pd.to_numeric(df.get("High Freq (Hz)", pd.Series([None] * len(df))), errors="coerce"),
        }
    ).dropna(subset=["Confidence", "BeginTime", "EndTime"])


def species_code(value):
    text = re.sub(r"[^A-Za-z0-9]+", "", str(value).lower())
    return text[:12] or "unknown"


def normalise_batdetect2_csv(df):
    required = {"start_time", "end_time", "class", "det_prob"}
    missing = required - set(df.columns)
    if missing:
        return pd.DataFrame()

    df = df.copy()
    class_prob = pd.to_numeric(df.get("class_prob", pd.Series([np.nan] * len(df))), errors="coerce")
    det_prob = pd.to_numeric(df["det_prob"], errors="coerce")
    confidence = (det_prob * class_prob).fillna(det_prob)

    return pd.DataFrame(
        {
            "CommonName": df["class"].astype(str),
            "Confidence": confidence,
            "BeginTime": pd.to_numeric(df["start_time"], errors="coerce"),
            "EndTime": pd.to_numeric(df["end_time"], errors="coerce"),
            "SpeciesCode": df["class"].map(species_code),
            "LowFreqHz": pd.to_numeric(df.get("low_freq", pd.Series([None] * len(df))), errors="coerce"),
            "HighFreqHz": pd.to_numeric(df.get("high_freq", pd.Series([None] * len(df))), errors="coerce"),
        }
    ).dropna(subset=["Confidence", "BeginTime", "EndTime"])


def normalise_batdetect2_json(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    annotations = data.get("annotation", [])
    if not annotations:
        return pd.DataFrame()
    return normalise_batdetect2_csv(pd.DataFrame(annotations).rename(columns={"start_time": "start_time"}))


def normalise_birda_csv(df):
    required = {"Start (s)", "End (s)", "Common name", "Confidence"}
    missing = required - set(df.columns)
    if missing:
        return pd.DataFrame()

    result = pd.DataFrame(
        {
            "CommonName": df["Common name"].astype(str),
            "Confidence": pd.to_numeric(df["Confidence"], errors="coerce"),
            "BeginTime": pd.to_numeric(df["Start (s)"], errors="coerce"),
            "EndTime": pd.to_numeric(df["End (s)"], errors="coerce"),
            "SpeciesCode": df.get("Scientific name", df["Common name"]).map(species_code),
            "LowFreqHz": np.nan,
            "HighFreqHz": np.nan,
        }
    )
    return result.dropna(subset=["Confidence", "BeginTime", "EndTime"])


def infer_birda_audio_path(table_path, raw_df, audio_index):
    if "File" in raw_df.columns:
        values = raw_df["File"].dropna().astype(str)
        values = values[values.str.strip() != ""]
        if not values.empty:
            file_path = Path(values.iloc[0])
            keys = [file_path.name.lower(), file_path.stem.lower()]
            for key in keys:
                if key in audio_index:
                    return audio_index[key]
            timestamp = parse_recording_datetime(file_path.name)
            if timestamp is not None:
                key = timestamp.strftime("%Y%m%d_%H%M%S").lower()
                if key in audio_index:
                    return audio_index[key]
            return file_path.name

    base_name = strip_birda_suffix(table_path.name)
    for key in (base_name.lower(), Path(base_name).stem.lower()):
        if key in audio_index:
            return audio_index[key]

    timestamp = parse_recording_datetime(base_name)
    if timestamp is not None:
        key = timestamp.strftime("%Y%m%d_%H%M%S").lower()
        if key in audio_index:
            return audio_index[key]

    return f"UNMATCHED/{base_name}.WAV"


def detection_id(dataset_id, recording_id, detector, row):
    raw = "|".join(
        [
            dataset_id,
            recording_id,
            detector,
            str(row.CommonName),
            f"{row.BeginTime:.3f}",
            f"{row.EndTime:.3f}",
            f"{row.Confidence:.5f}",
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def recording_id(dataset_id, rel_audio_path):
    return hashlib.sha1(f"{dataset_id}|{rel_audio_path}".encode("utf-8")).hexdigest()


def dbfs_from_samples(samples):
    if len(samples) == 0:
        return np.nan
    rms = float(np.sqrt(np.mean(np.square(samples))))
    return 20 * np.log10(max(rms, 1e-9))


def noise_floor_dbfs(samples, sample_rate):
    if len(samples) == 0:
        return np.nan
    frame_size = max(1, int(sample_rate * METRIC_NOISE_FRAME_SECONDS))
    frame_levels = []
    for start in range(0, len(samples), frame_size):
        frame = samples[start : start + frame_size]
        if len(frame) >= max(1, frame_size // 2):
            frame_levels.append(dbfs_from_samples(frame))
    if not frame_levels:
        return dbfs_from_samples(samples)
    return float(np.percentile(frame_levels, 20))


def audio_detection_metrics(audio_path, begin_time, end_time):
    metric_start = max(0.0, begin_time - METRIC_PADDING_SECONDS)
    metric_duration = max(1.0, end_time - begin_time + 2 * METRIC_PADDING_SECONDS)
    with sf.SoundFile(str(audio_path)) as audio_file:
        sr = audio_file.samplerate
        audio_file.seek(max(0, int(metric_start * sr)))
        frames = max(1, int(metric_duration * sr))
        y = audio_file.read(frames=frames, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)

    if len(y) == 0:
        return {
            "mean_dbfs": np.nan,
            "signal_dbfs": np.nan,
            "noise_dbfs": np.nan,
            "snr_db": np.nan,
            "metric_start": metric_start,
            "metric_duration": metric_duration,
        }

    signal_start = max(0, int((begin_time - metric_start) * sr))
    signal_end = min(len(y), max(signal_start + 1, int((end_time - metric_start) * sr)))
    guard = int(METRIC_NOISE_GUARD_SECONDS * sr)
    noise_parts = [y[: max(0, signal_start - guard)], y[min(len(y), signal_end + guard) :]]
    noise = np.concatenate([part for part in noise_parts if len(part)]) if any(len(part) for part in noise_parts) else np.array([])

    mean_dbfs = dbfs_from_samples(y)
    signal_dbfs = dbfs_from_samples(y[signal_start:signal_end])
    background_dbfs = noise_floor_dbfs(noise, sr)
    snr_db = signal_dbfs - background_dbfs if np.isfinite(signal_dbfs) and np.isfinite(background_dbfs) else np.nan
    return {
        "mean_dbfs": mean_dbfs,
        "signal_dbfs": signal_dbfs,
        "noise_dbfs": background_dbfs,
        "snr_db": snr_db,
        "metric_start": metric_start,
        "metric_duration": metric_duration,
    }


def create_schema(con):
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id TEXT PRIMARY KEY,
            dataset_type TEXT,
            location_name TEXT,
            latitude DOUBLE,
            longitude DOUBLE,
            period_label TEXT,
            birdnet_dir TEXT,
            batdetect2_dir TEXT,
            birda_dir TEXT,
            audio_dir TEXT
        )
        """
    )
    dataset_columns = con.execute("PRAGMA table_info('datasets')").df()["name"].tolist()
    if "dataset_name" in dataset_columns and "dataset_type" not in dataset_columns:
        con.execute("ALTER TABLE datasets RENAME COLUMN dataset_name TO dataset_type")
    if "batdetect2_dir" not in dataset_columns:
        con.execute("ALTER TABLE datasets ADD COLUMN batdetect2_dir TEXT")
    if "birda_dir" not in dataset_columns:
        con.execute("ALTER TABLE datasets ADD COLUMN birda_dir TEXT")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS recordings (
            recording_id TEXT PRIMARY KEY,
            dataset_id TEXT,
            file_name TEXT,
            relative_audio_path TEXT,
            recording_datetime TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS detections (
            detection_id TEXT PRIMARY KEY,
            dataset_id TEXT,
            recording_id TEXT,
            CommonName TEXT,
            Confidence DOUBLE,
            BeginPath TEXT,
            BeginTime DOUBLE,
            EndTime DOUBLE,
            DateTime TIMESTAMP,
            Detector TEXT,
            SpeciesCode TEXT,
            LowFreqHz DOUBLE,
            HighFreqHz DOUBLE,
            SourceTable TEXT
        )
        """
    )
    detection_columns = con.execute("PRAGMA table_info('detections')").df()["name"].tolist()
    if "Detector" not in detection_columns:
        con.execute("ALTER TABLE detections ADD COLUMN Detector TEXT DEFAULT 'birdnet'")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS detection_audio_metrics (
            detection_id TEXT PRIMARY KEY,
            mean_dbfs DOUBLE,
            signal_dbfs DOUBLE,
            noise_dbfs DOUBLE,
            snr_db DOUBLE,
            metric_start DOUBLE,
            metric_duration DOUBLE,
            computed_at TIMESTAMP
        )
        """
    )
    metric_columns = con.execute("PRAGMA table_info('detection_audio_metrics')").df()["name"].tolist()
    for column in ("signal_dbfs", "noise_dbfs", "snr_db"):
        if column not in metric_columns:
            con.execute(f"ALTER TABLE detection_audio_metrics ADD COLUMN {column} DOUBLE")
    con.execute("DROP TABLE IF EXISTS recording_audio_metrics")


def delete_datasets(con, dataset_ids):
    if not dataset_ids:
        return
    dataset_df = pd.DataFrame({"dataset_id": dataset_ids})
    con.register("datasets_to_replace", dataset_df)
    con.execute(
        """
        DELETE FROM detection_audio_metrics
        WHERE detection_id IN (
            SELECT detection_id
            FROM detections
            WHERE dataset_id IN (SELECT dataset_id FROM datasets_to_replace)
        )
        """
    )
    con.execute("DELETE FROM detections WHERE dataset_id IN (SELECT dataset_id FROM datasets_to_replace)")
    con.execute("DELETE FROM recordings WHERE dataset_id IN (SELECT dataset_id FROM datasets_to_replace)")
    con.execute("DELETE FROM datasets WHERE dataset_id IN (SELECT dataset_id FROM datasets_to_replace)")
    con.unregister("datasets_to_replace")


def path_value(row, column):
    value = row.get(column)
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    return str(value)


def detector_sources(row, data_root, dataset_dir):
    sources = []
    dataset_type = str(row.get("dataset_type", "")).lower()

    birdnet_value = path_value(row, "birdnet_dir")
    birdnet_dir = None
    if birdnet_value:
        birdnet_dir = (data_root / birdnet_value).resolve()
    elif dataset_type != "bats":
        birdnet_birdfreq = dataset_dir / "birdnet_birdfreq"
        birdnet_dir = birdnet_birdfreq if birdnet_birdfreq.exists() else dataset_dir / "birdnet"
    if birdnet_dir and birdnet_dir.exists():
        sources.append({"detector": "birdnet", "path": birdnet_dir})

    batdetect2_value = path_value(row, "batdetect2_dir")
    batdetect2_dir = None
    if batdetect2_value:
        batdetect2_dir = (data_root / batdetect2_value).resolve()
    elif dataset_type == "bats":
        batdetect2_dir = dataset_dir / "batdetect2_outputs"
    if batdetect2_dir and batdetect2_dir.exists():
        sources.append({"detector": "batdetect2", "path": batdetect2_dir})

    birda_value = path_value(row, "birda_dir")
    birda_dir = None
    if birda_value:
        birda_dir = (data_root / birda_value).resolve()
    else:
        fallback_birda_dir = dataset_dir / "birda"
        if fallback_birda_dir.exists():
            birda_dir = fallback_birda_dir
    if birda_dir and birda_dir.exists():
        sources.append({"detector": "birda", "path": birda_dir})

    return sources


def batdetect_audio_key(path):
    return strip_batdetect2_suffix(path.name)


def import_dataset(row, data_root):
    dataset_id = clean_id(row["dataset_id"])
    audio_dir = (data_root / str(row["audio_dir"])).resolve()
    dataset_dir = audio_dir.parent

    audio_index = build_audio_index(audio_dir, data_root)
    recordings = {}
    detections = []
    skipped = []

    for source in detector_sources(row, data_root, dataset_dir):
        detector = source["detector"]
        source_dir = source["path"]
        if detector == "birdnet":
            source_files = sorted(source_dir.rglob("*.txt")) + sorted(source_dir.rglob("*.csv"))
        elif detector == "batdetect2":
            source_files = sorted(source_dir.rglob("*.csv"))
        else:
            source_files = sorted(source_dir.rglob("*.csv"))

        for table_path in source_files:
            try:
                raw_df = read_table(table_path)
            except Exception as exc:
                skipped.append((str(table_path), f"could not read: {exc}"))
                continue

            if detector == "birdnet":
                df = normalise_detections(raw_df)
                rel_audio_path = infer_audio_path(table_path, raw_df, audio_index)
            elif detector == "batdetect2":
                df = normalise_batdetect2_csv(raw_df)
                audio_key = batdetect_audio_key(table_path)
                rel_audio_path = audio_index.get(audio_key.lower()) or audio_index.get(Path(audio_key).stem.lower())
                if rel_audio_path is None:
                    rel_audio_path = f"UNMATCHED/{audio_key}"
            else:
                df = normalise_birda_csv(raw_df)
                rel_audio_path = infer_birda_audio_path(table_path, raw_df, audio_index)

            if df.empty:
                skipped.append((str(table_path), f"no valid {detector} detections"))
                continue

            rec_id = recording_id(dataset_id, rel_audio_path)
            rec_dt = parse_recording_datetime(Path(rel_audio_path).name) or parse_recording_datetime(table_path.name)

            recordings[rec_id] = {
                "recording_id": rec_id,
                "dataset_id": dataset_id,
                "file_name": Path(rel_audio_path).name,
                "relative_audio_path": rel_audio_path,
                "recording_datetime": rec_dt,
            }

            if rec_dt is None:
                skipped.append((str(table_path), "could not infer recording timestamp"))
                continue

            df["dataset_id"] = dataset_id
            df["recording_id"] = rec_id
            df["BeginPath"] = rel_audio_path
            df["DateTime"] = rec_dt + pd.to_timedelta(df["BeginTime"], unit="s")
            df["Detector"] = detector
            df["SourceTable"] = relative_to_root(table_path, data_root)
            df["detection_id"] = [detection_id(dataset_id, rec_id, detector, item) for item in df.itertuples()]
            detections.append(
                df[
                    [
                        "detection_id",
                        "dataset_id",
                        "recording_id",
                        "CommonName",
                        "Confidence",
                        "BeginPath",
                        "BeginTime",
                        "EndTime",
                        "DateTime",
                        "Detector",
                        "SpeciesCode",
                        "LowFreqHz",
                        "HighFreqHz",
                        "SourceTable",
                    ]
                ]
            )

    dataset = {
        "dataset_id": dataset_id,
        "dataset_type": row.get("dataset_type", "birds"),
        "location_name": row.get("location_name", ""),
        "latitude": pd.to_numeric(row.get("latitude"), errors="coerce"),
        "longitude": pd.to_numeric(row.get("longitude"), errors="coerce"),
        "period_label": row.get("period_label", ""),
        "birdnet_dir": path_value(row, "birdnet_dir") or "",
        "batdetect2_dir": path_value(row, "batdetect2_dir") or "",
        "birda_dir": path_value(row, "birda_dir") or "",
        "audio_dir": str(row["audio_dir"]),
    }

    detections_df = pd.concat(detections, ignore_index=True) if detections else pd.DataFrame()
    recordings_df = pd.DataFrame(recordings.values())
    return dataset, recordings_df, detections_df, skipped


def insert_df(con, table_name, df):
    if df.empty:
        return
    temp_name = f"temp_{table_name}"
    con.register(temp_name, df)
    columns = ", ".join(f'"{column}"' for column in df.columns)
    con.execute(f"INSERT INTO {table_name} ({columns}) SELECT {columns} FROM {temp_name}")
    con.unregister(temp_name)


def insert_missing_df(con, table_name, df, key_column):
    if df.empty:
        return
    temp_name = f"temp_{table_name}"
    con.register(temp_name, df)
    columns = ", ".join(f'"{column}"' for column in df.columns)
    select_columns = ", ".join(f't."{column}"' for column in df.columns)
    con.execute(
        f"""
        INSERT INTO {table_name} ({columns})
        SELECT {select_columns}
        FROM {temp_name} t
        WHERE NOT EXISTS (
            SELECT 1
            FROM {table_name} existing
            WHERE existing."{key_column}" = t."{key_column}"
        )
        """
    )
    con.unregister(temp_name)


def upsert_detection_audio_metrics(con, df):
    if df.empty:
        return
    con.register("temp_detection_audio_metrics", df)
    con.execute(
        """
        DELETE FROM detection_audio_metrics
        WHERE detection_id IN (
            SELECT detection_id FROM temp_detection_audio_metrics
        )
        """
    )
    columns = ", ".join(f'"{column}"' for column in df.columns)
    con.execute(
        f"INSERT INTO detection_audio_metrics ({columns}) "
        f"SELECT {columns} FROM temp_detection_audio_metrics"
    )
    con.unregister("temp_detection_audio_metrics")


def upsert_dataset(con, dataset):
    dataset_df = pd.DataFrame([dataset])
    con.register("temp_dataset_upsert", dataset_df)
    con.execute(
        """
        DELETE FROM datasets
        WHERE dataset_id IN (SELECT dataset_id FROM temp_dataset_upsert)
        """
    )
    columns = ", ".join(f'"{column}"' for column in dataset_df.columns)
    con.execute(f"INSERT INTO datasets ({columns}) SELECT {columns} FROM temp_dataset_upsert")
    con.unregister("temp_dataset_upsert")


def compute_missing_audio_metrics(con, data_root, refresh=False, limit=None):
    if refresh:
        con.execute("DELETE FROM detection_audio_metrics")

    query = """
        SELECT
            d.detection_id,
            d.BeginTime,
            d.EndTime,
            COALESCE(r.relative_audio_path, d.BeginPath) AS relative_audio_path
        FROM detections d
        LEFT JOIN recordings r ON d.recording_id = r.recording_id
        LEFT JOIN detection_audio_metrics m ON d.detection_id = m.detection_id
        WHERE m.detection_id IS NULL
           OR m.signal_dbfs IS NULL
           OR m.noise_dbfs IS NULL
           OR m.snr_db IS NULL
        ORDER BY d.DateTime
    """
    if limit:
        query += f" LIMIT {int(limit)}"

    pending = con.execute(query).df()
    if pending.empty:
        print("Audio metrics: no missing rows")
        return

    rows = []
    skipped = 0
    for index, row in pending.iterrows():
        rel_path = row["relative_audio_path"]
        audio_path = Path(str(rel_path))
        if not audio_path.is_absolute():
            audio_path = data_root / audio_path

        if not audio_path.exists():
            skipped += 1
            continue

        begin_time = float(row["BeginTime"])
        end_time = float(row["EndTime"])
        try:
            metrics = audio_detection_metrics(audio_path, begin_time, end_time)
        except Exception:
            skipped += 1
            continue

        rows.append(
            {
                "detection_id": row["detection_id"],
                **metrics,
                "computed_at": pd.Timestamp.now("UTC").to_pydatetime(),
            }
        )

        if len(rows) % 500 == 0:
            upsert_detection_audio_metrics(con, pd.DataFrame(rows))
            print(f"Audio metrics: computed {index + 1}/{len(pending)}")
            rows = []

    if rows:
        upsert_detection_audio_metrics(con, pd.DataFrame(rows))

    print(f"Audio metrics ready: computed {len(pending) - skipped}, skipped {skipped}")


def main():
    parser = argparse.ArgumentParser(description="Build or update an AudioMoth Dashboard DuckDB database.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH, help="CSV manifest with one row per dataset.")
    parser.add_argument("--data-root", default=".", help="Root folder used for relative manifest paths.")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Output DuckDB database path.")
    parser.add_argument("--replace-db", action="store_true", help="Delete and rebuild the whole database.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append rows without replacing datasets from the manifest. Usually you do not want this.",
    )
    parser.add_argument(
        "--skip-audio-metrics",
        action="store_true",
        help="Skip computing detection-level dBFS, background-noise, and SNR metrics.",
    )
    parser.add_argument(
        "--refresh-audio-metrics",
        action="store_true",
        help="Delete and recompute all detection-level dBFS and SNR metrics.",
    )
    parser.add_argument(
        "--audio-metrics-limit",
        type=int,
        help="Compute at most this many missing audio metrics.",
    )
    parser.add_argument(
        "--audio-metrics-only",
        action="store_true",
        help="Do not import datasets; only create/fill missing audio metrics in the existing database.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = data_root / manifest_path

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = data_root / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if args.replace_db and db_path.exists():
        os.remove(db_path)

    con = duckdb.connect(str(db_path))
    create_schema(con)

    all_skipped = []
    if not args.audio_metrics_only:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        manifest = pd.read_csv(manifest_path)
        required_manifest_cols = {"dataset_id", "birdnet_dir", "audio_dir"}
        missing = required_manifest_cols - set(manifest.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")

        dataset_ids = [clean_id(value) for value in manifest["dataset_id"]]
        if not args.append:
            delete_datasets(con, dataset_ids)

        for _, manifest_row in manifest.iterrows():
            dataset, recordings_df, detections_df, skipped = import_dataset(manifest_row, data_root)

            if args.append:
                upsert_dataset(con, dataset)
                insert_missing_df(con, "recordings", recordings_df, "recording_id")
                insert_missing_df(con, "detections", detections_df, "detection_id")
            else:
                insert_df(con, "datasets", pd.DataFrame([dataset]))
                insert_df(con, "recordings", recordings_df)
                insert_df(con, "detections", detections_df)

            print(
                f"Imported {dataset['dataset_id']}: "
                f"{len(recordings_df)} recordings, {len(detections_df)} detections"
            )
            all_skipped.extend(skipped)

    if not args.skip_audio_metrics:
        compute_missing_audio_metrics(
            con,
            data_root,
            refresh=args.refresh_audio_metrics,
            limit=args.audio_metrics_limit,
        )

    if all_skipped:
        print("\nSkipped files / warnings:")
        for path, reason in all_skipped[:50]:
            print(f"- {path}: {reason}")
        if len(all_skipped) > 50:
            print(f"... plus {len(all_skipped) - 50} more")

    con.close()
    print(f"\nDatabase ready: {db_path}")


if __name__ == "__main__":
    main()
