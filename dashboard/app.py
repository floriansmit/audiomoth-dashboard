import base64
import hashlib
import os
import html
import io
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import math
import queue
import re
import threading
import time
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlparse

import duckdb
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MatplotlibPath
import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    truststore = None
import requests
import soundfile as sf
import streamlit as st
from bs4 import BeautifulSoup
try:
    import folium
    from streamlit_folium import st_folium
except ImportError:
    folium = None
    st_folium = None


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent if APP_DIR.name.lower() == "dashboard" else APP_DIR
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DEFAULT_DB_PATH = PROJECT_ROOT / "datasets" / "audiomoth.duckdb"
LOGO_PATH = APP_DIR / "assets" / "audiomoth-bird-bat-logo-sidebar.png"
APP_TITLE = "AudioMoth Dashboard"
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3"}
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_HEADERS = {"User-Agent": "AudioMoth-Dashboard/1.0 (local Streamlit app)"}
WIKIPEDIA_CACHE_DIR = PROJECT_ROOT / "cache" / "wikipedia"
WIKIPEDIA_CACHE_VERSION = 1
DEMO_DATA_DIR = PROJECT_ROOT / "demo_data" / "datasets"
NOMINATIM_API = "https://nominatim.openstreetmap.org/search"
WIKIPEDIA_NAME_ALIASES = {
    "blackbird": "Common blackbird",
    "goldfinch": "European goldfinch",
    "house sparrow": "House sparrow",
    "jay": "Eurasian jay",
    "magpie": "Eurasian magpie",
    "robin": "European robin",
    "tree sparrow": "Eurasian tree sparrow",
    "wren": "Eurasian wren",
}
WIKIPEDIA_DETAIL_SECTIONS = {
    "Description": ("description", "appearance", "characteristics"),
    "Habitat and distribution": ("habitat", "distribution", "range"),
    "Diet and foraging": ("diet", "feeding", "food", "foraging", "ecology"),
    "Lifespan": ("lifespan", "longevity", "life history"),
}


def connect_database(db_path):
    if not Path(db_path).exists():
        st.error(f"Database not found: {db_path}")
        st.stop()
    return duckdb.connect(db_path, read_only=True)


def resolve_db_path(db_path, data_root):
    resolved = Path(db_path).expanduser()
    if not resolved.is_absolute():
        resolved = data_root / resolved
    return resolved.resolve()


def ensure_local_data_folders(data_root):
    datasets_dir = data_root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    return datasets_dir


def normalize_data_root(path):
    resolved = Path(path).expanduser().resolve()
    if resolved.name.lower() == "datasets":
        return resolved.parent
    return resolved


def render_first_run_page(data_root, db_path):
    datasets_dir = ensure_local_data_folders(data_root)
    st.title(APP_TITLE)
    st.info("No dashboard database was found yet. Open Processing to create or import a dataset, run analyzers, then build the database.")
    st.markdown(
        f"""
        **Current paths**

        - Data root: `{data_root}`
        - Datasets folder: `{datasets_dir}`
        - Database: `{db_path}`

        **First run**

        1. Choose `Processing` in the sidebar.
        2. Create or select a dataset folder.
        3. Copy audio into that dataset's `audio` folder, or point the app at an existing audio folder.
        4. Run BirdNET Analyzer and/or BatDetect2.
        5. Press `Regenerate manifest`.
        6. Press `Append to database` for new datasets, or `Rebuild listed datasets` after re-running analyzer outputs.
        """
    )
    st.caption("The manifest is generated from dataset folders. It is not required before the first processing run.")
    if DEMO_DATA_DIR.exists():
        st.divider()
        st.subheader("Try the included demo")
        st.write(
            "Install two small Copenhagen projects containing three bird recordings and three ultrasonic bat recordings. "
            "Analyzer results are included, so the dashboard, galleries, playback, and spectrograms work immediately."
        )
        if st.button("Install demo projects", type="primary"):
            command = [
                sys.executable,
                str(SCRIPTS_DIR / "install_demo.py"),
                "--data-root",
                str(data_root),
                "--db",
                str(db_path),
            ]
            with st.spinner("Installing demo recordings and building the database..."):
                result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Demo projects installed. Opening the dashboard...")
                st.cache_data.clear()
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("The demo installation did not complete.")
                st.code((result.stdout + "\n" + result.stderr).strip())


def has_platform_schema(con):
    tables = con.execute("SHOW TABLES").df()["name"].tolist()
    return {"datasets", "recordings", "detections"}.issubset(set(tables))


def load_platform_datasets(con):
    datasets = con.execute("SELECT * FROM datasets").df()
    if "dataset_type" not in datasets.columns:
        datasets["dataset_type"] = np.where(
            datasets["dataset_id"].str.contains("bat", case=False, na=False),
            "bats",
            "birds",
        )
    return datasets


def load_platform_detections(con, detector=None, dataset_ids=None):
    dataset_columns = con.execute("PRAGMA table_info('datasets')").df()["name"].tolist()
    detection_columns = con.execute("PRAGMA table_info('detections')").df()["name"].tolist()
    tables = con.execute("SHOW TABLES").df()["name"].tolist()
    has_audio_metrics = "detection_audio_metrics" in tables
    metric_columns = (
        con.execute("PRAGMA table_info('detection_audio_metrics')").df()["name"].tolist()
        if has_audio_metrics
        else []
    )

    dataset_type_sql = (
        "ds.dataset_type"
        if "dataset_type" in dataset_columns
        else "CASE WHEN lower(ds.dataset_id) LIKE '%bat%' THEN 'bats' ELSE 'birds' END AS dataset_type"
    )
    metric_fields = [
        ("mean_dbfs", "MeanDbfs"),
        ("signal_dbfs", "SignalDbfs"),
        ("noise_dbfs", "NoiseDbfs"),
        ("snr_db", "SnrDb"),
    ]
    metrics_select_sql = ",\n                ".join(
        f"m.{source} AS {alias}" if source in metric_columns else f"CAST(NULL AS DOUBLE) AS {alias}"
        for source, alias in metric_fields
    )
    metrics_join_sql = (
        "LEFT JOIN detection_audio_metrics m ON d.detection_id = m.detection_id"
        if has_audio_metrics
        else ""
    )
    detector_select_sql = "d.Detector" if "Detector" in detection_columns else "'birdnet' AS Detector"
    detector_filter_sql = ""
    params = []
    if detector:
        detectors = [str(item).lower() for item in detector] if isinstance(detector, (list, tuple, set)) else [str(detector).lower()]
        if "Detector" in detection_columns:
            con.register("selected_detection_detectors", pd.DataFrame({"detector": detectors}))
            detector_filter_sql = "AND lower(COALESCE(d.Detector, 'birdnet')) IN (SELECT detector FROM selected_detection_detectors)"
        elif "birdnet" not in detectors:
            detector_filter_sql = "AND 1 = 0"

    dataset_filter_sql = ""
    if dataset_ids is not None:
        selected_ids = [str(dataset_id) for dataset_id in dataset_ids]
        if not selected_ids:
            return pd.DataFrame()
        con.register("selected_detection_datasets", pd.DataFrame({"dataset_id": selected_ids}))
        dataset_filter_sql = "AND d.dataset_id IN (SELECT dataset_id FROM selected_detection_datasets)"

    try:
        detections = con.execute(
            f"""
            SELECT
                d.detection_id,
                d.dataset_id,
                {dataset_type_sql},
                {detector_select_sql},
                ds.location_name,
                ds.latitude,
                ds.longitude,
                ds.period_label,
                d.recording_id,
                r.relative_audio_path,
                d.CommonName,
                d.Confidence,
                d.BeginPath,
                d.BeginTime,
                d.EndTime,
                d.DateTime,
                d.SpeciesCode,
                d.LowFreqHz,
                d.HighFreqHz,
                d.SourceTable,
                {metrics_select_sql}
            FROM detections d
            LEFT JOIN datasets ds ON d.dataset_id = ds.dataset_id
            LEFT JOIN recordings r ON d.recording_id = r.recording_id
            {metrics_join_sql}
            WHERE 1 = 1
            {detector_filter_sql}
            {dataset_filter_sql}
            """,
            params,
        ).df()
    finally:
        if detector and "Detector" in detection_columns:
            con.unregister("selected_detection_detectors")
        if dataset_ids is not None:
            con.unregister("selected_detection_datasets")

    return detections


def load_platform_data(con):
    datasets = load_platform_datasets(con)
    detections = load_platform_detections(con)
    return datasets, detections


def load_legacy_data(con):
    df = con.execute("SELECT * FROM detections").df()
    df["dataset_id"] = "legacy"
    df["dataset_type"] = "birds"
    df["Detector"] = "birdnet"
    df["location_name"] = "Unknown location"
    df["latitude"] = 59.6069
    df["longitude"] = 25.9136
    df["period_label"] = ""
    df["relative_audio_path"] = df.get("BeginPath", "")
    df["MeanDbfs"] = np.nan
    df["SignalDbfs"] = np.nan
    df["NoiseDbfs"] = np.nan
    df["SnrDb"] = np.nan
    datasets = pd.DataFrame(
        [
            {
                "dataset_id": "legacy",
                "dataset_type": "birds",
                "location_name": "Unknown location",
                "latitude": 59.6069,
                "longitude": 25.9136,
                "period_label": "",
            }
        ]
    )
    return datasets, df


def metric_box(label, value):
    st.markdown(
        f"""
        <div style="
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
            background: rgba(250, 250, 250, 0.7);
            min-height: 78px;
        ">
            <div style="font-size: 0.78rem; color: #5f6368; margin-bottom: 0.25rem;">{label}</div>
            <div style="font-size: 1.35rem; font-weight: 650; line-height: 1.2;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_design_system():
    st.markdown(
        """
        <style>
            .stApp {
                background: #f5f7fb;
            }
            header[data-testid="stHeader"] {
                height: 3rem;
                background: transparent;
                pointer-events: none;
            }
            div[data-testid="stToolbar"] {
                display: flex;
                pointer-events: none;
            }
            div[data-testid="stToolbar"] * {
                pointer-events: none;
            }
            div[data-testid="stToolbar"] button,
            div[data-testid="stToolbar"] [role="button"] {
                pointer-events: auto;
            }
            div[data-testid="stDecoration"],
            div[data-testid="stStatusWidget"] {
                display: none;
            }
            section[data-testid="stSidebar"] {
                background: #eef2f7;
                border-right: 1px solid rgba(16, 24, 40, 0.08);
            }
            div[data-testid="stVerticalBlockBorderWrapper"] {
                border-radius: 8px;
                border-color: rgba(16, 24, 40, 0.10);
                box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
                background: rgba(255, 255, 255, 0.92);
            }
            div[data-testid="stDataFrame"],
            div[data-testid="stTable"],
            div[data-testid="stPlotlyChart"],
            div[data-testid="stPydeckChart"] {
                border-radius: 8px;
                overflow: hidden;
            }
            div[data-testid="stMetric"] {
                padding: 0.7rem 0.85rem;
                border: 1px solid rgba(16, 24, 40, 0.10);
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.92);
            }
            .stButton > button,
            div[data-baseweb="select"] > div,
            div[data-baseweb="input"] > div {
                border-radius: 8px;
            }
            .stTabs [data-baseweb="tab-list"],
            [data-testid="stTabs"] [role="tablist"] {
                gap: 0.35rem;
                padding: 0.45rem 0;
                border-bottom: 1px solid rgba(128, 128, 128, 0.30);
                background: #f5f7fb;
                box-shadow: 0 6px 12px rgba(16, 24, 40, 0.05);
                overflow-x: auto;
            }
            [data-testid="stTabs"] [role="tablist"]:not([role="tabpanel"] [role="tablist"]),
            .stTabs [data-baseweb="tab-list"]:not([role="tabpanel"] [data-baseweb="tab-list"]) {
                position: sticky !important;
                top: 0 !important;
                z-index: 1000 !important;
            }
            .stTabs [data-baseweb="tab"],
            [data-testid="stTabs"] [role="tab"] {
                flex: 0 0 auto;
                height: 2.8rem;
                padding: 0 0.78rem;
                border: 1px solid rgba(128, 128, 128, 0.32);
                border-radius: 8px;
                background: rgba(128, 128, 128, 0.09);
                font-size: 0.95rem;
                font-weight: 650;
                white-space: nowrap;
                transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease;
            }
            .stTabs [data-baseweb="tab"]:hover,
            [data-testid="stTabs"] [role="tab"]:hover {
                border-color: rgba(239, 68, 68, 0.65);
                background: rgba(239, 68, 68, 0.09);
            }
            .stTabs [data-baseweb="tab"][aria-selected="true"],
            [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
                border-color: #ef4444;
                background: rgba(239, 68, 68, 0.18);
                color: #ef4444;
                box-shadow: inset 0 -3px 0 #ef4444;
            }
            .stTabs [data-baseweb="tab"]:focus-visible,
            [data-testid="stTabs"] [role="tab"]:focus-visible {
                outline: 2px solid #ef4444;
                outline-offset: 2px;
            }
            .stTabs [data-baseweb="tab-highlight"],
            [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
                display: none;
            }
            .species-matrix-shell {
                width: 100%;
                overflow-x: auto;
                border: 1px solid rgba(16, 24, 40, 0.10);
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.96);
            }
            .species-matrix {
                width: max-content;
                min-width: 100%;
                border-collapse: separate;
                border-spacing: 3px;
                padding: 0.35rem;
            }
            .species-matrix th {
                height: 2.7rem;
                padding: 0 0.45rem;
                color: #526071;
                background: #f5f7fb;
                font-size: 0.75rem;
                font-weight: 650;
                white-space: nowrap;
            }
            .species-matrix .species-column {
                position: sticky;
                left: 0;
                z-index: 2;
                width: 250px;
                min-width: 250px;
                max-width: 250px;
                background: #ffffff;
                box-shadow: 5px 0 8px rgba(16, 24, 40, 0.05);
            }
            .species-matrix thead .species-column {
                z-index: 3;
                background: #f5f7fb;
                text-align: left;
            }
            .species-matrix-species {
                display: flex;
                align-items: center;
                gap: 0.65rem;
                min-height: 42px;
                color: #172033;
                font-size: 0.88rem;
                font-weight: 600;
                line-height: 1.2;
                text-align: left;
                white-space: normal;
            }
            .species-matrix-photo,
            .species-matrix-placeholder {
                flex: 0 0 36px;
                width: 36px;
                height: 36px;
                border-radius: 6px;
                object-fit: cover;
                background: #dbe4ee;
            }
            .species-matrix-placeholder {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                color: #526071;
                font-size: 0.82rem;
                font-weight: 700;
            }
            .species-matrix-count {
                min-width: 42px;
                height: 42px;
                border-radius: 5px;
                text-align: center;
                font-size: 0.78rem;
                font-weight: 700;
            }
            .species-matrix-legend {
                display: flex;
                align-items: center;
                justify-content: flex-end;
                gap: 0.5rem;
                margin: 0.5rem 0 0.2rem;
                color: #526071;
                font-size: 0.75rem;
            }
            .species-matrix-gradient {
                width: 170px;
                height: 10px;
                border-radius: 5px;
                background: linear-gradient(90deg, #38bdf8, #22c55e, #fde047, #ef4444);
            }
            h1, h2, h3 {
                color: #172033;
                letter-spacing: 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sticky_dashboard_header(location, date_range, total_observations):
    location = html.escape(str(location))
    date_range = html.escape(str(date_range))
    total_observations = html.escape(str(total_observations))

    st.markdown(
        f"""
        <style>
            div[data-testid="stElementContainer"]:has(.sticky-dashboard-header),
            div.element-container:has(.sticky-dashboard-header) {{
                position: sticky;
                top: 0;
                z-index: 9999;
                background: rgba(255, 255, 255, 0.97);
                border-bottom: 1px solid rgba(49, 51, 63, 0.12);
                backdrop-filter: blur(8px);
            }}
            .sticky-dashboard-header {{
                margin: -0.6rem 0 1.1rem 0;
                padding: 0.9rem 0 0.95rem 0;
                background: rgba(255, 255, 255, 0.97);
            }}
            .sticky-dashboard-title {{
                margin: 0 0 0.78rem 0;
                color: #242637;
                font-size: clamp(1.75rem, 2.4vw, 2.65rem);
                font-weight: 760;
                line-height: 1.15;
            }}
            .sticky-dashboard-metrics {{
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.75rem;
            }}
            .sticky-dashboard-metric {{
                min-height: 74px;
                padding: 0.72rem 0.9rem;
                border: 1px solid rgba(49, 51, 63, 0.18);
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.95);
            }}
            .sticky-dashboard-metric-label {{
                margin-bottom: 0.25rem;
                color: #5f6368;
                font-size: 0.78rem;
            }}
            .sticky-dashboard-metric-value {{
                color: #1f2937;
                font-size: 1.2rem;
                font-weight: 650;
                line-height: 1.2;
                overflow-wrap: anywhere;
            }}
            @media (max-width: 900px) {{
                .sticky-dashboard-metrics {{
                    grid-template-columns: 1fr;
                }}
                .sticky-dashboard-title {{
                    font-size: 1.55rem;
                }}
            }}
        </style>
        <div class="sticky-dashboard-header">
            <div class="sticky-dashboard-title">AudioMoth Dashboard</div>
            <div class="sticky-dashboard-metrics">
                <div class="sticky-dashboard-metric">
                    <div class="sticky-dashboard-metric-label">Location</div>
                    <div class="sticky-dashboard-metric-value">{location}</div>
                </div>
                <div class="sticky-dashboard-metric">
                    <div class="sticky-dashboard-metric-label">Date range</div>
                    <div class="sticky-dashboard-metric-value">{date_range}</div>
                </div>
                <div class="sticky-dashboard-metric">
                    <div class="sticky-dashboard-metric-label">Total observations</div>
                    <div class="sticky-dashboard-metric-value">{total_observations}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def auto_time_resolution(start_time, end_time):
    span = end_time - start_time
    if span <= pd.Timedelta(days=4):
        return "Hour"
    if span <= pd.Timedelta(days=90):
        return "Day"
    return "Week"


def auto_temporal_resolution(start_time, end_time):
    span = end_time - start_time
    if span <= pd.Timedelta(hours=12):
        return "Minute"
    if span <= pd.Timedelta(days=7):
        return "Hour"
    if span <= pd.Timedelta(days=120):
        return "Day"
    return "Week"


def time_resolution_settings(resolution):
    settings = {
        "Minute": {
            "freq": "min",
            "label": "Recording minute",
            "axis": "Date and minute",
            "format": "%d %b %H:%M",
        },
        "Hour": {
            "freq": "h",
            "label": "Recording hour",
            "axis": "Date and hour",
            "format": "%d %b %H:%M",
        },
        "Day": {
            "freq": "D",
            "label": "Recording date",
            "axis": "Date",
            "format": "%d %b %Y",
        },
        "Week": {
            "freq": "W-MON",
            "label": "Recording week",
            "axis": "Week starting",
            "format": "%d %b %Y",
        },
    }
    return settings[resolution]


def add_time_bin(df, resolution):
    settings = time_resolution_settings(resolution)
    result = df.copy()
    if resolution == "Week":
        result["TimeBin"] = result["DateTime"].dt.to_period("W-SUN").dt.start_time
    else:
        result["TimeBin"] = result["DateTime"].dt.floor(settings["freq"])
    return result


def complete_time_counts(df, resolution):
    settings = time_resolution_settings(resolution)
    timeline = add_time_bin(df, resolution)
    counts = timeline.groupby("TimeBin").size().reset_index(name="Count")
    full_range = pd.date_range(
        start=timeline["TimeBin"].min(),
        end=timeline["TimeBin"].max(),
        freq=settings["freq"],
    )
    full_df = pd.DataFrame({"TimeBin": full_range})
    counts = full_df.merge(counts, on="TimeBin", how="left")
    counts["Count"] = counts["Count"].fillna(0).astype(int)
    counts["TimeLabel"] = counts["TimeBin"].dt.strftime(settings["format"])
    return counts


def render_detection_count_note():
    st.info(
        "Counts are detection activity, not animal abundance. BirdNET rows are species predictions for short audio windows, "
        "while BatDetect2 rows are detected bat call events; one individual can produce multiple detections."
    )


def resolve_audio_path(row, data_root):
    candidates = []
    for column in ("relative_audio_path", "BeginPath"):
        value = row.get(column)
        if pd.isna(value) or not str(value).strip():
            continue
        path = Path(str(value))
        candidates.append(path)
        if not path.is_absolute():
            candidates.append(data_root / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[-1] if candidates else None


def normalized_audio_reference(value, data_root):
    if pd.isna(value) or not str(value).strip():
        return ""
    path = Path(str(value))
    if not path.is_absolute():
        path = data_root / path
    return str(path.resolve(strict=False)).replace("\\", "/").casefold()


def clip_manifest_candidates(row, data_root):
    candidates = []
    for column in ("relative_audio_path", "BeginPath"):
        value = row.get(column)
        if pd.isna(value) or not str(value).strip():
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = data_root / path
        for parent in path.parents:
            if parent.name.casefold() == "audio":
                candidates.append(parent.parent / "clips" / "clips_manifest.csv")
                break

    unique = []
    seen = set()
    for path in candidates:
        key = str(path.resolve(strict=False)).casefold()
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


@st.cache_data(ttl=300, show_spinner=False)
def cached_clip_manifest(path, mtime_ns):
    manifest = pd.read_csv(path)
    for column in ("clip_start_seconds", "clip_end_seconds", "clip_duration_seconds"):
        if column in manifest.columns:
            manifest[column] = pd.to_numeric(manifest[column], errors="coerce")
    return manifest


def resolve_playback_audio(row, data_root, margin_seconds=3.0):
    begin_time = float(row["BeginTime"])
    end_time = float(row["EndTime"])
    desired_start = max(0.0, begin_time - margin_seconds)
    desired_end = end_time + margin_seconds
    source_references = {
        normalized_audio_reference(row.get(column), data_root)
        for column in ("relative_audio_path", "BeginPath")
    }
    source_references.discard("")

    matches = []
    for manifest_path in clip_manifest_candidates(row, data_root):
        if not manifest_path.exists():
            continue
        try:
            manifest = cached_clip_manifest(str(manifest_path), manifest_path.stat().st_mtime_ns)
        except Exception:
            continue
        required = {"clip_file", "source_audio_path", "clip_start_seconds", "clip_end_seconds"}
        if not required.issubset(manifest.columns):
            continue

        source_matches = manifest["source_audio_path"].apply(
            lambda value: normalized_audio_reference(value, data_root) in source_references
        )
        containing = manifest[
            source_matches
            & (manifest["clip_start_seconds"] <= begin_time + 1e-3)
            & (manifest["clip_end_seconds"] >= end_time - 1e-3)
        ].copy()
        if containing.empty:
            continue
        containing["_duration"] = containing["clip_end_seconds"] - containing["clip_start_seconds"]
        matches.extend((manifest_path, clip_row) for _, clip_row in containing.iterrows())

    if matches:
        manifest_path, clip_row = min(matches, key=lambda item: float(item[1]["_duration"]))
        dataset_dir = manifest_path.parent.parent
        clip_path = Path(str(clip_row["clip_file"]))
        if not clip_path.is_absolute():
            clip_path = dataset_dir / clip_path
        if clip_path.exists():
            clip_start = float(clip_row["clip_start_seconds"])
            clip_end = float(clip_row["clip_end_seconds"])
            available_start = max(desired_start, clip_start)
            available_end = min(desired_end, clip_end)
            return {
                "path": clip_path.resolve(),
                "start": max(0.0, available_start - clip_start),
                "duration": max(0.5, available_end - available_start),
                "is_clip": True,
                "source_start": available_start,
            }

    wav_path = resolve_audio_path(row, data_root)
    return {
        "path": wav_path,
        "start": desired_start,
        "duration": max(0.5, desired_end - desired_start),
        "is_clip": False,
        "source_start": desired_start,
    }


def load_audio_segment(wav_path, start, duration):
    try:
        with sf.SoundFile(str(wav_path)) as audio_file:
            sr = audio_file.samplerate
            audio_file.seek(int(start * sr))
            frames = int(duration * sr)
            y = audio_file.read(frames=frames, dtype="float32", always_2d=False)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            return y, sr
    except Exception:
        return librosa.load(str(wav_path.resolve()), sr=None, offset=start, duration=duration)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_audio_segment(path, mtime_ns, start, duration):
    return load_audio_segment(Path(path), start, duration)


@st.cache_data(ttl=86400, show_spinner=False)
def segment_dbfs(path, start, duration):
    wav_path = Path(path)
    y, _ = load_audio_segment(wav_path, start, duration)
    if len(y) == 0:
        return np.nan
    rms = float(np.sqrt(np.mean(np.square(y))))
    return 20 * np.log10(max(rms, 1e-9))


def quality_candidates(species_df, data_root, max_candidates=30):
    candidates = species_df.sort_values("Confidence", ascending=False).head(max_candidates).copy()
    rows = []

    for _, row in candidates.iterrows():
        playback = resolve_playback_audio(row, data_root)
        wav_path = playback["path"]
        if wav_path is None or not wav_path.exists():
            continue

        mean_dbfs = row.get("MeanDbfs")
        if pd.isna(mean_dbfs):
            try:
                mean_dbfs = segment_dbfs(str(wav_path), playback["start"], playback["duration"])
            except Exception:
                mean_dbfs = np.nan

        rows.append(
            {
                "Date": pd.to_datetime(row["DateTime"]).strftime("%Y-%m-%d %H:%M"),
                "Location": row.get("location_name", ""),
                "Confidence": float(row["Confidence"]),
                "Mean dBFS": mean_dbfs,
                "Background dBFS": row.get("NoiseDbfs"),
                "SNR dB": row.get("SnrDb"),
                "Audio": Path(str(row.get("relative_audio_path") or row.get("BeginPath"))).name,
                "RowIndex": row.name,
            }
        )

    if not rows:
        return pd.DataFrame()

    results = pd.DataFrame(rows)
    results["_confidence_rank"] = results["Confidence"].rank(pct=True)
    results["_snr_rank"] = pd.to_numeric(results["SNR dB"], errors="coerce").rank(pct=True).fillna(0.0)
    results["_quiet_rank"] = (-pd.to_numeric(results["Background dBFS"], errors="coerce")).rank(pct=True).fillna(0.0)
    results["Quality"] = (
        (0.50 * results["_confidence_rank"])
        + (0.35 * results["_snr_rank"])
        + (0.15 * results["_quiet_rank"])
    )
    results = results.sort_values(["Quality", "Confidence"], ascending=[False, False]).head(5).reset_index(drop=True)
    results = results.drop(columns=["_confidence_rank", "_snr_rank", "_quiet_rank"])
    results.insert(0, "Rank", np.arange(1, len(results) + 1))
    return results


def bat_call_morphology(row):
    low_hz = pd.to_numeric(pd.Series([row.get("LowFreqHz")]), errors="coerce").iloc[0]
    high_hz = pd.to_numeric(pd.Series([row.get("HighFreqHz")]), errors="coerce").iloc[0]
    duration_ms = max(0.0, float(row["EndTime"]) - float(row["BeginTime"])) * 1000
    if pd.isna(low_hz) or pd.isna(high_hz) or high_hz <= low_hz:
        return None

    bandwidth_hz = float(high_hz - low_hz)
    center_hz = float((high_hz + low_hz) / 2)
    relative_bandwidth = bandwidth_hz / max(center_hz, 1.0)
    if relative_bandwidth >= 0.35:
        shape = "Short pulse sweeping through a wide range of pitches"
        technical_shape = "broadband FM"
        explanation = "The bat changes pitch quickly during this pulse, which helps it judge nearby objects in detail."
    elif relative_bandwidth <= 0.15:
        shape = "Longer pulse staying near one pitch"
        technical_shape = "narrowband / QCF"
        explanation = "The pitch changes little, making this pulse well suited to detecting targets over a longer distance."
    else:
        shape = "Pitch sweep followed by a steadier tone"
        technical_shape = "mixed FM-QCF"
        explanation = "This combines a quick pitch change with a steadier ending and is common during general navigation."

    return {
        "shape": shape,
        "technical_shape": technical_shape,
        "explanation": explanation,
        "duration_ms": duration_ms,
        "low_khz": float(low_hz) / 1000,
        "high_khz": float(high_hz) / 1000,
        "bandwidth_khz": bandwidth_hz / 1000,
    }


def bat_view_context(row, context_df, source_start, duration):
    if context_df is None or context_df.empty:
        return pd.DataFrame([row])

    candidates = context_df.copy()
    if "Detector" in candidates.columns:
        candidates = candidates[candidates["Detector"].fillna("").astype(str).str.lower() == "batdetect2"]
    if "recording_id" in candidates.columns and pd.notna(row.get("recording_id")):
        candidates = candidates[candidates["recording_id"] == row["recording_id"]]
    elif "relative_audio_path" in candidates.columns:
        candidates = candidates[candidates["relative_audio_path"] == row.get("relative_audio_path")]
    source_end = source_start + duration
    candidates = candidates[
        (pd.to_numeric(candidates["EndTime"], errors="coerce") >= source_start)
        & (pd.to_numeric(candidates["BeginTime"], errors="coerce") <= source_end)
    ].copy()
    return candidates.sort_values("BeginTime")


def bat_sequence_context(row, context_df, source_start, duration):
    candidates = bat_view_context(row, context_df, source_start, duration)

    selected_low = pd.to_numeric(pd.Series([row.get("LowFreqHz")]), errors="coerce").iloc[0]
    selected_high = pd.to_numeric(pd.Series([row.get("HighFreqHz")]), errors="coerce").iloc[0]
    if pd.notna(selected_low) and pd.notna(selected_high) and selected_high > selected_low:
        selected_center = (float(selected_low) + float(selected_high)) / 2
        candidate_low = pd.to_numeric(candidates["LowFreqHz"], errors="coerce")
        candidate_high = pd.to_numeric(candidates["HighFreqHz"], errors="coerce")
        candidate_center = (candidate_low + candidate_high) / 2
        frequency_ratio = candidate_center / selected_center
        candidates = candidates[frequency_ratio.between(0.70, 1.30)].copy()

    if candidates.empty:
        return pd.DataFrame([row])

    # BatDetect2 can emit overlapping boxes with different species guesses for one
    # acoustic pulse. Keep the strongest box from each overlapping time group.
    merged_rows = []
    overlap_group = []
    group_end = None
    for _, candidate in candidates.sort_values(["BeginTime", "EndTime"]).iterrows():
        begin = float(candidate["BeginTime"])
        end = float(candidate["EndTime"])
        if overlap_group and begin > group_end:
            merged_rows.append(max(overlap_group, key=lambda item: float(item.get("Confidence", 0))))
            overlap_group = []
            group_end = None
        overlap_group.append(candidate)
        group_end = end if group_end is None else max(group_end, end)
    if overlap_group:
        merged_rows.append(max(overlap_group, key=lambda item: float(item.get("Confidence", 0))))

    return pd.DataFrame(merged_rows).sort_values("BeginTime")


def bat_sequence_description(sequence_df):
    pulse_count = len(sequence_df)
    if pulse_count < 2:
        return f"{pulse_count} detected pulse in this frequency band", None, "Single detected pulse"

    pulse_times = pd.to_numeric(sequence_df["BeginTime"], errors="coerce").dropna().sort_values()
    intervals_ms = pulse_times.diff().dropna() * 1000
    if intervals_ms.empty:
        return f"{pulse_count} detected pulses in this frequency band", None, "Detected pulse sequence"

    median_interval = float(intervals_ms.median())
    spacing_variation = float(intervals_ms.std(ddof=0) / max(intervals_ms.mean(), 1.0))
    if pulse_count >= 5 and median_interval < 15:
        pattern = "Very rapid sequence; this can occur during close approach or a possible feeding buzz."
        plot_label = "Very rapid echolocation sequence"
    elif pulse_count >= 4 and median_interval < 45:
        pattern = "Rapid sequence; this can occur as a bat approaches or inspects something nearby."
        plot_label = "Rapid approach-like echolocation sequence"
    elif pulse_count >= 4 and spacing_variation <= 0.45:
        pattern = "Regularly spaced echolocation sequence, consistent with general searching or navigation."
        plot_label = "Regularly spaced echolocation sequence"
    else:
        pattern = "Irregular or incomplete sequence; the visible spacing does not support a stronger behavioral interpretation."
        plot_label = "Irregular or incomplete pulse sequence"
    return f"{pulse_count} detected pulses in this frequency band; median spacing {median_interval:.0f} ms", pattern, plot_label


def add_horizontal_brace(ax, x_start, x_end, label):
    if x_end <= x_start:
        return
    width = x_end - x_start
    midpoint = (x_start + x_end) / 2
    y = 0.58
    depth = 0.20
    vertices = [
        (x_start, y),
        (x_start + 0.02 * width, y),
        (x_start + 0.02 * width, y - depth / 2),
        (x_start + 0.08 * width, y - depth / 2),
        (x_start + 0.25 * width, y - depth / 2),
        (midpoint - 0.03 * width, y - depth / 2),
        (midpoint, y - depth),
        (midpoint + 0.03 * width, y - depth / 2),
        (x_end - 0.25 * width, y - depth / 2),
        (x_end - 0.08 * width, y - depth / 2),
        (x_end - 0.02 * width, y - depth / 2),
        (x_end - 0.02 * width, y),
        (x_end, y),
    ]
    codes = [MatplotlibPath.MOVETO] + [MatplotlibPath.CURVE4] * 12
    brace = PathPatch(
        MatplotlibPath(vertices, codes),
        transform=ax.get_xaxis_transform(),
        fill=False,
        color="#d39b00",
        linewidth=1.8,
        clip_on=False,
    )
    ax.add_patch(brace)
    ax.text(
        midpoint,
        0.82,
        label,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        color="#1f2937",
        fontsize=9,
    )


def render_audio_inspector(row, data_root, dataset_type, key_prefix, context_df=None):
    playback = resolve_playback_audio(row, data_root)
    wav_path = playback["path"]
    start = playback["start"]
    duration = playback["duration"]

    if wav_path is None or not wav_path.exists():
        st.error(f"Audio file not found: {wav_path}")
        return

    try:
        cache_start = round(start, 3)
        cache_duration = round(duration, 3)
        y, sr = cached_audio_segment(str(wav_path.resolve()), wav_path.stat().st_mtime_ns, cache_start, cache_duration)

        playback_options = {
            "Original speed": 1,
            "4x time expansion": 4,
            "10x time expansion": 10,
        }
        playback_kind = str(dataset_type).lower()
        default_playback = "10x time expansion" if playback_kind in {"bats", "batdetect2"} else "Original speed"

        morphology = bat_call_morphology(row) if playback_kind in {"bats", "batdetect2"} else None
        sequence_df = None
        sequence_summary = None
        sequence_interpretation = None
        sequence_plot_label = None
        sequence_species_colors = {}
        if morphology:
            view_df = bat_view_context(row, context_df, playback["source_start"], duration)
            sequence_df = bat_sequence_context(row, context_df, playback["source_start"], duration)
            sequence_summary, sequence_interpretation, sequence_plot_label = bat_sequence_description(sequence_df)

        playback_label = st.selectbox(
            "Audio playback",
            list(playback_options),
            index=list(playback_options).index(default_playback),
            key=f"{key_prefix}_playback",
        )
        expansion_factor = playback_options[playback_label]
        playback_sr = max(1, int(round(sr / expansion_factor)))

        buffer = io.BytesIO()
        sf.write(buffer, y, playback_sr, format="WAV")
        st.audio(buffer.getvalue(), format="audio/wav")

        if playback["is_clip"]:
            st.caption(f"Using generated clip: {wav_path.name}")

        if expansion_factor > 1:
            st.caption(
                f"Time-expanded {expansion_factor}x: pitch and playback speed are both lowered "
                f"by a factor of {expansion_factor}."
            )

        spectrum = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        if morphology:
            fig = plt.figure(figsize=(12, 5.8))
            grid = fig.add_gridspec(2, 1, height_ratios=[1.15, 5], hspace=0.02)
            annotation_ax = fig.add_subplot(grid[0])
            ax = fig.add_subplot(grid[1], sharex=annotation_ax)
            annotation_ax.set_xlim(0, duration)
            annotation_ax.set_ylim(0, 1)
            annotation_ax.axis("off")
        else:
            fig, ax = plt.subplots(figsize=(12, 5))
        librosa.display.specshow(spectrum, sr=sr, x_axis="time", y_axis="log", ax=ax)
        ax.set_ylim([20, sr / 2])
        detection_start = max(0.0, float(row["BeginTime"]) - playback["source_start"])
        detection_end = min(duration, float(row["EndTime"]) - playback["source_start"])
        if detection_end > detection_start:
            if morphology:
                sequence_start = max(0.0, float(sequence_df["BeginTime"].min()) - playback["source_start"])
                sequence_end = min(duration, float(sequence_df["EndTime"].max()) - playback["source_start"])
                add_horizontal_brace(annotation_ax, sequence_start, sequence_end, sequence_plot_label)
                species_names = sorted(view_df["CommonName"].fillna("Unclassified").astype(str).unique())
                pulse_colors = ["#168aad", "#f4a261", "#52b788", "#9b5de5", "#e63946", "#6c757d"]
                sequence_species_colors = {
                    species: pulse_colors[index % len(pulse_colors)]
                    for index, species in enumerate(species_names)
                }
                marker_levels = {
                    species: 0.05 + (index % 4) * 0.075
                    for index, species in enumerate(species_names)
                }
                for _, sequence_row in view_df.iterrows():
                    pulse_x = (
                        float(sequence_row["BeginTime"])
                        + float(sequence_row["EndTime"])
                    ) / 2 - playback["source_start"]
                    pulse_species = str(sequence_row.get("CommonName") or "Unclassified")
                    annotation_ax.scatter(
                        pulse_x,
                        marker_levels[pulse_species],
                        marker="v",
                        s=34,
                        color=sequence_species_colors[pulse_species],
                        edgecolor="white",
                        linewidth=0.6,
                        zorder=4,
                    )
                selected_x = (detection_start + detection_end) / 2
                selected_y = (float(row["LowFreqHz"]) + float(row["HighFreqHz"])) / 2
                label_x = min(max(selected_x + 0.65, 0.75), max(0.75, duration - 0.2))
                selected_marker_y = marker_levels.get(str(row.get("CommonName") or "Unclassified"), 0.05)
                annotation_ax.annotate(
                    "Selected pulse",
                    xy=(selected_x, selected_marker_y),
                    xytext=(label_x, 0.35),
                    ha="center",
                    color="white",
                    fontsize=9,
                    arrowprops={"arrowstyle": "->", "color": "#ff5a5f", "linewidth": 1.8},
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "#ff5a5f", "edgecolor": "none", "alpha": 0.9},
                )
                arrow_start_y = min((sr / 2) * 0.97, max(selected_y * 1.25, selected_y + 1500))
                ax.annotate(
                    "",
                    xy=(selected_x, selected_y),
                    xytext=(selected_x, arrow_start_y),
                    arrowprops={"arrowstyle": "->", "color": "#ff5a5f", "linewidth": 1.5},
                )
            else:
                ax.axvspan(detection_start, detection_end, color="#ff5a5f", alpha=0.18)
                label_y = (sr / 2) * 0.88
                ax.annotate(
                    str(row.get("CommonName") or "Selected detection"),
                    xy=(detection_start, label_y),
                    color="white",
                    fontsize=9,
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "#ff5a5f", "edgecolor": "none", "alpha": 0.9},
                )
        st.pyplot(fig)
        plt.close(fig)

        if morphology:
            if sequence_species_colors:
                legend_items = "".join(
                    '<span style="display:inline-flex;align-items:center;gap:0.35rem;margin-right:1rem;">'
                    f'<span style="width:0.65rem;height:0.65rem;background:{color};border-radius:50%;"></span>'
                    f"{html.escape(species)}</span>"
                    for species, color in sequence_species_colors.items()
                )
                st.markdown(
                    '<div style="font-size:0.85rem;color:#1f2937;margin:0.1rem 0 0.75rem 0;">'
                    f"<strong>BatDetect2 pulse predictions:</strong> {legend_items}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(f"**Visible call sequence:** {sequence_summary}")
            if sequence_interpretation:
                st.write(sequence_interpretation)
            st.markdown(
                f"**Selected pulse:** {morphology['shape']} "
                f"({morphology['technical_shape']})"
            )
            st.write(morphology["explanation"])
            st.caption(
                f"BatDetect2 box: {morphology['low_khz']:.1f}-{morphology['high_khz']:.1f} kHz, "
                f"{morphology['duration_ms']:.1f} ms, {morphology['bandwidth_khz']:.1f} kHz bandwidth. "
                "The brace marks nearby BatDetect2 pulses in the selected frequency band; an arrow marks the selected pulse. "
                "These are clues, not confirmed behavior such as search, approach, feeding buzz, or social calling."
            )
        elif playback_kind not in {"bats", "batdetect2"}:
            st.caption(
                "BirdNET identifies the species in an audio window but does not classify the vocalization "
                "as song, contact call, or alarm call."
            )

    except Exception as exc:
        st.exception(exc)


def dataset_label(row):
    dataset_id = str(row.get("dataset_id") or "Dataset").strip()
    location = str(row.get("location_name") or "").strip()
    dataset_type = str(row.get("dataset_type") or "").strip()
    period = str(row.get("period_label") or "").strip()

    label_parts = [dataset_id]
    if location:
        label_parts.append(location)
    if dataset_type:
        label_parts.append(dataset_type)
    if period:
        label_parts.append(period)
    return " - ".join(label_parts)


def dataset_sort_key(dataset_id):
    text = str(dataset_id)
    match = re.match(r"^(\d+)([A-Za-z]*)", text)
    if not match:
        return (10**9, "", text.lower())
    return (int(match.group(1)), match.group(2).lower(), text.lower())


def location_points(datasets):
    points = datasets.copy()
    points["latitude"] = pd.to_numeric(points["latitude"], errors="coerce")
    points["longitude"] = pd.to_numeric(points["longitude"], errors="coerce")
    points = points.dropna(subset=["latitude", "longitude"])
    if points.empty:
        return points

    points["lat"] = points["latitude"]
    points["lon"] = points["longitude"]
    points["label"] = points.apply(dataset_label, axis=1)
    return points


def extent_view_state(points, selected_dataset_id=None, fallback_zoom=10):
    if points.empty:
        return None

    lat_min, lat_max = float(points["lat"].min()), float(points["lat"].max())
    lon_min, lon_max = float(points["lon"].min()), float(points["lon"].max())
    span = max(lat_max - lat_min, lon_max - lon_min)
    zoom = 11 if span <= 0 else max(2, min(11, math.log2(180 / span) - 1.2))

    return pdk.ViewState(
        latitude=(lat_min + lat_max) / 2,
        longitude=(lon_min + lon_max) / 2,
        zoom=zoom,
        pitch=0,
    )


def map_deck(points, selected_dataset_id=None, satellite=True, zoom=7):
    if points.empty:
        return None

    view_state = extent_view_state(points, selected_dataset_id, fallback_zoom=zoom)
    points = points.copy()
    points["is_selected"] = points["dataset_id"] == selected_dataset_id
    points["radius"] = np.where(points["is_selected"], 260, 160)
    points["color"] = points["is_selected"].apply(lambda value: [220, 53, 69, 235] if value else [39, 125, 161, 210])

    layers = []
    if satellite:
        layers.append(
            pdk.Layer(
                "TileLayer",
                data="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                min_zoom=0,
                max_zoom=19,
                tile_size=256,
                opacity=1,
            )
        )

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=points,
            get_position="[lon, lat]",
            get_radius="radius",
            get_fill_color="color",
            get_line_color=[255, 255, 255],
            radius_min_pixels=6,
            radius_max_pixels=24,
            line_width_min_pixels=1,
            pickable=True,
        )
    )

    return pdk.Deck(
        map_style=None if satellite else "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
        initial_view_state=view_state,
        layers=layers,
        tooltip={"text": "{label}"},
    )


def satellite_map_figure(points, zoom=12):
    center = {"lat": float(points["lat"].mean()), "lon": float(points["lon"].mean())}
    figure = px.scatter_map(
        points,
        lat="lat",
        lon="lon",
        hover_name="label",
        zoom=zoom,
        center=center,
    )
    figure.update_traces(marker={"size": 15, "color": "#ffd246"})
    figure.update_layout(
        map_style="white-bg",
        map_layers=[
            {
                "below": "traces",
                "sourcetype": "raster",
                "source": [
                    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                ],
            }
        ],
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=500,
    )
    return figure


def wikipedia_request(params):
    response = requests.get(WIKIPEDIA_API, headers=WIKIPEDIA_HEADERS, timeout=12, params=params)
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After", "a few minutes")
        raise RuntimeError(f"Wikipedia rate limit reached. Please try again after {retry_after}.")
    response.raise_for_status()
    return response.json()


def normalise_image_url(url):
    if not url:
        return None
    return f"https:{url}" if url.startswith("//") else url


def wikipedia_cache_directory(species_name):
    name = " ".join(str(species_name).split())
    ascii_name = name.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name).strip("-") or "species"
    digest = hashlib.sha1(name.casefold().encode("utf-8")).hexdigest()[:10]
    return WIKIPEDIA_CACHE_DIR / f"{slug}-{digest}"


def cached_media_path(cache_dir, stem):
    matches = sorted(cache_dir.glob(f"{stem}.*")) if cache_dir.exists() else []
    return matches[0] if matches else None


def atomic_write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_path, path)


def download_wikipedia_media(url, cache_dir, stem):
    url = normalise_image_url(url)
    if not url:
        return None

    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (
        hostname.endswith(".wikimedia.org") or hostname.endswith(".wikipedia.org")
    ):
        return None

    response = requests.get(url, headers=WIKIPEDIA_HEADERS, timeout=20, stream=True)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
    extensions = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/svg+xml": ".svg",
    }
    extension = extensions.get(content_type)
    if not extension:
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    target_path = cache_dir / f"{stem}{extension}"
    temporary_path = cache_dir / f".{stem}{extension}.tmp"
    total_bytes = 0
    with temporary_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            total_bytes += len(chunk)
            if total_bytes > 15 * 1024 * 1024:
                handle.close()
                temporary_path.unlink(missing_ok=True)
                raise RuntimeError("Wikipedia image exceeds the 15 MB cache limit.")
            handle.write(chunk)
    os.replace(temporary_path, target_path)
    return target_path


def cached_profile_for_ui(species_name):
    cache_dir = wikipedia_cache_directory(species_name)
    metadata_path = cache_dir / "profile.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if metadata.get("cache_version") != WIKIPEDIA_CACHE_VERSION:
        return None

    image_path = cached_media_path(cache_dir, "profile-image")
    range_map_path = cached_media_path(cache_dir, "range-map")
    return {
        "title": metadata.get("title"),
        "extract": metadata.get("extract"),
        "image_url": str(image_path) if image_path else metadata.get("image_source_url"),
        "range_map_url": str(range_map_path) if range_map_path else metadata.get("range_map_source_url"),
        "page_url": metadata.get("page_url"),
        "details": metadata.get("details", {}),
        "cached": True,
    }


@st.cache_data(show_spinner=False)
def local_image_data_uri(path_string, modified_ns):
    path = Path(path_string)
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
    }
    mime_type = mime_types.get(path.suffix.lower())
    if not mime_type:
        return path_string
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def image_source_for_html(source):
    if not source:
        return None
    path = Path(source)
    if path.exists():
        return local_image_data_uri(str(path), path.stat().st_mtime_ns)
    return source


def shorten_text(text, max_characters=320):
    text = " ".join(str(text or "").split())
    if len(text) <= max_characters:
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text)
    selected = []
    for sentence in sentences:
        if selected and len(" ".join(selected + [sentence])) > max_characters:
            break
        selected.append(sentence)

    shortened = " ".join(selected)
    if shortened and len(shortened) <= max_characters:
        return shortened
    return text[:max_characters].rsplit(" ", 1)[0] + "..."


def extract_article_content(html, max_characters=320):
    soup = BeautifulSoup(html, "html.parser")

    range_map_url = None
    for image in soup.find_all("img"):
        descriptor = " ".join(
            [
                image.get("alt", ""),
                image.get("title", ""),
                image.get("src", ""),
                image.parent.get_text(" ", strip=True)[:300] if image.parent else "",
            ]
        ).lower()
        if "range" in descriptor or "distribution" in descriptor:
            range_map_url = normalise_image_url(image.get("src") or image.get("data-src"))
            break

    for element in soup.select("table, style, script, sup, .mw-editsection"):
        element.decompose()

    details = {}
    for heading in soup.find_all(["h2", "h3"]):
        heading_text = " ".join(heading.get_text(" ", strip=True).split()).lower()
        display_name = next(
            (
                name
                for name, keywords in WIKIPEDIA_DETAIL_SECTIONS.items()
                if name not in details and any(keyword in heading_text for keyword in keywords)
            ),
            None,
        )
        if not display_name:
            continue

        paragraphs = []
        heading_anchor = heading.parent if "mw-heading" in heading.parent.get("class", []) else heading
        for sibling in heading_anchor.find_next_siblings():
            if sibling.name in ("h2", "h3"):
                break
            candidates = [sibling] if sibling.name == "p" else sibling.find_all("p")
            for paragraph in candidates:
                text = " ".join(paragraph.get_text(" ", strip=True).split())
                if text:
                    paragraphs.append(text)
            if sum(len(item) for item in paragraphs) >= max_characters:
                break

        text = shorten_text(" ".join(paragraphs), max_characters)
        if text:
            details[display_name] = text

    return details, range_map_url


@st.cache_data(ttl=86400, show_spinner=False)
def wikipedia_species_profile(species_name):
    cached_profile = cached_profile_for_ui(species_name)
    if cached_profile:
        return cached_profile

    search_name = WIKIPEDIA_NAME_ALIASES.get(species_name.lower(), species_name)
    query_data = wikipedia_request(
        {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": search_name,
            "gsrnamespace": 0,
            "gsrlimit": 1,
            "prop": "extracts|pageimages|info",
            "exintro": 1,
            "explaintext": 1,
            "inprop": "url",
            "piprop": "thumbnail",
            "pithumbsize": 800,
        }
    )
    pages = query_data.get("query", {}).get("pages", {})
    if not pages:
        return None

    page = next(iter(pages.values()))
    title = page["title"]
    article_data = wikipedia_request(
        {"action": "parse", "format": "json", "page": title, "prop": "text"}
    )
    article_html = article_data.get("parse", {}).get("text", {}).get("*", "")
    details, range_map_url = extract_article_content(article_html)

    profile = {
        "title": title,
        "extract": shorten_text(page.get("extract"), 420),
        "image_source_url": normalise_image_url(page.get("thumbnail", {}).get("source")),
        "range_map_source_url": normalise_image_url(range_map_url),
        "page_url": page.get("fullurl"),
        "details": details,
    }
    cache_dir = wikipedia_cache_directory(species_name)
    image_path = None
    range_map_path = None
    try:
        image_path = download_wikipedia_media(profile["image_source_url"], cache_dir, "profile-image")
    except Exception:
        pass
    try:
        range_map_path = download_wikipedia_media(profile["range_map_source_url"], cache_dir, "range-map")
    except Exception:
        pass

    atomic_write_json(
        cache_dir / "profile.json",
        {
            "cache_version": WIKIPEDIA_CACHE_VERSION,
            "species_name": species_name,
            "search_name": search_name,
            "cached_at": datetime.now().astimezone().isoformat(),
            **profile,
        },
    )
    return {
        "title": profile["title"],
        "extract": profile["extract"],
        "image_url": str(image_path) if image_path else profile["image_source_url"],
        "range_map_url": str(range_map_path) if range_map_path else profile["range_map_source_url"],
        "page_url": profile["page_url"],
        "details": profile["details"],
        "cached": True,
    }


@st.cache_data(ttl=604800, show_spinner=False)
def wikipedia_species_images(species_names):
    """Load cached thumbnails first, then download only missing species images."""
    species_names = tuple(dict.fromkeys(str(name) for name in species_names if str(name).strip()))
    lookup_titles = {
        name: WIKIPEDIA_NAME_ALIASES.get(name.lower(), name)
        for name in species_names
    }
    images = {name: None for name in species_names}

    missing_names = []
    for name in species_names:
        cache_dir = wikipedia_cache_directory(name)
        image_path = cached_media_path(cache_dir, "thumbnail")
        if image_path is None:
            image_path = cached_media_path(cache_dir, "profile-image")
        if image_path:
            images[name] = str(image_path)
        else:
            missing_names.append(name)

    for start_index in range(0, len(missing_names), 40):
        batch_names = missing_names[start_index : start_index + 40]
        requested_titles = [lookup_titles[name] for name in batch_names]
        try:
            data = wikipedia_request(
                {
                    "action": "query",
                    "format": "json",
                    "titles": "|".join(requested_titles),
                    "redirects": 1,
                    "prop": "pageimages",
                    "piprop": "thumbnail",
                    "pithumbsize": 96,
                }
            )
        except Exception:
            continue

        query = data.get("query", {})
        title_changes = {}
        for change in query.get("normalized", []) + query.get("redirects", []):
            if change.get("from") and change.get("to"):
                title_changes[str(change["from"]).casefold()] = str(change["to"])

        pages_by_title = {
            str(page.get("title", "")).casefold(): page
            for page in query.get("pages", {}).values()
        }

        def resolved_title(title):
            current = title
            for _ in range(6):
                replacement = title_changes.get(current.casefold())
                if not replacement or replacement.casefold() == current.casefold():
                    break
                current = replacement
            return current

        for name in batch_names:
            requested = lookup_titles[name]
            page = pages_by_title.get(resolved_title(requested).casefold())
            if page:
                source_url = normalise_image_url(page.get("thumbnail", {}).get("source"))
                try:
                    image_path = download_wikipedia_media(
                        source_url, wikipedia_cache_directory(name), "thumbnail"
                    )
                except Exception:
                    image_path = None
                images[name] = str(image_path) if image_path else source_url

    return images


def activity_cell_color(value, maximum):
    if value <= 0 or maximum <= 0:
        return "#edf2f7", "#8a96a5"

    if maximum <= 1:
        position = 0.0
    else:
        position = (math.log1p(value) - math.log(2)) / (math.log1p(maximum) - math.log(2))
        position = max(0.0, min(1.0, position))

    stops = [
        (0.0, (56, 189, 248)),
        (0.38, (34, 197, 94)),
        (0.68, (253, 224, 71)),
        (1.0, (239, 68, 68)),
    ]
    for (left_position, left_color), (right_position, right_color) in zip(stops, stops[1:]):
        if position <= right_position:
            fraction = (position - left_position) / (right_position - left_position)
            rgb = tuple(
                round(left + (right - left) * fraction)
                for left, right in zip(left_color, right_color)
            )
            break
    else:
        rgb = stops[-1][1]

    text_color = "#ffffff" if position >= 0.9 else "#102033"
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})", text_color


def render_occurrence_matrix(occurrence_matrix, time_axis_label):
    species_names = tuple(str(name) for name in occurrence_matrix.index)
    image_urls = wikipedia_species_images(species_names)
    maximum = int(occurrence_matrix.to_numpy().max()) if not occurrence_matrix.empty else 0

    header_cells = "".join(
        f'<th title="{html.escape(str(label), quote=True)}">{html.escape(str(label))}</th>'
        for label in occurrence_matrix.columns
    )
    rows = []
    for species, values in occurrence_matrix.iterrows():
        safe_species = html.escape(str(species))
        image_url = image_source_for_html(image_urls.get(str(species)))
        if image_url:
            image_html = (
                f'<img class="species-matrix-photo" src="{html.escape(image_url, quote=True)}" '
                f'alt="{safe_species}" loading="lazy">'
            )
        else:
            initial = html.escape(str(species).strip()[:1].upper() or "?")
            image_html = f'<span class="species-matrix-placeholder">{initial}</span>'

        count_cells = []
        for value in values:
            count = int(value)
            background, foreground = activity_cell_color(count, maximum)
            label = str(count) if count else ""
            count_cells.append(
                f'<td class="species-matrix-count" style="background:{background};color:{foreground}" '
                f'title="{safe_species}: {count} detections">{label}</td>'
            )

        rows.append(
            '<tr><td class="species-column"><div class="species-matrix-species">'
            f'{image_html}<span>{safe_species}</span></div></td>{"".join(count_cells)}</tr>'
        )

    matrix_html = (
        '<div class="species-matrix-legend"><span>Fewer</span>'
        '<span class="species-matrix-gradient"></span><span>More detections</span></div>'
        '<div class="species-matrix-shell"><table class="species-matrix">'
        f'<thead><tr><th class="species-column">Species</th>{header_cells}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
        f'<div class="species-matrix-legend"><span>Columns: {html.escape(time_axis_label)}</span></div>'
    )
    st.markdown(matrix_html, unsafe_allow_html=True)


def checkbox_selection(label, options, default_options, key_prefix):
    selected = []
    st.caption(label)
    defaults = set(default_options)
    for index, option in enumerate(options):
        option_key = re.sub(r"[^A-Za-z0-9_]+", "_", str(option))
        if st.checkbox(str(option), value=option in defaults, key=f"{key_prefix}_{index}_{option_key}"):
            selected.append(option)
    return selected


def render_highlight_scope_selector(datasets, dataset_labels, selected_dataset_id, gallery_type):
    scope_key = f"{gallery_type}_highlight_scope"
    selected_key = f"{gallery_type}_highlight_selected"

    st.subheader("Highlight Scope")
    with st.form(f"{gallery_type}_highlight_scope_form"):
        scope = st.radio(
            "Use detections from",
            ["Current dataset", "Selected datasets", "All datasets"],
            horizontal=True,
            key=f"{gallery_type}_highlight_scope_radio",
        )

        chosen_dataset_ids = []
        if scope == "Selected datasets":
            chosen_labels = checkbox_selection(
                "Datasets",
                list(dataset_labels.keys()),
                [label for label, dataset_id in dataset_labels.items() if dataset_id == selected_dataset_id],
                f"{gallery_type}_highlight_dataset",
            )
            chosen_dataset_ids = [dataset_labels[label] for label in chosen_labels]

        submitted = st.form_submit_button("Generate highlights")

    if submitted:
        if scope == "Current dataset":
            chosen_dataset_ids = [selected_dataset_id]
        elif scope == "All datasets":
            chosen_dataset_ids = datasets["dataset_id"].tolist()

        st.session_state[scope_key] = scope
        st.session_state[selected_key] = chosen_dataset_ids

    if selected_key not in st.session_state:
        st.info("Choose a scope and press Generate highlights.")
        return None

    selected_ids = st.session_state[selected_key]
    if not selected_ids:
        st.warning("Select at least one dataset before generating highlights.")
        return None

    selected_labels = [
        label
        for label, dataset_id in dataset_labels.items()
        if dataset_id in set(selected_ids)
    ]
    if st.session_state.get(scope_key) == "All datasets":
        st.caption(f"Generated from all datasets: {len(selected_ids)} selected.")
    else:
        st.caption("Generated from: " + ", ".join(selected_labels))

    return selected_ids


def render_gallery(gallery_df, data_root, gallery_type):
    label = "Bird" if gallery_type == "birdnet" else "Bat"
    st.header(f"{label} Highlights Gallery")
    st.caption(
        "Highlights combine classifier confidence, signal-to-noise ratio, and background quietness. "
        "High SNR means the detected sound stands clearly above the local noise floor."
    )

    if gallery_df.empty:
        st.warning(f"No {gallery_type} detections found.")
        return

    dataset_count = gallery_df["dataset_id"].nunique() if "dataset_id" in gallery_df.columns else 0
    species_counts = gallery_df["CommonName"].value_counts()
    st.caption(
        f"Scope contains {len(gallery_df):,} detections across {dataset_count:,} dataset(s) "
        f"and {len(species_counts):,} species."
    )
    max_species = st.slider(
        "Species in gallery",
        min_value=1,
        max_value=int(min(50, len(species_counts))),
        value=int(min(12, len(species_counts))),
    )
    max_candidates = st.slider("High-confidence candidates checked per species", 5, 50, 20, step=5)

    selected_species = species_counts.head(max_species).index.tolist()
    gallery_progress = st.progress(0.0, text=f"Preparing spectrograms: 0 / {len(selected_species)}")
    gallery_image_urls = wikipedia_species_images(tuple(selected_species))
    for species_index, species in enumerate(selected_species, start=1):
        gallery_progress.progress(
            (species_index - 1) / len(selected_species),
            text=f"Loading spectrogram {species_index} / {len(selected_species)}: {species}",
        )
        species_key = re.sub(r"[^A-Za-z0-9_]+", "_", species)
        species_df = gallery_df[gallery_df["CommonName"] == species]
        with st.container(border=True):
            image_col, table_col = st.columns([1, 3])
            with image_col:
                image_url = gallery_image_urls.get(species)
                if image_url:
                    st.image(image_url, width="stretch")
                st.subheader(species)
                st.caption(f"{len(species_df):,} detections")

            with table_col:
                best = quality_candidates(species_df, data_root, max_candidates=max_candidates)
                if best.empty:
                    st.info("No playable audio candidates found.")
                else:
                    display_best = best.drop(columns=["RowIndex"])
                    st.dataframe(
                        display_best,
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "Confidence": st.column_config.NumberColumn(format="%.3f"),
                            "Mean dBFS": st.column_config.NumberColumn(format="%.1f"),
                            "Background dBFS": st.column_config.NumberColumn(format="%.1f"),
                            "SNR dB": st.column_config.NumberColumn(format="%.1f"),
                            "Quality": st.column_config.NumberColumn(format="%.3f"),
                        },
                    )
                    best["Choice"] = (
                        best["Rank"].astype(str)
                        + " | "
                        + best["Date"]
                        + " | conf "
                        + best["Confidence"].round(3).astype(str)
                        + " | SNR "
                        + best["SNR dB"].round(1).astype(str)
                        + " dB"
                    )
                    selected_choice = st.selectbox(
                        "Play highlight",
                        best["Choice"],
                        key=f"gallery_{gallery_type}_{species_key}",
                    )
                    selected_rank = int(selected_choice.split(" | ")[0])
                    selected_row_index = best.loc[best["Rank"] == selected_rank, "RowIndex"].iloc[0]
                    render_audio_inspector(
                        species_df.loc[selected_row_index],
                        data_root,
                        gallery_type,
                        key_prefix=f"gallery_{gallery_type}_{species_key}_{selected_rank}",
                        context_df=species_df,
                    )
        gallery_progress.progress(
            species_index / len(selected_species),
            text=f"Loaded spectrograms: {species_index} / {len(selected_species)}",
        )

    gallery_progress.progress(1.0, text=f"Spectrograms ready: {len(selected_species)} / {len(selected_species)}")


def bird_highlight_filter(df):
    detectors = df["Detector"].fillna("birdnet").astype(str).str.lower()
    dataset_types = df["dataset_type"].fillna("birds").astype(str).str.lower()
    return (detectors == "birdnet") | ((detectors == "birda") & (dataset_types != "bats"))


def bat_highlight_filter(df):
    detectors = df["Detector"].fillna("birdnet").astype(str).str.lower()
    dataset_types = df["dataset_type"].fillna("birds").astype(str).str.lower()
    return (detectors == "batdetect2") | ((detectors == "birda") & (dataset_types == "bats"))


def split_command(command_text):
    return shlex.split(command_text, posix=os.name != "nt")


def without_managed_birdnet_args(args):
    managed_args = {"--lat", "--lon", "--week", "-o", "--output", "--i", "--input"}
    managed_prefixes = ("--lat=", "--lon=", "--week=", "--output=", "--i=", "--input=")
    cleaned = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in managed_args:
            skip_next = True
            continue
        if arg.startswith(managed_prefixes):
            continue
        cleaned.append(arg)
    return cleaned


def without_managed_birda_args(args):
    managed_args = {
        "--lat",
        "--lon",
        "--week",
        "-o",
        "--output-dir",
        "-f",
        "--format",
        "-c",
        "--min-confidence",
        "-b",
        "--batch-size",
        "-m",
        "--model",
        "--bat",
        "--output-mode",
        "--no-progress",
        "--gpu",
        "--cpu",
        "--tensorrt",
        "--force",
    }
    managed_prefixes = tuple(f"{arg}=" for arg in managed_args if arg.startswith("--"))
    cleaned = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in managed_args:
            if arg not in {"--gpu", "--cpu", "--tensorrt", "--force"}:
                skip_next = True
            continue
        if arg.startswith(managed_prefixes):
            continue
        cleaned.append(arg)
    return cleaned


def run_captured_command(command, cwd):
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=command_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def command_environment():
    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    return env


def format_seconds(seconds):
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def output_files(folder, suffixes):
    if not folder.exists() or not folder.is_dir():
        return []
    return [
        path
        for path in folder.rglob("*")
        if path.is_file() and any(path.name.endswith(suffix) for suffix in suffixes)
    ]


def output_unit_count(folder, suffixes):
    units = set()
    for path in output_files(folder, suffixes):
        name = path.name
        for suffix in sorted(suffixes, key=len, reverse=True):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        units.add(name)
    return len(units)


def audio_has_direct_outputs(audio_path, output_dir, suffixes):
    return any((output_dir / f"{audio_path.name}{suffix}").exists() for suffix in suffixes)


def analyzer_output_keys(output_dir, analyzer):
    if not output_dir.exists() or not output_dir.is_dir():
        return set()
    suffixes = {
        "birda": (".birdnet.results.csv", ".birdnet.results.json", ".birdnet.results.parquet"),
        "birdnet": (
            ".birdnet.selection.table.txt",
            ".birdnet.selection.table.csv",
            ".selection.table.txt",
            ".selection.table.csv",
        ),
        "batdetect2": (".wav.csv", ".wav.json", ".flac.csv", ".flac.json", ".mp3.csv", ".mp3.json"),
    }[analyzer]
    keys = set()
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.casefold()
        for suffix in suffixes:
            if name.endswith(suffix):
                key = name[: -len(suffix)]
                for audio_suffix in AUDIO_EXTENSIONS:
                    if key.endswith(audio_suffix.casefold()):
                        key = key[: -len(audio_suffix)]
                        break
                keys.add(key)
                break
    return keys


def pending_analyzer_files(audio_dir, output_dir, analyzer):
    outputs = analyzer_output_keys(output_dir, analyzer)
    pending = []
    for path in audio_files(audio_dir):
        key = path.stem.casefold()
        if key not in outputs:
            pending.append(path)
    return pending


@st.cache_data(ttl=10, show_spinner=False)
def processing_dataset_inventory(dataset_dir_text):
    dataset_dir = Path(dataset_dir_text)
    audio_dir = dataset_dir / "audio"
    audio_list = audio_files(audio_dir)
    audio_keys = [path.stem.casefold() for path in audio_list]
    output_dirs = {
        "birda": dataset_dir / "birda",
        "birdnet": dataset_dir / "birdnet",
        "batdetect2": dataset_dir / "batdetect2_outputs",
    }
    output_keys = {
        analyzer: analyzer_output_keys(output_dir, analyzer)
        for analyzer, output_dir in output_dirs.items()
    }
    pending = {
        analyzer: sum(1 for audio_key in audio_keys if audio_key not in output_keys[analyzer])
        for analyzer in output_dirs
    }
    return {
        "audio": len(audio_list),
        "birda_pending": pending["birda"],
        "birdnet_pending": pending["birdnet"],
        "batdetect2_pending": pending["batdetect2"],
        "birda_outputs": len(output_keys["birda"]),
        "birdnet_outputs": len(output_keys["birdnet"]),
        "batdetect2_outputs": len(output_keys["batdetect2"]),
    }


def manifest_dataset_ids_for_folder(manifest_path, dataset_name):
    if not manifest_path.exists():
        return []
    manifest = pd.read_csv(manifest_path)
    matching_ids = []
    for _, row in manifest.iterrows():
        audio_value = str(row.get("audio_dir") or "")
        if Path(audio_value).parent.name == dataset_name:
            matching_ids.append(str(row["dataset_id"]))
    return matching_ids


def database_detector_counts(db_path, dataset_ids):
    counts = {}
    present_ids = set()
    import_snapshots = {}
    if not db_path.exists() or not dataset_ids:
        return counts, present_ids, import_snapshots
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        con.register("status_dataset_ids", pd.DataFrame({"dataset_id": dataset_ids}))
        present_ids = set(
            con.execute(
                "SELECT dataset_id FROM datasets WHERE dataset_id IN (SELECT dataset_id FROM status_dataset_ids)"
            ).df()["dataset_id"].astype(str)
        )
        rows = con.execute(
            """
            SELECT lower(COALESCE(Detector, 'birdnet')) AS detector, count(*) AS detections
            FROM detections
            WHERE dataset_id IN (SELECT dataset_id FROM status_dataset_ids)
            GROUP BY 1
            """
        ).fetchall()
        counts = {str(detector): int(count) for detector, count in rows}
        tables = set(con.execute("SHOW TABLES").df()["name"].astype(str))
        if "dataset_import_status" in tables:
            snapshot_rows = con.execute(
                """
                SELECT detector, max(output_count) AS output_count
                FROM dataset_import_status
                WHERE dataset_id IN (SELECT dataset_id FROM status_dataset_ids)
                GROUP BY detector
                """
            ).fetchall()
            import_snapshots = {str(detector): int(count) for detector, count in snapshot_rows}
    finally:
        con.close()
    return counts, present_ids, import_snapshots


@st.cache_data(ttl=10, show_spinner=False)
def cached_audio_count(audio_dir_text):
    return count_audio_files(Path(audio_dir_text))


def parse_birda_event(line):
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict):
        return None
    event_name = event.get("event") or event.get("type")
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else event
    return event_name, payload


def update_birda_progress(event_name, payload, state):
    if not event_name:
        return False

    file_payload = payload.get("file") if isinstance(payload.get("file"), dict) else {}
    file_name = (
        file_payload.get("path")
        or file_payload.get("file")
        or payload.get("file")
        or payload.get("path")
        or payload.get("input")
        or payload.get("filename")
        or state.get("current_file")
    )
    if file_name:
        state["current_file"] = Path(str(file_name)).name

    total_files = payload.get("total_files") or payload.get("file_count") or payload.get("total")
    if total_files is not None:
        state["total_files"] = int(total_files)

    if event_name == "file_started" and payload.get("index") is not None:
        state["current_index"] = int(payload["index"])
        state["file_progress"] = 0.0

    if event_name in {"file_completed", "file_complete", "file_finished"}:
        completed_files = state.setdefault("completed_files", set())
        completed_key = str(file_name) if file_name else f"file-{len(completed_files) + 1}"
        completed_files.add(completed_key)
        state["done_files"] = len(completed_files)
        state["file_progress"] = 0.0

    if event_name == "pipeline_completed":
        processed = payload.get("files_processed", 0)
        skipped = payload.get("files_skipped", 0)
        state["done_files"] = int(processed) + int(skipped)
        state["file_progress"] = 0.0

    progress = payload.get("progress") or payload.get("fraction")
    if progress is not None:
        state["progress"] = float(progress)
    elif file_payload:
        percent = file_payload.get("percent")
        if percent is not None:
            state["file_progress"] = max(0.0, min(1.0, float(percent) / 100.0))

    detections = payload.get("detections") or payload.get("detection_count")
    if detections is not None:
        state["detections"] = int(detections)
    elif payload.get("total_detections") is not None:
        state["detections"] = int(payload["total_detections"])

    return event_name in {
        "pipeline_started",
        "file_started",
        "progress",
        "file_completed",
        "file_complete",
        "file_finished",
        "pipeline_completed",
        "result",
    }


def enqueue_stream_lines(stream, line_queue):
    try:
        for line in iter(stream.readline, ""):
            line_queue.put(line)
    finally:
        try:
            stream.close()
        except Exception:
            pass


def process_is_running(pid):
    if pid is None:
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return False
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def remove_stale_birda_locks(output_dir):
    removed = []
    for lock_path in output_dir.glob("*.birda.lock"):
        try:
            lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not process_is_running(lock_data.get("pid")):
            lock_path.unlink()
            removed.append(lock_path.name)
    return removed


def run_monitored_command(command, cwd, label, audio_dir, output_dir, output_suffixes):
    audio_list = audio_files(audio_dir)
    is_birda = any(Path(str(part)).name.lower().startswith("birda") for part in command[:2])
    direct_output_monitor = set(output_suffixes) == {".csv", ".json"}
    if direct_output_monitor:
        monitor_audio_list = [
            path for path in audio_list if not audio_has_direct_outputs(path, output_dir, output_suffixes)
        ]
    else:
        monitor_audio_list = audio_list
    total_files = len(audio_list)
    initial_outputs = output_unit_count(output_dir, output_suffixes)
    expected_new_outputs = len(monitor_audio_list) if direct_output_monitor else max(0, total_files - initial_outputs)

    progress_bar = st.progress(0.0, text=f"{label}: starting...")
    status_box = st.empty()
    log_box = st.empty()

    stderr_file = tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False)
    stderr_path = Path(stderr_file.name)

    started = time.time()
    stdout_lines = []
    birda_state = {"done_files": 0, "total_files": expected_new_outputs or total_files, "file_progress": 0.0}
    line_queue = queue.Queue()
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=command_environment(),
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
            bufsize=1,
        )
        if process.stdout:
            threading.Thread(target=enqueue_stream_lines, args=(process.stdout, line_queue), daemon=True).start()

        while process.poll() is None:
            while True:
                try:
                    line = line_queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    stdout_lines.append(line)
                    parsed = parse_birda_event(line) if is_birda else None
                    if parsed:
                        update_birda_progress(parsed[0], parsed[1], birda_state)

            current_outputs = output_unit_count(output_dir, output_suffixes)
            new_outputs = max(0, current_outputs - initial_outputs)
            monitor_total = expected_new_outputs if expected_new_outputs else total_files
            done_files = min(new_outputs, monitor_total) if expected_new_outputs else min(current_outputs, total_files)
            elapsed = time.time() - started

            if is_birda and birda_state.get("total_files"):
                birda_total = int(birda_state.get("total_files") or monitor_total or 1)
                birda_done = min(int(birda_state.get("done_files", 0)), birda_total)
                fraction = float(
                    birda_state.get(
                        "progress",
                        (birda_done + float(birda_state.get("file_progress", 0.0))) / birda_total,
                    )
                )
                fraction = max(0.0, min(1.0, fraction))
                current_name = birda_state.get("current_file") or (monitor_audio_list[0].name if monitor_audio_list else "unknown")
                eta = None
                if birda_done > 0:
                    remaining = max(0, birda_total - birda_done)
                    eta = remaining * (elapsed / birda_done)
                progress_bar.progress(fraction, text=f"{label}: {birda_done}/{birda_total} files")
                details = f"Current file: {current_name}\n\nElapsed: {format_seconds(elapsed)} | Remaining: {format_seconds(eta)}"
                if birda_state.get("detections") is not None:
                    details += f"\n\nDetections reported: {birda_state['detections']}"
                status_box.info(details)
            elif monitor_total:
                fraction = min(1.0, done_files / monitor_total)
                current_index = min(done_files, len(monitor_audio_list) - 1)
                current_name = monitor_audio_list[current_index].name if monitor_audio_list else "unknown"
                eta = None
                if new_outputs > 0:
                    remaining = max(0, monitor_total - done_files)
                    seconds_per_file = elapsed / new_outputs
                    eta = remaining * seconds_per_file
                progress_bar.progress(
                    fraction,
                    text=f"{label}: {done_files}/{monitor_total} new files",
                )
                status_box.info(
                    f"Likely current file: {current_name}\n\n"
                    f"Elapsed: {format_seconds(elapsed)} | Remaining: {format_seconds(eta)}"
                )
            else:
                progress_bar.progress(0.0, text=f"{label}: running...")
                status_box.info(f"Elapsed: {format_seconds(elapsed)}")

            time.sleep(1.0)

        returncode = process.wait()
        while True:
            try:
                stdout_lines.append(line_queue.get_nowait())
            except queue.Empty:
                break
        stderr_file.close()

        stdout = "".join(stdout_lines)
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")

        final_outputs = output_unit_count(output_dir, output_suffixes)
        final_new_outputs = max(0, final_outputs - initial_outputs)
        monitor_total = expected_new_outputs if expected_new_outputs else total_files
        final_done = min(final_new_outputs, monitor_total) if expected_new_outputs else min(final_outputs, total_files)
        if monitor_total:
            progress_bar.progress(
                min(1.0, final_done / monitor_total),
                text=f"{label}: {final_done}/{monitor_total} new files",
            )
        if returncode == 0:
            status_box.success(f"{label} finished. Output files now found: {final_outputs}.")
        else:
            status_box.error(f"{label} failed with exit code {returncode}.")

        return returncode, stdout, stderr
    finally:
        try:
            stderr_file.close()
        except Exception:
            pass
        for path in (stderr_path,):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass


def render_birda_bird_runner(audio_dir, birda_output_dir, birdnet_lat, birdnet_lon, birdnet_week):
    inventory = processing_dataset_inventory(str(audio_dir.parent.resolve()))
    audio_total = inventory["audio"]
    pending_total = inventory["birda_pending"]
    st.caption(f"New files not yet analyzed by Birda: {pending_total:,} of {audio_total:,}")
    birda_available = shutil.which("birda")
    birda_command = st.text_input("Birda command", value=birda_available or "birda", key="bird_birda_command")

    def use_recommended_birda_batch_size():
        device = st.session_state.get("bird_birda_device", "Auto")
        st.session_state["bird_birda_batch_size"] = 32 if device == "TensorRT" else 8 if device == "CPU" else 256

    if "bird_birda_batch_size" not in st.session_state:
        use_recommended_birda_batch_size()
    birda_cols = st.columns(3)
    with birda_cols[0]:
        birda_device = st.selectbox(
            "Device",
            ["Auto", "GPU", "TensorRT", "CPU"],
            key="bird_birda_device",
            on_change=use_recommended_birda_batch_size,
            help="GPU uses CUDA. TensorRT is fastest when the TensorRT DLL folder is on PATH or copied beside birda.exe.",
        )
    with birda_cols[1]:
        birda_confidence = st.number_input(
            "Minimum confidence",
            min_value=0.01,
            max_value=0.99,
            value=0.30,
            step=0.05,
            key="bird_birda_confidence",
        )
    with birda_cols[2]:
        birda_batch_size = st.number_input(
            "Batch size",
            min_value=1,
            max_value=512,
            step=8,
            key="bird_birda_batch_size",
        )
    st.caption("The batch size switches automatically with the device: CUDA/GPU 256, TensorRT 32, CPU 8.")

    birda_model = st.text_input(
        "Birda model",
        value="birdnet-v24",
        key="bird_birda_model",
        help="Use an installed Birda model such as birdnet-v24, birdnet-v30, or perch-v2.",
    )
    birda_force = st.checkbox(
        "Re-run files that already have Birda outputs",
        value=False,
        key="bird_birda_force",
        help="Leave this off when adding new WAV files. Birda skips existing outputs unless --force is used.",
    )
    birda_extra_args = st.text_input("Birda extra arguments", value="", key="bird_birda_extra")

    if not birda_available:
        st.warning("Birda is not currently on PATH. Install the Windows CUDA Birda release, then restart the dashboard.")

    if st.button("Run Birda bird detection"):
        birda_output_dir.mkdir(parents=True, exist_ok=True)
        removed_locks = remove_stale_birda_locks(birda_output_dir)
        if removed_locks:
            st.info(f"Removed {len(removed_locks)} stale Birda lock file(s) before starting.")
        extra_args = without_managed_birda_args(split_command(birda_extra_args))
        command = split_command(birda_command) + [
            "--output-mode",
            "ndjson",
            "--no-progress",
            "--output-dir",
            str(birda_output_dir),
            "--format",
            "csv",
            "--min-confidence",
            str(birda_confidence),
            "--batch-size",
            str(int(birda_batch_size)),
            "--lat",
            f"{birdnet_lat:.6f}",
            "--lon",
            f"{birdnet_lon:.6f}",
            "--week",
            str(birdnet_week),
        ]
        if birda_device == "GPU":
            command.append("--gpu")
        elif birda_device == "TensorRT":
            command.append("--tensorrt")
        elif birda_device == "CPU":
            command.append("--cpu")
        if birda_model.strip():
            command.extend(["--model", birda_model.strip()])
        if birda_force:
            command.append("--force")
        command += extra_args + [str(audio_dir)]

        st.code(" ".join(f'"{part}"' if " " in part else part for part in command), language="powershell")
        returncode, stdout, stderr = run_monitored_command(
            command,
            PROJECT_ROOT,
            "Birda birds",
            audio_dir,
            birda_output_dir,
            (".csv", ".json", ".parquet"),
        )
        processing_dataset_inventory.clear()
        show_command_result(returncode, stdout, stderr)


def render_birdnet_analyzer_runner(audio_dir, birdnet_output_dir, birdnet_lat, birdnet_lon, birdnet_week):
    inventory = processing_dataset_inventory(str(audio_dir.parent.resolve()))
    audio_total = inventory["audio"]
    pending_total = inventory["birdnet_pending"]
    st.caption(f"New files not yet analyzed by BirdNET Analyzer: {pending_total:,} of {audio_total:,}")
    birdnet_available = (
        shutil.which("birdnet-analyze")
        or shutil.which("birdnet-analyzer")
        or shutil.which("birdnet_analyzer")
    )
    birdnet_command = st.text_input("BirdNET command", value=birdnet_available or "birdnet-analyze", key="birdnet_command")
    birdnet_extra_args = st.text_input("BirdNET extra arguments", value="", key="birdnet_extra_args")
    rerun_birdnet = st.checkbox(
        "Re-run files that already have BirdNET outputs",
        value=False,
        key="birdnet_rerun_existing",
        help="Leave this off when adding WAV files. BirdNET will process only files without an existing result.",
    )
    st.caption("The app runs: BirdNET command + location + week + extra arguments + `-o <birdnet folder> <audio folder>`.")

    if st.button("Run BirdNET Analyzer"):
        birdnet_output_dir.mkdir(parents=True, exist_ok=True)
        extra_args = without_managed_birdnet_args(split_command(birdnet_extra_args))
        command = split_command(birdnet_command) + [
            "--lat",
            f"{birdnet_lat:.6f}",
            "--lon",
            f"{birdnet_lon:.6f}",
            "--week",
            str(birdnet_week),
        ]
        if not rerun_birdnet:
            command.append("--skip_existing_results")
        command += extra_args + [
            "-o",
            str(birdnet_output_dir),
            str(audio_dir),
        ]
        st.code(" ".join(f'"{part}"' if " " in part else part for part in command), language="powershell")
        returncode, stdout, stderr = run_monitored_command(
            command,
            PROJECT_ROOT,
            "BirdNET Analyzer",
            audio_dir,
            birdnet_output_dir,
            (
                ".BirdNET.selection.table.txt",
                ".BirdNET.selection.table.csv",
                ".selection.table.txt",
                ".selection.table.csv",
                ".csv",
                ".txt",
            ),
        )
        processing_dataset_inventory.clear()
        show_command_result(returncode, stdout, stderr)


def render_batdetect_runner(audio_dir, batdetect_output_dir):
    inventory = processing_dataset_inventory(str(audio_dir.parent.resolve()))
    audio_total = inventory["audio"]
    pending_total = inventory["batdetect2_pending"]
    st.caption(f"New files not yet analyzed by BatDetect2: {pending_total:,} of {audio_total:,}")
    bat_cols = st.columns(2)
    with bat_cols[0]:
        bat_threshold = st.number_input("BatDetect2 threshold", min_value=0.01, max_value=0.99, value=0.30, step=0.05)
    with bat_cols[1]:
        bat_model = st.text_input("BatDetect2 model checkpoint", value="")
    rerun_batdetect = st.checkbox(
        "Re-run files that already have BatDetect2 outputs",
        value=False,
        help=(
            "Leave this off when you add new WAV files to an existing folder. "
            "The wrapper skips audio files that already have matching .csv or .json BatDetect2 outputs."
        ),
    )
    st.caption("BatDetect2 skips already analyzed files by default. Turn on re-run only when you want to recompute existing outputs.")

    if st.button("Run BatDetect2"):
        batdetect_output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(SCRIPTS_DIR / "run_batdetect2.py"),
            "--audio-dir",
            str(audio_dir),
            "--batdetect-output-dir",
            str(batdetect_output_dir),
            "--detection-threshold",
            str(bat_threshold),
        ]
        if bat_model.strip():
            command.extend(["--model", bat_model.strip()])
        if rerun_batdetect:
            command.append("--rerun-existing")
        st.code(" ".join(f'"{part}"' if " " in part else part for part in command), language="powershell")
        returncode, stdout, stderr = run_monitored_command(
            command,
            PROJECT_ROOT,
            "BatDetect2",
            audio_dir,
            batdetect_output_dir,
            (".csv", ".json"),
        )
        processing_dataset_inventory.clear()
        show_command_result(returncode, stdout, stderr)


def clip_filename_timestamp(row, start):
    base_time = pd.to_datetime(row["DateTime"]) - pd.to_timedelta(float(row["BeginTime"]), unit="s")
    clip_time = base_time + pd.to_timedelta(start, unit="s")
    return clip_time.strftime("%Y%m%d_%H%M%S")


def clean_clip_species_name(value):
    value = str(value or "Unknown").strip()
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return value[:80] or "Unknown"


def merge_clip_groups(detections, pad_seconds):
    groups = []
    if detections.empty:
        return groups
    for rel_audio_path, file_df in detections.groupby("relative_audio_path", sort=True):
        file_df = file_df.sort_values(["BeginTime", "EndTime"]).copy()
        active = None
        for _, row in file_df.iterrows():
            start = max(0.0, float(row["BeginTime"]) - pad_seconds)
            end = float(row["EndTime"]) + pad_seconds
            if active is None or start > active["end"]:
                if active is not None:
                    groups.append(active)
                active = {"relative_audio_path": rel_audio_path, "start": start, "end": end, "rows": [row]}
            else:
                active["end"] = max(active["end"], end)
                active["rows"].append(row)
        if active is not None:
            groups.append(active)
    return groups


def create_detection_clips(
    detections,
    data_root,
    dataset_dir,
    min_confidence,
    pad_seconds,
    max_clips,
    overwrite=False,
    progress_bar=None,
    status_box=None,
):
    clips_dir = dataset_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = clips_dir / "clips_manifest.csv"

    selected = detections[detections["Confidence"] >= min_confidence].copy()
    selected = selected.dropna(subset=["relative_audio_path", "BeginTime", "EndTime"])
    if selected.empty:
        return {"created": 0, "skipped": 0, "failed": 0, "manifest": manifest_path}

    selected["relative_audio_path"] = selected["relative_audio_path"].astype(str)
    clip_groups = merge_clip_groups(selected, float(pad_seconds))
    if max_clips:
        clip_groups = clip_groups[: int(max_clips)]
    total_groups = len(clip_groups)

    manifest_rows = []
    if manifest_path.exists() and not overwrite:
        try:
            manifest_rows = pd.read_csv(manifest_path).to_dict("records")
        except Exception:
            manifest_rows = []

    created = 0
    skipped = 0
    failed = 0
    started = time.time()
    for index, group in enumerate(clip_groups, start=1):
        rows = group["rows"]
        rel_audio = group["relative_audio_path"]
        elapsed = time.time() - started
        eta = None
        if index > 1:
            eta = ((total_groups - index + 1) * elapsed) / (index - 1)
        if progress_bar is not None:
            progress_bar.progress(
                index / total_groups if total_groups else 0.0,
                text=f"Creating clips: {index:,}/{total_groups:,}",
            )
        if status_box is not None:
            status_box.info(
                f"Current source: {Path(str(rel_audio)).name}\n\n"
                f"Created: {created:,} | Skipped: {skipped:,} | Failed: {failed:,}\n\n"
                f"Elapsed: {format_seconds(elapsed)} | Remaining: {format_seconds(eta)}"
            )

        source_path = Path(rel_audio)
        if not source_path.is_absolute():
            source_path = data_root / source_path
        if not source_path.exists():
            failed += 1
            continue

        try:
            with sf.SoundFile(str(source_path)) as audio_file:
                sr = audio_file.samplerate
                total_duration = audio_file.frames / sr if sr else 0
                start = max(0.0, min(group["start"], total_duration))
                end = max(start, min(group["end"], total_duration))
                duration = end - start
                if duration <= 0:
                    failed += 1
                    continue
                audio_file.seek(int(start * sr))
                y = audio_file.read(frames=int(duration * sr), dtype="float32", always_2d=False)
        except Exception:
            failed += 1
            continue

        species = sorted({str(row.get("CommonName") or "Unknown") for row in rows})
        detectors = sorted({str(row.get("Detector") or "unknown") for row in rows})
        first_row = rows[0]
        species_part = clean_clip_species_name("_".join(species[:3]))
        if len(species) > 3:
            species_part += "_multi"
        clip_name = f"{clip_filename_timestamp(first_row, start)}_{int(start):04d}-{int(end):04d}_{species_part}.wav"
        clip_path = clips_dir / clip_name

        if clip_path.exists() and not overwrite:
            skipped += 1
        else:
            sf.write(str(clip_path), y, sr)
            created += 1

        mean_values = [row.get("MeanDbfs") for row in rows if pd.notna(row.get("MeanDbfs"))]
        noise_values = [row.get("NoiseDbfs") for row in rows if pd.notna(row.get("NoiseDbfs"))]
        snr_values = [row.get("SnrDb") for row in rows if pd.notna(row.get("SnrDb"))]
        manifest_rows.append(
            {
                "clip_file": clip_path.relative_to(dataset_dir).as_posix(),
                "source_audio_path": rel_audio,
                "clip_start_seconds": round(start, 3),
                "clip_end_seconds": round(end, 3),
                "clip_duration_seconds": round(duration, 3),
                "species": "; ".join(species),
                "detectors": "; ".join(detectors),
                "detections": len(rows),
                "max_confidence": max(float(row.get("Confidence") or 0) for row in rows),
                "min_mean_dbfs": min(mean_values) if mean_values else np.nan,
                "quietest_background_dbfs": min(noise_values) if noise_values else np.nan,
                "max_snr_db": max(snr_values) if snr_values else np.nan,
            }
        )

    if manifest_rows:
        manifest_df = pd.DataFrame(manifest_rows).drop_duplicates(subset=["clip_file"], keep="last")
        manifest_df.to_csv(manifest_path, index=False)

    if progress_bar is not None:
        progress_bar.progress(1.0, text=f"Creating clips: {total_groups:,}/{total_groups:,}")
    if status_box is not None:
        status_box.success(f"Clip export finished. Created {created:,}, skipped {skipped:,}, failed {failed:,}.")

    return {"created": created, "skipped": skipped, "failed": failed, "manifest": manifest_path}


def select_quality_clip_detections(
    detections,
    detectors,
    min_confidence,
    min_snr_db,
    max_noise_dbfs,
    per_species,
):
    selected = detections[detections["Detector"].fillna("birdnet").astype(str).isin(detectors)].copy()
    selected["Confidence"] = pd.to_numeric(selected["Confidence"], errors="coerce")
    selected["SnrDb"] = pd.to_numeric(selected["SnrDb"], errors="coerce")
    selected["NoiseDbfs"] = pd.to_numeric(selected["NoiseDbfs"], errors="coerce")
    selected = selected.dropna(subset=["Confidence", "SnrDb", "NoiseDbfs", "CommonName"])
    selected = selected[
        (selected["Confidence"] >= float(min_confidence))
        & (selected["SnrDb"] >= float(min_snr_db))
        & (selected["NoiseDbfs"] <= float(max_noise_dbfs))
    ]
    selected = selected.drop_duplicates(
        subset=["Detector", "relative_audio_path", "CommonName", "BeginTime", "EndTime", "Confidence"]
    )
    if selected.empty:
        return selected

    selected["_confidence_rank"] = selected.groupby("CommonName")["Confidence"].rank(pct=True)
    selected["_snr_rank"] = selected.groupby("CommonName")["SnrDb"].rank(pct=True)
    selected["_quiet_rank"] = (-selected["NoiseDbfs"]).groupby(selected["CommonName"]).rank(pct=True)
    selected["QualityScore"] = (
        (0.50 * selected["_confidence_rank"])
        + (0.35 * selected["_snr_rank"])
        + (0.15 * selected["_quiet_rank"])
    )
    selected = selected.sort_values(
        ["CommonName", "QualityScore", "Confidence", "SnrDb", "NoiseDbfs"],
        ascending=[True, False, False, False, True],
    )
    if per_species:
        selected = selected.groupby("CommonName", group_keys=False).head(int(per_species))
    return selected.drop(columns=["_confidence_rank", "_snr_rank", "_quiet_rank"])


@st.cache_data(ttl=300, show_spinner=False)
def source_audio_size(audio_dir):
    audio_dir = Path(audio_dir)
    return sum(
        path.stat().st_size
        for path in audio_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    )


def estimate_clip_export_size(clip_groups, data_root, max_clips=None):
    groups = clip_groups[: int(max_clips)] if max_clips else clip_groups
    estimated_bytes = 0.0
    estimated_seconds = 0.0
    readable_groups = 0
    for group in groups:
        source_path = Path(str(group["relative_audio_path"]))
        if not source_path.is_absolute():
            source_path = data_root / source_path
        if not source_path.exists():
            continue
        try:
            info = sf.info(str(source_path))
            clip_start = max(0.0, min(float(group["start"]), info.duration))
            clip_end = max(clip_start, min(float(group["end"]), info.duration))
            clip_duration = clip_end - clip_start
            if info.duration <= 0 or clip_duration <= 0:
                continue
            estimated_seconds += clip_duration
            estimated_bytes += source_path.stat().st_size * (clip_duration / info.duration)
            readable_groups += 1
        except Exception:
            continue
    return estimated_bytes, estimated_seconds, readable_groups


def render_clip_snr_analytics(detections, detectors, min_confidence, min_snr_db, max_noise_dbfs):
    plot_df = detections[detections["Detector"].fillna("birdnet").astype(str).isin(detectors)].copy()
    for column in ("Confidence", "MeanDbfs", "SignalDbfs", "NoiseDbfs", "SnrDb"):
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df["DateTime"] = pd.to_datetime(plot_df["DateTime"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Confidence", "NoiseDbfs", "SnrDb", "DateTime"])
    plot_df = plot_df.drop_duplicates(
        subset=["Detector", "relative_audio_path", "CommonName", "BeginTime", "EndTime", "Confidence"]
    )
    if plot_df.empty:
        st.info("Compute the missing SNR metrics from the Database tab to enable clip analytics.")
        return

    passes = (
        (plot_df["Confidence"] >= float(min_confidence))
        & (plot_df["SnrDb"] >= float(min_snr_db))
        & (plot_df["NoiseDbfs"] <= float(max_noise_dbfs))
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric("Measured detections", f"{len(plot_df):,}")
    metric_cols[1].metric("Median SNR", f"{plot_df['SnrDb'].median():.1f} dB")
    metric_cols[2].metric("Median background", f"{plot_df['NoiseDbfs'].median():.1f} dBFS")
    metric_cols[3].metric("Passing quality filters", f"{passes.mean() * 100:.1f}%")

    snr_tab, timeline_tab = st.tabs(["Confidence And SNR", "Night Timeline"])
    with snr_tab:
        display_df = plot_df.sample(10000, random_state=42) if len(plot_df) > 10000 else plot_df
        fig = px.scatter(
            display_df,
            x="Confidence",
            y="SnrDb",
            color="NoiseDbfs",
            symbol="Detector",
            hover_data=["CommonName", "DateTime", "SignalDbfs", "MeanDbfs"],
            labels={
                "SnrDb": "Signal-to-noise ratio (dB)",
                "NoiseDbfs": "Background (dBFS)",
                "SignalDbfs": "Signal (dBFS)",
            },
            color_continuous_scale="Viridis_r",
            render_mode="webgl",
            opacity=0.5,
        )
        fig.add_vline(x=float(min_confidence), line_dash="dash", line_color="#e4572e")
        fig.add_hline(y=float(min_snr_db), line_dash="dash", line_color="#2a9d8f")
        fig.update_layout(height=520, margin={"l": 10, "r": 10, "t": 20, "b": 10})
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "The upper-right area contains confident detections that stand clearly above their background. "
            "Color distinguishes quiet backgrounds from loud ones."
        )

    with timeline_tab:
        timeline = plot_df.copy()
        timeline["TimeBin"] = timeline["DateTime"].dt.floor("15min")
        timeline = (
            timeline.groupby("TimeBin", as_index=False)
            .agg(
                MeanDbfs=("MeanDbfs", "median"),
                SignalDbfs=("SignalDbfs", "median"),
                NoiseDbfs=("NoiseDbfs", "median"),
                SnrDb=("SnrDb", "median"),
                Detections=("Confidence", "size"),
            )
        )
        level_data = timeline.melt(
            id_vars=["TimeBin"],
            value_vars=["SignalDbfs", "NoiseDbfs", "MeanDbfs"],
            var_name="Level",
            value_name="dBFS",
        )
        level_fig = px.line(
            level_data,
            x="TimeBin",
            y="dBFS",
            color="Level",
            markers=True,
            color_discrete_map={"SignalDbfs": "#e4572e", "NoiseDbfs": "#2a9d8f", "MeanDbfs": "#4c78a8"},
        )
        level_fig.add_hline(y=float(max_noise_dbfs), line_dash="dash", line_color="#202124")
        level_fig.update_layout(height=400, xaxis_title="Recording time", margin={"l": 10, "r": 10, "t": 20, "b": 10})
        st.plotly_chart(level_fig, width="stretch")

        snr_fig = px.bar(
            timeline,
            x="TimeBin",
            y="SnrDb",
            color="Detections",
            labels={"SnrDb": "Median SNR (dB)", "TimeBin": "Recording time"},
            color_continuous_scale="Viridis",
        )
        snr_fig.add_hline(y=float(min_snr_db), line_dash="dash", line_color="#e4572e")
        snr_fig.update_layout(height=340, margin={"l": 10, "r": 10, "t": 20, "b": 10})
        st.plotly_chart(snr_fig, width="stretch")
        st.caption(
            "Values are 15-minute medians across detection windows. This describes detected clips, not continuous ambient sound."
        )


def render_clip_quality_plot(detections, detectors, min_confidence):
    plot_df = detections[detections["Detector"].fillna("birdnet").astype(str).isin(detectors)].copy()
    plot_df["Confidence"] = pd.to_numeric(plot_df["Confidence"], errors="coerce")
    plot_df["MeanDbfs"] = pd.to_numeric(plot_df["MeanDbfs"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Confidence", "MeanDbfs"])
    plot_df = plot_df.drop_duplicates(
        subset=["Detector", "relative_audio_path", "CommonName", "BeginTime", "EndTime", "Confidence"]
    )
    if len(plot_df) < 2 or plot_df["Confidence"].nunique() < 2:
        st.info("At least two distinct confidence values with audio metrics are needed for the quality plot.")
        return

    pearson = plot_df["Confidence"].corr(plot_df["MeanDbfs"], method="pearson")
    spearman = plot_df["Confidence"].corr(plot_df["MeanDbfs"], method="spearman")
    slope, intercept = np.polyfit(plot_df["Confidence"], plot_df["MeanDbfs"], 1)
    line_x = np.linspace(plot_df["Confidence"].min(), plot_df["Confidence"].max(), 100)

    bin_edges = np.linspace(plot_df["Confidence"].min(), plot_df["Confidence"].max(), 16)
    plot_df["_confidence_bin"] = pd.cut(plot_df["Confidence"], bins=bin_edges, include_lowest=True)
    binned = (
        plot_df.groupby("_confidence_bin", observed=True)
        .agg(Confidence=("Confidence", "median"), MeanDbfs=("MeanDbfs", "median"))
        .dropna()
    )

    display_df = plot_df
    if len(display_df) > 10000:
        display_df = display_df.sample(10000, random_state=42)
    fig = px.scatter(
        display_df,
        x="Confidence",
        y="MeanDbfs",
        color="Detector",
        hover_data=["CommonName"],
        labels={"MeanDbfs": "Mean level (dBFS)"},
        render_mode="webgl",
        opacity=0.35,
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.add_scatter(
        x=line_x,
        y=(slope * line_x) + intercept,
        mode="lines",
        name="Linear fit",
        line={"color": "#202124", "width": 2},
    )
    fig.add_scatter(
        x=binned["Confidence"],
        y=binned["MeanDbfs"],
        mode="lines+markers",
        name="Binned median",
        line={"color": "#e4572e", "width": 3},
    )
    fig.add_vline(x=float(min_confidence), line_dash="dash", line_color="#2a9d8f")
    fig.update_layout(height=500, margin={"l": 10, "r": 10, "t": 20, "b": 10})
    st.plotly_chart(fig, width="stretch")

    correlation_cols = st.columns(3)
    correlation_cols[0].metric("Pearson correlation", f"{pearson:.3f}", help="Linear relationship; values near zero indicate little linear association.")
    correlation_cols[1].metric("Spearman correlation", f"{spearman:.3f}", help="Monotonic relationship, including curved relationships.")
    correlation_cols[2].metric("Linear R-squared", f"{pearson ** 2:.3f}")
    detector_correlations = []
    for detector, detector_df in plot_df.groupby("Detector"):
        detector_pearson = detector_df["Confidence"].corr(detector_df["MeanDbfs"], method="pearson")
        detector_spearman = detector_df["Confidence"].corr(detector_df["MeanDbfs"], method="spearman")
        detector_correlations.append(
            {
                "Detector": detector,
                "Detections": len(detector_df),
                "Pearson": detector_pearson,
                "Spearman": detector_spearman,
                "Linear R-squared": detector_pearson ** 2,
            }
        )
    st.dataframe(
        pd.DataFrame(detector_correlations),
        width="stretch",
        hide_index=True,
        column_config={
            "Pearson": st.column_config.NumberColumn(format="%.3f"),
            "Spearman": st.column_config.NumberColumn(format="%.3f"),
            "Linear R-squared": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption(
        "The dashed line shows the active confidence filter. The orange binned median helps reveal curvature "
        "that the straight linear fit may miss. Correlations are also reported per detector because analyzer behavior differs."
    )


def render_clip_creator(data_root, resolved_db_path, dataset_ids, dataset_dir):
    clips_dir = dataset_dir / "clips"
    existing_clips = count_files_with_suffixes(clips_dir, (".wav", ".WAV"))
    st.caption(f"Clip output folder: {clips_dir}")
    st.caption(f"Existing clips: {existing_clips}")

    if not resolved_db_path.exists():
        st.warning("Build or update the database before creating clips. Clips are generated from database detections.")
        return

    try:
        con = duckdb.connect(str(resolved_db_path), read_only=True)
        detections = load_platform_detections(con, dataset_ids=dataset_ids)
        con.close()
    except Exception as exc:
        st.warning(f"Could not load detections for clipping: {exc}")
        return

    if detections.empty:
        st.info("No database detections found for this audio folder yet. Regenerate the manifest and rebuild/append the database first.")
        return

    detectors = sorted(detections["Detector"].fillna("birdnet").astype(str).unique())
    dataset_summary = ", ".join(str(dataset_id) for dataset_id in dataset_ids)
    st.caption(f"Using detections from database dataset(s): {dataset_summary}")
    selected_detectors = checkbox_selection("Detectors to clip", detectors, detectors, "clip_detector")
    quality_presets = {
        "Balanced library (recommended)": {
            "confidence": 0.20,
            "snr": 3.0,
            "noise": -25.0,
            "per_species": 20,
            "description": "A useful listening library with moderate quality filtering and broader species coverage.",
        },
        "Curated highlights": {
            "confidence": 0.50,
            "snr": 10.0,
            "noise": -35.0,
            "per_species": 5,
            "description": "A small showcase of clear, confident detections recorded against quiet backgrounds.",
        },
        "Broad collection": {
            "confidence": 0.15,
            "snr": 0.0,
            "noise": -20.0,
            "per_species": 0,
            "description": "Keeps nearly every plausible detection whose signal is at least as loud as its background.",
        },
        "Custom": None,
    }
    quality_preset = st.selectbox("Clip collection", list(quality_presets), key="clip_quality_preset")
    preset = quality_presets[quality_preset]
    if preset is not None:
        min_confidence = preset["confidence"]
        min_snr_db = preset["snr"]
        max_noise_dbfs = preset["noise"]
        per_species = preset["per_species"]
        cap_text = "all" if not per_species else str(per_species)
        st.info(
            f"{preset['description']} Confidence >= {min_confidence:.2f}, SNR >= {min_snr_db:.0f} dB, "
            f"background <= {max_noise_dbfs:.0f} dBFS, up to {cap_text} detections per species."
        )
    else:
        custom_cols = st.columns(4)
        with custom_cols[0]:
            min_confidence = st.number_input(
                "Minimum confidence", min_value=0.01, max_value=0.99, value=0.20, step=0.05, key="clip_custom_confidence"
            )
        with custom_cols[1]:
            min_snr_db = st.number_input(
                "Minimum SNR (dB)", min_value=-20.0, max_value=80.0, value=3.0, step=1.0, key="clip_custom_snr"
            )
        with custom_cols[2]:
            max_noise_dbfs = st.number_input(
                "Maximum background dBFS", min_value=-100.0, max_value=0.0, value=-25.0, step=1.0, key="clip_custom_noise"
            )
        with custom_cols[3]:
            per_species = st.number_input(
                "Best per species", min_value=0, max_value=1000, value=20, step=1, help="Use 0 for all passing detections.", key="clip_custom_per_species"
            )

    estimate_container = st.container()

    with st.expander("Output options"):
        output_cols = st.columns(3)
        with output_cols[0]:
            pad_seconds = st.number_input("Margin before/after", min_value=0.0, max_value=30.0, value=3.0, step=1.0, key="clip_pad_seconds")
        with output_cols[1]:
            max_clips = st.number_input("Maximum clips", min_value=0, max_value=100000, value=0, step=100, help="Use 0 for no limit.", key="clip_max_clips")
        with output_cols[2]:
            overwrite = st.checkbox("Overwrite matching clips", value=False, key="clip_overwrite")

    with st.expander("Signal, noise, and nightly patterns", expanded=True):
        render_clip_snr_analytics(
            detections,
            selected_detectors,
            min_confidence,
            min_snr_db,
            max_noise_dbfs,
        )

    with st.expander("Confidence versus mean recording level"):
        render_clip_quality_plot(detections, selected_detectors, min_confidence)

    missing_metrics = int(detections[["SnrDb", "NoiseDbfs"]].isna().any(axis=1).sum())
    if missing_metrics:
        st.warning(
            f"{missing_metrics:,} detections have no SNR metric and cannot be quality-ranked. "
            "Use Compute missing audio quality metrics in the Database tab."
        )

    eligible_clip_df = select_quality_clip_detections(
        detections,
        selected_detectors,
        min_confidence,
        min_snr_db,
        max_noise_dbfs,
        None,
    )
    clip_df = select_quality_clip_detections(
        detections,
        selected_detectors,
        min_confidence,
        min_snr_db,
        max_noise_dbfs,
        int(per_species) if per_species else None,
    )
    clip_groups = merge_clip_groups(clip_df, float(pad_seconds)) if not clip_df.empty else []
    limited_groups = clip_groups[: int(max_clips)] if max_clips else clip_groups
    species_count = clip_df["CommonName"].nunique() if not clip_df.empty else 0
    projected_bytes, projected_seconds, readable_groups = estimate_clip_export_size(
        clip_groups,
        data_root,
        int(max_clips) if max_clips else None,
    )
    source_bytes = source_audio_size(str(dataset_dir / "audio"))
    reduction_percent = 100 * (1 - (projected_bytes / source_bytes)) if source_bytes else 0.0
    with estimate_container:
        st.caption(
            f"{len(eligible_clip_df):,} detections pass the quality thresholds. The per-species selection keeps "
            f"{len(clip_df):,} detections across {species_count:,} species "
            f"and would create/update up to {len(limited_groups):,} merged clips."
        )
        estimate_cols = st.columns(3)
        estimate_cols[0].metric("Source WAV collection", f"{source_bytes / (1024 ** 3):.2f} GB")
        estimate_cols[1].metric(
            "Estimated quality clips",
            f"{projected_bytes / (1024 ** 3):.2f} GB",
            help=f"Approximately {projected_seconds / 60:.1f} minutes across {readable_groups:,} readable clip groups.",
        )
        estimate_cols[2].metric("Estimated reduction", f"{max(0.0, reduction_percent):.1f}%")
        st.caption("Estimate assumes a clean quality-only clip set; existing older clips remain on disk until removed separately.")

    if st.button("Create detection clips"):
        progress_bar = st.progress(0.0, text="Creating clips: starting...")
        status_box = st.empty()
        with st.spinner("Creating clipped playback WAVs..."):
            result = create_detection_clips(
                clip_df,
                data_root,
                dataset_dir,
                float(min_confidence),
                float(pad_seconds),
                int(max_clips) if max_clips else None,
                overwrite=overwrite,
                progress_bar=progress_bar,
                status_box=status_box,
            )
        st.success(f"Clip export complete: created {result['created']}, skipped {result['skipped']}, failed {result['failed']}.")
        st.caption(f"Manifest: {result['manifest']}")


@st.cache_data(ttl=86400, show_spinner=False)
def geocode_location(query):
    response = requests.get(
        NOMINATIM_API,
        params={"q": query, "format": "jsonv2", "limit": 5},
        headers={"User-Agent": "AudioMoth-Dashboard/1.0 (local Streamlit app)"},
        timeout=15,
    )
    response.raise_for_status()
    rows = response.json()
    return [
        {
            "label": row.get("display_name", ""),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
        }
        for row in rows
        if row.get("lat") is not None and row.get("lon") is not None
    ]


def set_birdnet_coordinates(lat, lon, refresh_widgets=False):
    old_lat = st.session_state.get("birdnet_latitude")
    old_lon = st.session_state.get("birdnet_longitude")
    st.session_state["birdnet_latitude"] = float(lat)
    st.session_state["birdnet_longitude"] = float(lon)
    if refresh_widgets and (old_lat != float(lat) or old_lon != float(lon)):
        st.session_state["birdnet_location_version"] = int(st.session_state.get("birdnet_location_version", 0)) + 1


def render_birdnet_location_picker():
    st.markdown("**BirdNET Location**")
    st.caption("Use an address search or click the map. BirdNET uses latitude/longitude to choose the likely species list.")

    if "birdnet_latitude" not in st.session_state:
        st.session_state["birdnet_latitude"] = 55.6761
    if "birdnet_longitude" not in st.session_state:
        st.session_state["birdnet_longitude"] = 12.5683
    if "birdnet_location_version" not in st.session_state:
        st.session_state["birdnet_location_version"] = 0

    search_cols = st.columns([3, 1])
    with search_cols[0]:
        location_query = st.text_input("Search address or place", value="")
    with search_cols[1]:
        search_clicked = st.button("Search location")

    if search_clicked and location_query.strip():
        try:
            matches = geocode_location(location_query.strip())
            if matches:
                st.session_state["birdnet_location_matches"] = matches
                set_birdnet_coordinates(matches[0]["lat"], matches[0]["lon"], refresh_widgets=True)
                st.rerun(scope="fragment")
            else:
                st.warning("No matching location found.")
        except Exception as exc:
            st.warning(f"Location search failed: {exc}")

    matches = st.session_state.get("birdnet_location_matches", [])
    if matches:
        selected_match = st.selectbox(
            "Search results",
            matches,
            format_func=lambda row: row["label"],
        )
        if st.button("Use selected result"):
            set_birdnet_coordinates(selected_match["lat"], selected_match["lon"], refresh_widgets=True)
            st.rerun(scope="fragment")

    location_version = st.session_state["birdnet_location_version"]
    lat_col, lon_col = st.columns(2)
    with lat_col:
        lat = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=float(st.session_state["birdnet_latitude"]),
            format="%.6f",
            key=f"birdnet_lat_input_{location_version}",
        )
    with lon_col:
        lon = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=float(st.session_state["birdnet_longitude"]),
            format="%.6f",
            key=f"birdnet_lon_input_{location_version}",
        )
    set_birdnet_coordinates(lat, lon)

    if folium is not None and st_folium is not None:
        fmap = folium.Map(location=[lat, lon], zoom_start=12, control_scale=True)
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color="#d73027",
            weight=3,
            fill=True,
            fill_color="#d73027",
            fill_opacity=0.85,
            tooltip="BirdNET location",
            popup=f"{lat:.6f}, {lon:.6f}",
        ).add_to(fmap)
        map_key = f"birdnet_location_map_{lat:.6f}_{lon:.6f}"
        map_result = st_folium(fmap, height=340, use_container_width=True, key=map_key)
        clicked = map_result.get("last_clicked") if map_result else None
        if clicked:
            clicked_lat = float(clicked["lat"])
            clicked_lon = float(clicked["lng"])
            if abs(clicked_lat - lat) > 1e-6 or abs(clicked_lon - lon) > 1e-6:
                set_birdnet_coordinates(clicked_lat, clicked_lon, refresh_widgets=True)
                st.rerun(scope="fragment")
    else:
        st.info("Install `streamlit-folium` to enable the clickable map. Latitude and longitude boxes still work.")

    return float(st.session_state["birdnet_latitude"]), float(st.session_state["birdnet_longitude"])


def choose_folder_dialog(initial_path):
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askdirectory(
        initialdir=str(initial_path if initial_path.exists() else PROJECT_ROOT),
        title="Select audio folder",
    )
    root.destroy()
    return Path(selected) if selected else None


def update_folder_picker_path(key, path):
    st.session_state[f"{key}_path"] = str(Path(path).resolve())
    st.session_state[f"{key}_version"] = int(st.session_state.get(f"{key}_version", 0)) + 1


def open_folder(path):
    path = str(Path(path).resolve())
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def folder_picker(label, initial_path, key):
    state_key = f"{key}_path"
    version_key = f"{key}_version"
    if state_key not in st.session_state:
        st.session_state[state_key] = str(initial_path)
    if version_key not in st.session_state:
        st.session_state[version_key] = 0

    manual_path = st.text_input(
        label,
        value=st.session_state[state_key],
        key=f"{key}_text_{st.session_state[version_key]}",
    )
    current_path = Path(manual_path).expanduser()
    if not current_path.is_absolute():
        current_path = (PROJECT_ROOT / current_path).resolve()
    if not current_path.exists() and "\\_" in manual_path:
        unescaped_path = Path(manual_path.replace("\\_", "_")).expanduser()
        if not unescaped_path.is_absolute():
            unescaped_path = (PROJECT_ROOT / unescaped_path).resolve()
        if unescaped_path.exists():
            current_path = unescaped_path
            update_folder_picker_path(key, current_path)
    st.session_state[state_key] = str(current_path)

    nav_cols = st.columns([1, 1, 2])
    with nav_cols[0]:
        if st.button("Browse...", key=f"{key}_browse"):
            selected = choose_folder_dialog(current_path)
            if selected:
                update_folder_picker_path(key, selected)
                st.rerun(scope="fragment")

    with nav_cols[1]:
        if st.button("Open folder", key=f"{key}_explorer"):
            explorer_path = current_path if current_path.exists() else current_path.parent
            open_folder(explorer_path)

    with nav_cols[2]:
        if st.button("Parent folder", key=f"{key}_parent"):
            update_folder_picker_path(key, current_path.parent)
            st.rerun(scope="fragment")

    return current_path


def resolve_selected_audio_dir(path):
    if path.name.lower() == "audio":
        return path
    nested_audio = path / "audio"
    if nested_audio.exists() and nested_audio.is_dir():
        return nested_audio
    return path


def count_audio_files(audio_dir):
    if not audio_dir.exists() or not audio_dir.is_dir():
        return 0
    return sum(1 for path in audio_dir.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS)


def count_files_with_suffixes(folder, suffixes):
    if not folder.exists() or not folder.is_dir():
        return 0
    return sum(
        1
        for path in folder.rglob("*")
        if path.is_file() and any(path.name.endswith(suffix) for suffix in suffixes)
    )


def existing_dataset_folders(datasets_dir):
    if not datasets_dir.exists():
        return []
    return sorted([path for path in datasets_dir.iterdir() if path.is_dir()], key=lambda path: path.name.lower())


def processing_folder_label(dataset_dir):
    inventory = processing_dataset_inventory(str(dataset_dir.resolve()))
    audio_count = inventory["audio"]
    if not audio_count:
        return f"{dataset_dir.name} - no audio"

    birda_pending = inventory["birda_pending"]
    birdnet_pending = inventory["birdnet_pending"]
    batdetect_pending = inventory["batdetect2_pending"]
    status = f"new: {birda_pending} Birda, {birdnet_pending} BirdNET, {batdetect_pending} BatDetect2"
    return f"{dataset_dir.name} - {audio_count} audio - {status}"


def audio_files(audio_dir):
    if not audio_dir.exists() or not audio_dir.is_dir():
        return []
    return sorted(
        [
            path
            for path in audio_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        ]
    )


def recording_date_from_name(name):
    patterns = [
        (r"\d{8}_\d{6}", "%Y%m%d_%H%M%S"),
        (r"\d{8}-\d{6}", "%Y%m%d-%H%M%S"),
        (r"\d{4}-\d{2}-\d{2}[_-]\d{2}-\d{2}-\d{2}", None),
        (r"\d{8}", "%Y%m%d"),
    ]
    for pattern, fmt in patterns:
        match = re.search(pattern, name)
        if not match:
            continue
        stamp = match.group(0)
        formats = [fmt] if fmt else ["%Y-%m-%d_%H-%M-%S", "%Y-%m-%d-%H-%M-%S"]
        for date_format in formats:
            try:
                return datetime.strptime(stamp, date_format).date()
            except ValueError:
                pass
    return None


def first_recording_date(audio_dir):
    dates = [recording_date_from_name(path.name) for path in audio_files(audio_dir)]
    dates = [item for item in dates if item is not None]
    return min(dates) if dates else date.today()


def copy_audio_files(source_dir, target_dir, overwrite=False):
    files = audio_files(source_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for source_path in files:
        relative_path = source_path.relative_to(source_dir)
        target_path = target_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.exists() and not overwrite:
            skipped += 1
            continue

        shutil.copy2(source_path, target_path)
        copied += 1

    return copied, skipped, len(files)


def is_normal_analyzer_log(line):
    normal_patterns = (
        "All log messages before absl::InitializeLog()",
        "oneDNN custom operations are on",
        "Downloading model:",
        "port.cc:153",
        "I0000 00:00:",
    )
    return any(pattern in line for pattern in normal_patterns)


def show_command_result(returncode, stdout, stderr):
    if stdout.strip():
        st.text_area("Output", stdout, height=180)
    if stderr.strip():
        stderr_lines = [line for line in stderr.splitlines() if line.strip()]
        normal_lines = [line for line in stderr_lines if is_normal_analyzer_log(line)]
        other_lines = [line for line in stderr_lines if not is_normal_analyzer_log(line)]
        if returncode == 0 and normal_lines and not other_lines:
            with st.expander("Analyzer log", expanded=False):
                st.text("\n".join(normal_lines[-80:]))
        else:
            if other_lines:
                st.text_area("Errors / warnings", "\n".join(other_lines), height=180)
            if normal_lines:
                with st.expander("Analyzer log", expanded=False):
                    st.text("\n".join(normal_lines[-80:]))
    if returncode == 0:
        st.success("Command finished successfully.")
    else:
        st.error(f"Command failed with exit code {returncode}.")


def save_dataset_location_to_manifest(manifest_path, data_root, dataset_audio_dir, location_name, latitude, longitude):
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Regenerate the manifest first.")

    manifest = pd.read_csv(manifest_path)
    required = {"dataset_id", "audio_dir", "location_name", "latitude", "longitude"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")

    manifest["location_name"] = manifest["location_name"].fillna("").astype("object")
    manifest["latitude"] = pd.to_numeric(manifest["latitude"], errors="coerce")
    manifest["longitude"] = pd.to_numeric(manifest["longitude"], errors="coerce")

    rel_audio = dataset_audio_dir.resolve().relative_to(data_root.resolve()).as_posix()
    dataset_folder = dataset_audio_dir.parent.name
    mask = (
        manifest["audio_dir"].astype(str).str.replace("\\", "/", regex=False).eq(rel_audio)
        | manifest["dataset_id"].astype(str).eq(dataset_folder)
        | manifest["dataset_id"].astype(str).str.contains(re.escape(dataset_folder), case=False, na=False)
    )
    if not mask.any():
        raise ValueError(f"No manifest rows matched audio folder: {rel_audio}")

    manifest.loc[mask, "location_name"] = location_name
    manifest.loc[mask, "latitude"] = float(latitude)
    manifest.loc[mask, "longitude"] = float(longitude)
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return int(mask.sum())


def save_dataset_location_to_database(db_path, data_root, dataset_audio_dir, location_name, latitude, longitude):
    if not db_path.exists():
        return 0

    rel_audio = dataset_audio_dir.resolve().relative_to(data_root.resolve()).as_posix()
    dataset_folder = dataset_audio_dir.parent.name.lower()

    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute("SELECT dataset_id, audio_dir FROM datasets").df()
        if rows.empty:
            return 0

        audio_values = rows["audio_dir"].fillna("").astype(str).str.replace("\\", "/", regex=False)
        dataset_values = rows["dataset_id"].fillna("").astype(str)
        mask = (
            audio_values.eq(rel_audio)
            | dataset_values.eq(dataset_audio_dir.parent.name)
            | dataset_values.str.lower().str.contains(re.escape(dataset_folder), regex=True, na=False)
        )
        dataset_ids = dataset_values[mask].tolist()
        for dataset_id in dataset_ids:
            con.execute(
                """
                UPDATE datasets
                SET location_name = ?, latitude = ?, longitude = ?
                WHERE dataset_id = ?
                """,
                [location_name, float(latitude), float(longitude), dataset_id],
            )
        return len(dataset_ids)
    finally:
        con.close()


def load_dataset_location(manifest_path, db_path, data_root, dataset_audio_dir):
    rel_audio = dataset_audio_dir.resolve().relative_to(data_root.resolve()).as_posix()
    matching_rows = pd.DataFrame()
    if manifest_path.exists():
        try:
            manifest = pd.read_csv(manifest_path)
            if "audio_dir" in manifest.columns:
                audio_values = manifest["audio_dir"].fillna("").astype(str).str.replace("\\", "/", regex=False)
                matching_rows = manifest[audio_values.eq(rel_audio)]
        except Exception:
            matching_rows = pd.DataFrame()

    if matching_rows.empty and db_path.exists():
        con = None
        try:
            con = duckdb.connect(str(db_path), read_only=True)
            database_rows = con.execute(
                "SELECT location_name, latitude, longitude, audio_dir FROM datasets"
            ).df()
            audio_values = database_rows["audio_dir"].fillna("").astype(str).str.replace("\\", "/", regex=False)
            matching_rows = database_rows[audio_values.eq(rel_audio)]
        except Exception:
            matching_rows = pd.DataFrame()
        finally:
            if con is not None:
                con.close()

    if matching_rows.empty:
        return dataset_audio_dir.parent.name, 55.6761, 12.5683

    names = matching_rows.get("location_name", pd.Series(dtype="object")).dropna().astype(str)
    names = names[names.str.strip().ne("")]
    latitudes = pd.to_numeric(matching_rows.get("latitude", pd.Series(dtype="float64")), errors="coerce").dropna()
    longitudes = pd.to_numeric(matching_rows.get("longitude", pd.Series(dtype="float64")), errors="coerce").dropna()
    location_name = names.iloc[0] if not names.empty else dataset_audio_dir.parent.name
    latitude = float(latitudes.iloc[0]) if not latitudes.empty else 55.6761
    longitude = float(longitudes.iloc[0]) if not longitudes.empty else 12.5683
    return location_name, latitude, longitude


@st.fragment
def render_processing_page(data_root, db_path):
    st.header("Processing")

    datasets_dir = data_root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_root / "datasets" / "datasets_manifest.csv"
    resolved_db_path = resolve_db_path(db_path, data_root)

    project_tab, import_tab, analyzers_tab, database_tab, clips_tab = st.tabs(
        ["Project", "Import Audio", "Analyzers", "Database", "Playback Clips"]
    )

    with project_tab:
        st.subheader("Project Folder")
        st.caption("Create or select a dataset folder. Audio is stored in `audio`; analyzer outputs use `birda`, `birdnet`, and `batdetect2_outputs`.")
        dataset_folders = existing_dataset_folders(datasets_dir)
        create_dataset_label = "Create new dataset"
        dataset_labels = {path.name: processing_folder_label(path) for path in dataset_folders}
        dataset_options = [create_dataset_label] + list(dataset_labels)
        saved_dataset_selection = st.session_state.get("processing_dataset_folder")
        if saved_dataset_selection not in dataset_options:
            st.session_state["processing_dataset_folder"] = create_dataset_label
        selected_dataset_name = st.selectbox(
            "Dataset folder",
            dataset_options,
            format_func=lambda value: dataset_labels.get(value, value),
            key="processing_dataset_folder",
        )
        if selected_dataset_name != create_dataset_label:
            dataset_name = selected_dataset_name
            st.text_input(
                "Dataset folder name",
                value=dataset_name,
                disabled=True,
                key=f"selected_dataset_name_{dataset_name}",
            )
        else:
            dataset_name = st.text_input(
                "Dataset folder name",
                value="11_new_dataset",
                key="processing_new_dataset_name",
            )
    dataset_dir = datasets_dir / dataset_name
    dataset_audio_dir = dataset_dir / "audio"
    birdnet_output_dir = dataset_dir / "birdnet"
    batdetect_output_dir = dataset_dir / "batdetect2_outputs"
    birda_output_dir = dataset_dir / "birda"

    inventory = processing_dataset_inventory(str(dataset_dir.resolve()))
    target_count = inventory["audio"]
    birdnet_output_count = inventory["birdnet_outputs"]
    batdetect_output_count = inventory["batdetect2_outputs"]
    birda_output_count = inventory["birda_outputs"]
    birda_pending_count = inventory["birda_pending"]
    birdnet_pending_count = inventory["birdnet_pending"]
    batdetect_pending_count = inventory["batdetect2_pending"]
    with project_tab:
        if st.button("Create dataset folders"):
            dataset_audio_dir.mkdir(parents=True, exist_ok=True)
            birda_output_dir.mkdir(parents=True, exist_ok=True)
            birdnet_output_dir.mkdir(parents=True, exist_ok=True)
            batdetect_output_dir.mkdir(parents=True, exist_ok=True)
            processing_dataset_inventory.clear()
            st.success(f"Created folder structure: {dataset_dir}")

        st.caption(f"Dataset audio folder: {dataset_audio_dir}")
        status_cols = st.columns(4)
        status_cols[0].metric("Audio files", f"{target_count:,}")
        status_cols[1].metric("New for Birda", f"{birda_pending_count:,}")
        status_cols[2].metric("New for BirdNET", f"{birdnet_pending_count:,}")
        status_cols[3].metric("New for BatDetect2", f"{batdetect_pending_count:,}")
        if target_count and not birdnet_output_count and not batdetect_output_count and not birda_output_count:
            st.success("This dataset has audio and no analyzer outputs yet. It is ready to analyze.")
        elif target_count:
            st.info("This dataset already has audio and some analyzer outputs.")
        if st.button("Open dataset audio folder"):
            dataset_audio_dir.mkdir(parents=True, exist_ok=True)
            open_folder(dataset_audio_dir)
        if st.button("Refresh file status"):
            processing_dataset_inventory.clear()
            cached_audio_count.clear()
            st.rerun(scope="fragment")

    audio_dir = dataset_audio_dir
    audio_count = inventory["audio"]
    with import_tab:
        st.subheader("Import Audio Files")
        st.caption("Browse to the folder where the WAV/FLAC/MP3 files currently live, then copy them into the dataset audio folder.")
        default_source_audio = dataset_audio_dir if dataset_audio_dir.exists() else datasets_dir
        selected_source_path = folder_picker("Source audio folder", default_source_audio, "processing_source_audio")
        source_audio_dir = resolve_selected_audio_dir(selected_source_path)
        if source_audio_dir != selected_source_path:
            st.info(f"Using nested source audio folder: {source_audio_dir}")

        source_count = cached_audio_count(str(source_audio_dir.resolve()))
        st.caption(f"Resolved source folder: {source_audio_dir}")
        if source_audio_dir.exists():
            st.caption(f"Audio files found in source: {source_count}")
        else:
            st.warning(f"Source audio folder does not exist: {source_audio_dir}")

        overwrite_audio = st.checkbox("Overwrite existing audio files in dataset", value=False)
        if st.button("Copy source audio to dataset"):
            if not source_audio_dir.exists() or not source_audio_dir.is_dir():
                st.error(f"Source audio folder not found: {source_audio_dir}")
            elif source_audio_dir.resolve() == dataset_audio_dir.resolve():
                st.info("Source and dataset audio folders are the same. Nothing to copy.")
            else:
                with st.spinner("Copying audio files into the dataset..."):
                    copied, skipped, total = copy_audio_files(source_audio_dir, dataset_audio_dir, overwrite=overwrite_audio)
                processing_dataset_inventory.clear()
                cached_audio_count.clear()
                st.success(f"Audio import complete: copied {copied}, skipped {skipped}, source files {total}.")

        if audio_count == 0:
            st.warning("The dataset audio folder is empty. Copy audio files before running analyzers.")
        else:
            st.success(f"Ready to analyze {audio_count} audio files from the dataset audio folder.")

    with analyzers_tab:
        st.subheader("Run Analyzers")
        st.info(
            "For best performance, use Birda when it is installed. It supports GPU acceleration, modern model management, "
            "and skip-existing behavior. BirdNET Analyzer and BatDetect2 remain available as compatibility backends."
        )

        saved_location_name, saved_latitude, saved_longitude = load_dataset_location(
            manifest_path,
            resolved_db_path,
            data_root,
            dataset_audio_dir,
        )
        if st.session_state.get("birdnet_location_dataset") != dataset_name:
            set_birdnet_coordinates(saved_latitude, saved_longitude, refresh_widgets=True)
            st.session_state["birdnet_location_dataset"] = dataset_name
            st.session_state.pop("birdnet_location_matches", None)

        birdnet_lat, birdnet_lon = render_birdnet_location_picker()
        dataset_location_name = st.text_input(
            "Dataset location name",
            value=saved_location_name,
            key=f"dataset_location_name_{dataset_name}",
        )
        if st.button("Save analyzer location to manifest and database"):
            try:
                updated_rows = save_dataset_location_to_manifest(
                    manifest_path,
                    data_root,
                    dataset_audio_dir,
                    dataset_location_name,
                    birdnet_lat,
                    birdnet_lon,
                )
                updated_db_rows = save_dataset_location_to_database(
                    resolved_db_path,
                    data_root,
                    dataset_audio_dir,
                    dataset_location_name,
                    birdnet_lat,
                    birdnet_lon,
                )
                st.success(
                    f"Saved location to {updated_rows} manifest row(s) and {updated_db_rows} database row(s). "
                    "Switch back to Dashboard or refresh to see the sidebar map."
                )
            except Exception as exc:
                st.error(f"Could not save location: {exc}")
        default_birdnet_date = first_recording_date(audio_dir)
        birdnet_date = st.date_input(
            "Analyzer week date",
            value=default_birdnet_date,
            key=f"analyzer_week_date_{dataset_name}",
            help="Pick a representative recording date. The ISO week number is passed to BirdNET/Birda as week context.",
        )
        birdnet_week = int(birdnet_date.isocalendar().week)
        st.caption(f"Analyzer week: {birdnet_week} from {birdnet_date:%Y-%m-%d}")

        with st.container(border=True):
            st.markdown("**Bird Detection**")
            bird_backend = st.selectbox(
                "Bird analyzer",
                ["Birda", "BirdNET Analyzer"],
                help="Birda is the recommended fast backend. BirdNET Analyzer is kept for compatibility with original BirdNET output.",
            )
            if bird_backend == "Birda":
                render_birda_bird_runner(audio_dir, birda_output_dir, birdnet_lat, birdnet_lon, birdnet_week)
            else:
                render_birdnet_analyzer_runner(audio_dir, birdnet_output_dir, birdnet_lat, birdnet_lon, birdnet_week)

        with st.container(border=True):
            st.markdown("**Bat Detection**")
            bat_backend = st.selectbox(
                "Bat analyzer",
                ["BatDetect2"],
                help="BatDetect2 output is stored separately in batdetect2_outputs.",
            )
            if bat_backend == "BatDetect2":
                render_batdetect_runner(audio_dir, batdetect_output_dir)

    with database_tab:
        st.subheader("Database Status")
        try:
            project_dataset_ids = manifest_dataset_ids_for_folder(manifest_path, dataset_name)
        except Exception as exc:
            project_dataset_ids = []
            st.warning(f"Could not read the manifest: {exc}")

        try:
            db_detector_counts, db_project_ids, import_snapshots = database_detector_counts(
                resolved_db_path, project_dataset_ids
            )
        except Exception as exc:
            db_detector_counts, db_project_ids, import_snapshots = {}, set(), {}
            st.warning(f"Could not inspect the database: {exc}")

        database_status = pd.DataFrame(
            [
                {
                    "Analyzer": "Birda",
                    "Output files": birda_output_count,
                    "WAVs not analyzed": birda_pending_count,
                    "Detections in database": db_detector_counts.get("birda", 0),
                },
                {
                    "Analyzer": "BirdNET",
                    "Output files": birdnet_output_count,
                    "WAVs not analyzed": birdnet_pending_count,
                    "Detections in database": db_detector_counts.get("birdnet", 0),
                },
                {
                    "Analyzer": "BatDetect2",
                    "Output files": batdetect_output_count,
                    "WAVs not analyzed": batdetect_pending_count,
                    "Detections in database": db_detector_counts.get("batdetect2", 0),
                },
            ]
        )
        st.dataframe(database_status, hide_index=True, width="stretch")

        output_total = birda_output_count + birdnet_output_count + batdetect_output_count
        current_output_counts = {
            "birda": birda_output_count,
            "birdnet": birdnet_output_count,
            "batdetect2": batdetect_output_count,
        }
        changed_analyzers = [
            detector
            for detector, count in current_output_counts.items()
            if count and import_snapshots.get(detector) != count
        ]
        missing_project_ids = set(project_dataset_ids) - db_project_ids
        if output_total == 0:
            st.warning("No analyzer outputs are available yet. Run Birda, BirdNET, or BatDetect2 before updating the database.")
        elif not project_dataset_ids:
            st.warning("This project is not in the manifest yet. Use Update selected project; it will generate the manifest first.")
        elif not resolved_db_path.exists() or missing_project_ids:
            st.warning("This project has analyzer outputs that are not yet in DuckDB. Use Update selected project.")
        elif not import_snapshots:
            st.warning(
                "This database predates output tracking. Use Update selected project once; future changes will then be detected automatically."
            )
        elif changed_analyzers:
            analyzer_names = ", ".join(detector_names.title() for detector_names in changed_analyzers)
            st.warning(f"Output-file counts changed for {analyzer_names}. Use Update selected project.")
        else:
            st.success("The selected project appears up to date. Run Update selected project after adding or replacing analyzer outputs.")

        st.markdown("**Recommended action**")
        st.caption(
            "This one button is suitable for both new and existing projects. It regenerates the manifest, then replaces only "
            "the selected project's database rows with the current analyzer outputs. Other projects remain untouched."
        )
        if st.button("Update selected project", type="primary", width="stretch"):
            manifest_command = [
                sys.executable,
                str(SCRIPTS_DIR / "generate_manifest.py"),
                "--data-root",
                str(data_root),
                "--manifest",
                str(manifest_path),
            ]
            with st.spinner("Refreshing manifest and selected project..."):
                manifest_code, manifest_stdout, manifest_stderr = run_captured_command(manifest_command, PROJECT_ROOT)
                if manifest_code == 0:
                    refreshed_ids = manifest_dataset_ids_for_folder(manifest_path, dataset_name)
                    if refreshed_ids:
                        build_command = [
                            sys.executable,
                            str(SCRIPTS_DIR / "build_db.py"),
                            "--data-root",
                            str(data_root),
                            "--manifest",
                            str(manifest_path),
                            "--db",
                            str(resolved_db_path),
                        ]
                        for dataset_id in refreshed_ids:
                            build_command.extend(["--dataset-id", dataset_id])
                        build_code, build_stdout, build_stderr = run_captured_command(build_command, PROJECT_ROOT)
                        show_command_result(
                            build_code,
                            f"{manifest_stdout}\n{build_stdout}".strip(),
                            f"{manifest_stderr}\n{build_stderr}".strip(),
                        )
                    else:
                        st.error("The manifest was regenerated, but no rows matched the selected project folder.")
                else:
                    show_command_result(manifest_code, manifest_stdout, manifest_stderr)

        with st.expander("Advanced database actions"):
            st.caption("Most updates should use Update selected project above. These actions operate on the whole manifest or database.")
            action_cols = st.columns(3)
            with action_cols[0]:
                if st.button("Regenerate manifest only"):
                    command = [sys.executable, str(SCRIPTS_DIR / "generate_manifest.py"), "--data-root", str(data_root), "--manifest", str(manifest_path)]
                    with st.spinner("Updating manifest..."):
                        returncode, stdout, stderr = run_captured_command(command, PROJECT_ROOT)
                    show_command_result(returncode, stdout, stderr)
            with action_cols[1]:
                if st.button("Rebuild all listed projects"):
                    command = [sys.executable, str(SCRIPTS_DIR / "build_db.py"), "--data-root", str(data_root), "--manifest", str(manifest_path), "--db", str(resolved_db_path)]
                    with st.spinner("Rebuilding all manifest projects..."):
                        returncode, stdout, stderr = run_captured_command(command, PROJECT_ROOT)
                    show_command_result(returncode, stdout, stderr)
            with action_cols[2]:
                if st.button("Replace entire database"):
                    command = [sys.executable, str(SCRIPTS_DIR / "build_db.py"), "--data-root", str(data_root), "--manifest", str(manifest_path), "--db", str(resolved_db_path), "--replace-db"]
                    with st.spinner("Replacing the entire database..."):
                        returncode, stdout, stderr = run_captured_command(command, PROJECT_ROOT)
                    show_command_result(returncode, stdout, stderr)

            if st.button(
                "Compute missing audio quality metrics only",
                help="Backfill signal level, local background level, and SNR without importing detections.",
            ):
                command = [sys.executable, str(SCRIPTS_DIR / "build_db.py"), "--data-root", str(data_root), "--db", str(resolved_db_path), "--audio-metrics-only"]
                with st.spinner("Computing missing detection-level dBFS and SNR metrics..."):
                    returncode, stdout, stderr = run_captured_command(command, PROJECT_ROOT)
                show_command_result(returncode, stdout, stderr)

        if manifest_path.exists():
            st.subheader("Manifest Preview")
            try:
                st.dataframe(pd.read_csv(manifest_path), width="stretch")
            except Exception as exc:
                st.warning(f"Could not preview manifest: {exc}")

    clip_dataset_ids = []
    try:
        clip_dataset_ids = manifest_dataset_ids_for_folder(manifest_path, dataset_name)
    except Exception as exc:
        st.warning(f"Could not read manifest for clip dataset choices: {exc}")
    if not clip_dataset_ids:
        clip_dataset_ids = [dataset_name]
    with clips_tab:
        st.subheader("Create Playback Clips")
        st.info(
            "Clips are made from database detections after analyzer import. Overlapping detections are merged into one WAV clip, "
            "so the same moment is not saved repeatedly. Bird and bat detections that share this audio folder are clipped together. "
            "Original WAVs are left untouched."
        )
        render_clip_creator(data_root, resolved_db_path, clip_dataset_ids, dataset_dir)


st.set_page_config(page_title=APP_TITLE, page_icon="🎙️", layout="wide")
inject_design_system()

db_path = os.environ.get("AUDIOMOTH_DB", str(DEFAULT_DB_PATH))
raw_data_root = Path(os.environ.get("AUDIOMOTH_DATA_ROOT", str(PROJECT_ROOT)))
data_root = normalize_data_root(raw_data_root)
ensure_local_data_folders(data_root)
resolved_db_path = resolve_db_path(db_path, data_root)

if "view_mode" not in st.session_state:
    st.session_state["view_mode"] = "Dashboard"
view_mode = st.session_state["view_mode"]

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=220)
    st.markdown(f"### {APP_TITLE}")
    st.header("Project")
    if st.button("Create Project", width="stretch"):
        st.session_state["processing_dataset_folder"] = "Create new dataset"
        st.session_state["view_mode"] = "Processing"
        st.rerun()

if not resolved_db_path.exists() and view_mode != "Processing":
    render_first_run_page(data_root, resolved_db_path)
    st.stop()

if not resolved_db_path.exists() and view_mode == "Processing":
    with st.sidebar:
        st.header("Navigate")
        for page_name in ["Dashboard", "Processing", "Bird highlights", "Bat highlights"]:
            button_type = "primary" if st.session_state["view_mode"] == page_name else "secondary"
            if st.button(page_name, width="stretch", type=button_type):
                st.session_state["view_mode"] = page_name
                st.rerun()
    render_processing_page(data_root, db_path)
    st.stop()

con = connect_database(resolved_db_path)
platform_schema = has_platform_schema(con)

if platform_schema:
    datasets = load_platform_datasets(con)
    df = None
else:
    datasets, df = load_legacy_data(con)

datasets = datasets.copy()
datasets["_sort_key"] = datasets["dataset_id"].map(dataset_sort_key)
datasets = datasets.sort_values("_sort_key").drop(columns=["_sort_key"])

points = location_points(datasets)

dataset_labels = {}
for _, row in datasets.iterrows():
    label = dataset_label(row)
    if label in dataset_labels:
        label = f"{label} ({row['dataset_id']})"
    dataset_labels[label] = row["dataset_id"]

selected_dataset_label = st.sidebar.selectbox("Selected project", list(dataset_labels.keys()))
selected_dataset_id = dataset_labels[selected_dataset_label]

with st.sidebar:
    if st.button("Open Selected Project", width="stretch"):
        st.session_state["view_mode"] = "Dashboard"
        st.rerun()

    st.header("Navigate")
    for page_name in ["Dashboard", "Processing", "Bird highlights", "Bat highlights"]:
        button_type = "primary" if st.session_state["view_mode"] == page_name else "secondary"
        if st.button(page_name, width="stretch", type=button_type):
            st.session_state["view_mode"] = page_name
            st.rerun()

view_mode = st.session_state["view_mode"]

if view_mode == "Processing":
    con.close()
    render_processing_page(data_root, db_path)
    st.stop()

if view_mode == "Bird highlights":
    selected_dataset_ids = render_highlight_scope_selector(datasets, dataset_labels, selected_dataset_id, "birdnet")
    if selected_dataset_ids is not None:
        if platform_schema:
            scoped = load_platform_detections(con, detector=["birdnet", "birda"], dataset_ids=selected_dataset_ids)
            scoped = scoped[bird_highlight_filter(scoped)] if not scoped.empty else scoped
        else:
            scoped = df[bird_highlight_filter(df) & df["dataset_id"].isin(selected_dataset_ids)]
        if not scoped.empty:
            scoped["DateTime"] = pd.to_datetime(scoped["DateTime"])
        render_gallery(scoped, data_root, "birdnet")
    con.close()
    st.stop()

if view_mode == "Bat highlights":
    selected_dataset_ids = render_highlight_scope_selector(datasets, dataset_labels, selected_dataset_id, "batdetect2")
    if selected_dataset_ids is not None:
        if platform_schema:
            scoped = load_platform_detections(con, detector=["batdetect2", "birda"], dataset_ids=selected_dataset_ids)
            scoped = scoped[bat_highlight_filter(scoped)] if not scoped.empty else scoped
        else:
            scoped = df[bat_highlight_filter(df) & df["dataset_id"].isin(selected_dataset_ids)]
        if not scoped.empty:
            scoped["DateTime"] = pd.to_datetime(scoped["DateTime"])
        render_gallery(scoped, data_root, "batdetect2")
    con.close()
    st.stop()

with st.sidebar:
    if not points.empty:
        st.subheader("Locations")
        sidebar_deck = map_deck(points, selected_dataset_id=selected_dataset_id, satellite=False)
        st.pydeck_chart(sidebar_deck, width="stretch")
    else:
        st.caption("No dataset coordinates saved yet.")

if platform_schema:
    df = load_platform_detections(con, dataset_ids=[selected_dataset_id])
else:
    df = df[df["dataset_id"] == selected_dataset_id].copy()

if df.empty:
    st.error("No detections found for the selected dataset.")
    con.close()
    st.stop()

df["DateTime"] = pd.to_datetime(df["DateTime"])
dataset_row = datasets[datasets["dataset_id"] == selected_dataset_id].iloc[0]

start_time = df["DateTime"].min()
end_time = df["DateTime"].max()

species_counts = df["CommonName"].value_counts().reset_index()
species_counts.columns = ["Common Name", "Count"]
species_counts = species_counts.sort_values("Count", ascending=False)

st.title(APP_TITLE)
overview_tab, histogram_tab, matrix_tab, species_tab, temporal_tab, inspector_tab = st.tabs(
    [
        "Overview",
        "Histogram",
        "Matrix",
        "Species",
        "Temporal Pattern",
        "Detection Inspector",
    ]
)

selected_points = points[points["dataset_id"] == selected_dataset_id]
species_options = [f"{row['Common Name']} ({row['Count']})" for _, row in species_counts.iterrows()]
species_lookup = {f"{row['Common Name']} ({row['Count']})": row["Common Name"] for _, row in species_counts.iterrows()}

with overview_tab:
    st.header("Project Overview")
    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Location", dataset_row.get("location_name") or dataset_row.get("dataset_id"))
    with metric_cols[1]:
        st.metric("Date range", f"{start_time:%d %b %Y} - {end_time:%d %b %Y}")
    with metric_cols[2]:
        st.metric("Total observations", f"{len(df):,}")

    if not selected_points.empty:
        st.subheader("Location")
        map_style = st.radio("Map style", ["Satellite", "Standard"], horizontal=True, key="overview_map_style")
        if map_style == "Satellite":
            st.plotly_chart(
                satellite_map_figure(selected_points),
                width="stretch",
                config={"scrollZoom": True, "displaylogo": False},
            )
        else:
            selected_deck = map_deck(
                selected_points,
                selected_dataset_id=selected_dataset_id,
                satellite=False,
                zoom=12,
            )
            st.pydeck_chart(selected_deck, width="stretch")
    else:
        st.info("No coordinates saved for this project yet.")

with histogram_tab:
    st.header("Species Histogram")
    render_detection_count_note()
    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(species_counts, hide_index=True, width="stretch", height=800)

    with col2:
        fig = px.bar(
            species_counts,
            x="Count",
            y="Common Name",
            orientation="h",
            log_x=True,
            color="Common Name",
            color_discrete_sequence=px.colors.sequential.Viridis_r,
        )
        fig.update_layout(showlegend=False, height=800)
        fig.update_xaxes(
            type="log",
            tickvals=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
            ticktext=["1", "2", "5", "10", "20", "50", "100", "200", "500", "1k", "2k", "5k", "10k"],
        )
        st.plotly_chart(fig, width="stretch")

with matrix_tab:
    st.header("Species Occurrence Matrix")

    auto_resolution = auto_time_resolution(start_time, end_time)
    matrix_resolution_options = ["Auto", "Hour", "Day", "Week"]
    matrix_resolution_choice = st.radio(
        "Matrix time resolution",
        matrix_resolution_options,
        index=0,
        horizontal=True,
        help="Auto uses hourly bins for short datasets, daily bins for medium datasets, and weekly bins for longer deployments.",
    )
    matrix_resolution = auto_resolution if matrix_resolution_choice == "Auto" else matrix_resolution_choice
    matrix_settings = time_resolution_settings(matrix_resolution)
    st.caption(f"Matrix shows detections by {matrix_resolution.lower()}. Auto would use: {auto_resolution.lower()}.")

    matrix_df = add_time_bin(df, matrix_resolution)
    matrix_df["TimeLabel"] = matrix_df["TimeBin"].dt.strftime(matrix_settings["format"])
    occurrence_matrix = matrix_df.pivot_table(
        index="CommonName",
        columns="TimeLabel",
        values="Confidence",
        aggfunc="size",
        fill_value=0,
    )
    occurrence_matrix = occurrence_matrix.loc[occurrence_matrix.sum(axis=1).sort_values(ascending=False).index]
    ordered_time_labels = matrix_df[["TimeBin", "TimeLabel"]].drop_duplicates().sort_values("TimeBin")["TimeLabel"].tolist()
    occurrence_matrix = occurrence_matrix.reindex(columns=ordered_time_labels, fill_value=0)

    render_occurrence_matrix(occurrence_matrix, matrix_settings["label"])

with species_tab:
    st.header("Species Selection")

    selected_display = st.selectbox("Select species", species_options)
    selected_species = species_lookup[selected_display]

    species_df = df[df["CommonName"] == selected_species]

    if species_df.empty:
        st.warning("No detections for this species.")
    else:
        with st.expander("Species profile", expanded=True):
            profile_error = None
            try:
                profile = wikipedia_species_profile(selected_species)
            except Exception as exc:
                profile_error = exc
                profile = None

            if profile:
                image_col, text_col, map_col = st.columns([1, 2, 1])
                with image_col:
                    if profile["image_url"]:
                        st.image(profile["image_url"], width="stretch")
                with text_col:
                    st.subheader(profile["title"])
                    if profile["extract"]:
                        st.write(profile["extract"])
                    if profile["page_url"]:
                        st.link_button(
                            "Open Wikipedia",
                            profile["page_url"],
                            icon=":material/open_in_new:",
                        )
                    if profile.get("cached"):
                        st.caption("Profile saved locally for faster loading and offline reuse.")
                with map_col:
                    if profile["range_map_url"]:
                        st.image(profile["range_map_url"], width="stretch")

                details = profile.get("details", {})
                if details:
                    detail_columns = st.columns(2)
                    for index, (heading, text) in enumerate(details.items()):
                        with detail_columns[index % 2]:
                            st.markdown(f"**{heading}**")
                            st.write(text)
                if "Lifespan" not in details:
                    st.caption("Wikipedia does not provide specific lifespan information for this species.")
            else:
                if profile_error:
                    st.warning(f"Wikipedia lookup failed: {profile_error}")
                else:
                    st.info("No Wikipedia profile found for this selection.")

with temporal_tab:
    st.header("Temporal Pattern")

    auto_temporal = auto_temporal_resolution(species_df["DateTime"].min(), species_df["DateTime"].max())
    temporal_resolution_options = ["Auto", "Minute", "Hour", "Day", "Week"]
    temporal_resolution_choice = st.radio(
        "Temporal plot resolution",
        temporal_resolution_options,
        index=0,
        horizontal=True,
        help="Minute bins are useful for short recordings or single-night deployments. Longer spans are usually easier to read by hour, day, or week.",
    )
    temporal_resolution = auto_temporal if temporal_resolution_choice == "Auto" else temporal_resolution_choice
    temporal_settings = time_resolution_settings(temporal_resolution)
    st.caption(f"Temporal plot shows detections by {temporal_resolution.lower()}. Auto would use: {auto_temporal.lower()}.")

    temporal_counts = complete_time_counts(species_df, temporal_resolution)
    temporal_title = {
        "Minute": "Minute-by-minute",
        "Hour": "Hourly",
        "Day": "Daily",
        "Week": "Weekly",
    }[temporal_resolution]
    fig = px.bar(
        temporal_counts,
        x="TimeBin",
        y="Count",
        title=f"{temporal_title} detections of {selected_species}",
    )
    fig.update_layout(xaxis_title=temporal_settings["axis"], yaxis_title="Detections")
    st.plotly_chart(fig, width="stretch")

with inspector_tab:
    st.header("Detection Inspector")

    min_conf = float(species_df["Confidence"].min())
    max_conf = float(species_df["Confidence"].max())

    if min_conf < max_conf:
        selected_conf = st.slider(
            "Minimum confidence",
            min_value=min_conf,
            max_value=max_conf,
            value=min_conf,
        )
    else:
        selected_conf = min_conf
        st.caption(f"Only one confidence value for this selection: {selected_conf:.4f}")

    filtered_df = species_df[species_df["Confidence"] >= selected_conf].copy()
    filtered_df = filtered_df.sort_values("Confidence", ascending=False)

    visible_columns = [
        "DateTime",
        "Detector",
        "CommonName",
        "Confidence",
        "MeanDbfs",
        "SignalDbfs",
        "NoiseDbfs",
        "SnrDb",
        "relative_audio_path",
        "BeginTime",
        "EndTime",
        "SpeciesCode",
    ]
    visible_columns = [column for column in visible_columns if column in filtered_df.columns]

    if filtered_df.empty:
        st.dataframe(filtered_df[visible_columns], width="stretch")
        st.warning("No detections match the selected filters.")
    else:
        st.caption("Select a detection row to load its audio and spectrogram.")
        detection_table = filtered_df[visible_columns].reset_index(drop=True)
        table_event = st.dataframe(
            detection_table,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key=f"inspector_table_{selected_dataset_id}_{selected_species}",
        )
        selected_position = table_event.selection.rows[0] if table_event.selection.rows else 0
        row = filtered_df.iloc[selected_position]
        selected_index = row.name
        inspector_kind = str(row.get("Detector") or "birdnet").lower()
        if str(row.get("dataset_type") or "").lower() == "bats":
            inspector_kind = "bats"
        render_audio_inspector(
            row,
            data_root,
            inspector_kind,
            key_prefix=f"inspector_{selected_index}",
            context_df=df,
        )

con.close()
