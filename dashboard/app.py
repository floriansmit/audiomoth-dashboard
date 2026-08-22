import os
import html
import shlex
import shutil
import subprocess
import sys
import tempfile
import math
import re
import time
from datetime import date, datetime
from pathlib import Path

import duckdb
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
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
DEFAULT_DB_PATH = PROJECT_ROOT / "datasets" / "birdnet.duckdb"
LOGO_PATH = APP_DIR / "assets" / "audiomoth-bird-bat-logo-sidebar.png"
APP_TITLE = "AudioMoth Dashboard"
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3"}
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_HEADERS = {"User-Agent": "AudioMoth-Dashboard/1.0 (local Streamlit app)"}
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


def has_platform_schema(con):
    tables = con.execute("SHOW TABLES").df()["name"].tolist()
    return {"datasets", "recordings", "detections"}.issubset(set(tables))


def load_platform_data(con):
    dataset_columns = con.execute("PRAGMA table_info('datasets')").df()["name"].tolist()
    detection_columns = con.execute("PRAGMA table_info('detections')").df()["name"].tolist()
    tables = con.execute("SHOW TABLES").df()["name"].tolist()
    has_audio_metrics = "detection_audio_metrics" in tables
    datasets = con.execute("SELECT * FROM datasets").df()
    if "dataset_type" not in datasets.columns:
        datasets["dataset_type"] = np.where(
            datasets["dataset_id"].str.contains("bat", case=False, na=False),
            "bats",
            "birds",
        )

    dataset_type_sql = (
        "ds.dataset_type"
        if "dataset_type" in dataset_columns
        else "CASE WHEN lower(ds.dataset_id) LIKE '%bat%' THEN 'bats' ELSE 'birds' END AS dataset_type"
    )
    metrics_select_sql = "m.mean_dbfs AS MeanDbfs" if has_audio_metrics else "CAST(NULL AS DOUBLE) AS MeanDbfs"
    metrics_join_sql = (
        "LEFT JOIN detection_audio_metrics m ON d.detection_id = m.detection_id"
        if has_audio_metrics
        else ""
    )
    detector_select_sql = "d.Detector" if "Detector" in detection_columns else "'birdnet' AS Detector"
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
        """
    ).df()
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
        wav_path = resolve_audio_path(row, data_root)
        if wav_path is None or not wav_path.exists():
            continue

        begin_time = float(row["BeginTime"])
        end_time = float(row["EndTime"])
        noise_start = max(0, begin_time - 3)
        noise_duration = max(1.0, (end_time - begin_time) + 6)

        mean_dbfs = row.get("MeanDbfs")
        if pd.isna(mean_dbfs):
            try:
                mean_dbfs = segment_dbfs(str(wav_path), noise_start, noise_duration)
            except Exception:
                mean_dbfs = np.nan

        rows.append(
            {
                "Date": pd.to_datetime(row["DateTime"]).strftime("%Y-%m-%d %H:%M"),
                "Location": row.get("location_name", ""),
                "Confidence": float(row["Confidence"]),
                "Mean dBFS": mean_dbfs,
                "Audio": Path(str(row.get("relative_audio_path") or row.get("BeginPath"))).name,
                "RowIndex": row.name,
            }
        )

    if not rows:
        return pd.DataFrame()

    results = pd.DataFrame(rows)
    results = results.sort_values(["Confidence", "Mean dBFS"], ascending=[False, True]).head(5).reset_index(drop=True)
    results.insert(0, "Rank", np.arange(1, len(results) + 1))
    return results


def render_audio_inspector(row, data_root, dataset_type, key_prefix):
    wav_path = resolve_audio_path(row, data_root)
    start = max(0, float(row["BeginTime"]) - 3)
    end = float(row["EndTime"]) + 3
    duration = max(0.5, end - start)

    if wav_path is None or not wav_path.exists():
        st.error(f"Audio file not found: {wav_path}")
        return

    try:
        y, sr = load_audio_segment(wav_path, start, duration)

        playback_options = {
            "Original speed": 1,
            "4x time expansion": 4,
            "10x time expansion": 10,
        }
        playback_kind = str(dataset_type).lower()
        default_playback = "10x time expansion" if playback_kind in {"bats", "batdetect2"} else "Original speed"
        playback_label = st.selectbox(
            "Audio playback",
            list(playback_options),
            index=list(playback_options).index(default_playback),
            key=f"{key_prefix}_playback",
        )
        expansion_factor = playback_options[playback_label]
        playback_sr = max(1, int(round(sr / expansion_factor)))

        buffer = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(buffer.name, y, playback_sr)
        st.audio(buffer.name)

        if expansion_factor > 1:
            st.caption(
                f"Time-expanded {expansion_factor}x: pitch and playback speed are both lowered "
                f"by a factor of {expansion_factor}."
            )

        spectrum = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        fig, ax = plt.subplots(figsize=(12, 5))
        librosa.display.specshow(spectrum, sr=sr, x_axis="time", y_axis="log", ax=ax)
        ax.set_ylim([20, sr / 2])
        st.pyplot(fig)

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
    figure = px.scatter_mapbox(
        points,
        lat="lat",
        lon="lon",
        hover_name="label",
        zoom=zoom,
        center=center,
    )
    figure.update_traces(marker={"size": 15, "color": "#ffd246"})
    figure.update_layout(
        mapbox_style="white-bg",
        mapbox_layers=[
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

    return {
        "title": title,
        "extract": shorten_text(page.get("extract"), 420),
        "image_url": page.get("thumbnail", {}).get("source"),
        "range_map_url": range_map_url,
        "page_url": page.get("fullurl"),
        "details": details,
    }


@st.cache_data(ttl=86400, show_spinner=False)
def wikipedia_species_image(species_name):
    profile = wikipedia_species_profile(species_name)
    return profile.get("image_url") if profile else None


def render_gallery(gallery_df, data_root, gallery_type):
    label = "Bird" if gallery_type == "birdnet" else "Bat"
    st.header(f"{label} Highlights Gallery")
    st.caption(
        "Highlights combine high classifier confidence with a lower mean dBFS value "
        "from the surrounding audio segment. Lower dBFS is generally quieter background audio."
    )

    if gallery_df.empty:
        st.warning(f"No {gallery_type} detections found.")
        return

    species_counts = gallery_df["CommonName"].value_counts()
    max_species = st.slider(
        "Species in gallery",
        min_value=1,
        max_value=int(min(50, len(species_counts))),
        value=int(min(12, len(species_counts))),
    )
    max_candidates = st.slider("High-confidence candidates checked per species", 5, 50, 20, step=5)

    selected_species = species_counts.head(max_species).index.tolist()
    for species in selected_species:
        species_key = re.sub(r"[^A-Za-z0-9_]+", "_", species)
        species_df = gallery_df[gallery_df["CommonName"] == species]
        with st.container(border=True):
            image_col, table_col = st.columns([1, 3])
            with image_col:
                try:
                    image_url = wikipedia_species_image(species)
                except Exception:
                    image_url = None
                if image_url:
                    st.image(image_url, use_container_width=True)
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
                        use_container_width=True,
                        column_config={
                            "Confidence": st.column_config.NumberColumn(format="%.3f"),
                            "Mean dBFS": st.column_config.NumberColumn(format="%.1f"),
                        },
                    )
                    best["Choice"] = (
                        best["Rank"].astype(str)
                        + " | "
                        + best["Date"]
                        + " | conf "
                        + best["Confidence"].round(3).astype(str)
                        + " | "
                        + best["Mean dBFS"].round(1).astype(str)
                        + " dBFS"
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
                    )


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


def run_monitored_command(command, cwd, label, audio_dir, output_dir, output_suffixes):
    audio_list = audio_files(audio_dir)
    total_files = len(audio_list)
    initial_outputs = len(output_files(output_dir, output_suffixes))

    progress_bar = st.progress(0.0, text=f"{label}: starting...")
    status_box = st.empty()
    log_box = st.empty()

    stdout_file = tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False)
    stderr_file = tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False)
    stdout_path = Path(stdout_file.name)
    stderr_path = Path(stderr_file.name)

    started = time.time()
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=command_environment(),
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
        )

        while process.poll() is None:
            current_outputs = len(output_files(output_dir, output_suffixes))
            done_files = min(current_outputs, total_files) if total_files else current_outputs
            new_outputs = max(0, current_outputs - initial_outputs)
            elapsed = time.time() - started

            if total_files:
                fraction = min(1.0, done_files / total_files)
                current_index = min(done_files, total_files - 1)
                current_name = audio_list[current_index].name if audio_list else "unknown"
                eta = None
                if new_outputs > 0:
                    remaining = max(0, total_files - done_files)
                    seconds_per_file = elapsed / new_outputs
                    eta = remaining * seconds_per_file
                progress_bar.progress(
                    fraction,
                    text=f"{label}: {done_files}/{total_files} files",
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
        stdout_file.close()
        stderr_file.close()

        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")

        final_outputs = len(output_files(output_dir, output_suffixes))
        final_done = min(final_outputs, total_files) if total_files else final_outputs
        if total_files:
            progress_bar.progress(
                min(1.0, final_done / total_files),
                text=f"{label}: {final_done}/{total_files} files",
            )
        if returncode == 0:
            status_box.success(f"{label} finished. Output files now found: {final_outputs}.")
        else:
            status_box.error(f"{label} failed with exit code {returncode}.")

        return returncode, stdout, stderr
    finally:
        try:
            stdout_file.close()
            stderr_file.close()
        except Exception:
            pass
        for path in (stdout_path, stderr_path):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass


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
                st.rerun()
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
            st.rerun()

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
                st.rerun()
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
                st.rerun()

    with nav_cols[1]:
        if st.button("Open in Explorer", key=f"{key}_explorer"):
            explorer_path = current_path if current_path.exists() else current_path.parent
            subprocess.Popen(["explorer", str(explorer_path)])

    with nav_cols[2]:
        if st.button("Parent folder", key=f"{key}_parent"):
            update_folder_picker_path(key, current_path.parent)
            st.rerun()

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
    audio_count = count_audio_files(dataset_dir / "audio")
    birdnet_count = count_files_with_suffixes(
        dataset_dir / "birdnet",
        (".BirdNET.selection.table.txt", ".BirdNET.selection.table.csv", ".selection.table.txt", ".selection.table.csv", ".csv", ".txt"),
    )
    batdetect_count = count_files_with_suffixes(dataset_dir / "batdetect2_outputs", (".csv", ".json"))

    if audio_count and not birdnet_count and not batdetect_count:
        status = "ready to analyze"
    elif audio_count:
        status = f"{birdnet_count} BirdNET, {batdetect_count} BatDetect2 outputs"
    else:
        status = "no audio"
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


def render_processing_page(data_root, db_path):
    st.header("Processing")

    datasets_dir = data_root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_root / "datasets" / "datasets_manifest.csv"
    resolved_db_path = resolve_db_path(db_path, data_root)

    st.subheader("1. Prepare Dataset Folder")
    st.caption("Create or select a dataset folder. Audio will be stored in that dataset's `audio` folder; analyzer outputs go to `birdnet` and `batdetect2_outputs`.")
    dataset_folders = existing_dataset_folders(datasets_dir)
    dataset_options = ["Create new dataset"] + dataset_folders
    selected_dataset_option = st.selectbox(
        "Dataset folder",
        dataset_options,
        format_func=lambda option: option if isinstance(option, str) else processing_folder_label(option),
    )
    if isinstance(selected_dataset_option, Path):
        dataset_name = selected_dataset_option.name
        st.text_input("Dataset folder name", value=dataset_name, disabled=True)
    else:
        dataset_name = st.text_input("Dataset folder name", value="11_new_dataset")
    dataset_dir = datasets_dir / dataset_name
    dataset_audio_dir = dataset_dir / "audio"
    birdnet_output_dir = dataset_dir / "birdnet"
    batdetect_output_dir = dataset_dir / "batdetect2_outputs"

    if st.button("Create dataset folders"):
        dataset_audio_dir.mkdir(parents=True, exist_ok=True)
        birdnet_output_dir.mkdir(parents=True, exist_ok=True)
        batdetect_output_dir.mkdir(parents=True, exist_ok=True)
        st.success(f"Created folder structure: {dataset_dir}")

    target_count = count_audio_files(dataset_audio_dir)
    birdnet_output_count = count_files_with_suffixes(
        birdnet_output_dir,
        (".BirdNET.selection.table.txt", ".BirdNET.selection.table.csv", ".selection.table.txt", ".selection.table.csv", ".csv", ".txt"),
    )
    batdetect_output_count = count_files_with_suffixes(batdetect_output_dir, (".csv", ".json"))
    st.caption(f"Dataset audio folder: {dataset_audio_dir}")
    st.caption(f"Audio files already in dataset: {target_count}")
    st.caption(f"BirdNET output files: {birdnet_output_count}")
    st.caption(f"BatDetect2 output files: {batdetect_output_count}")
    if target_count and not birdnet_output_count and not batdetect_output_count:
        st.success("This dataset has audio and no analyzer outputs yet. It is ready to analyze.")
    elif target_count:
        st.info("This dataset already has audio and some analyzer outputs.")
    if st.button("Open dataset audio folder in Explorer"):
        dataset_audio_dir.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", str(dataset_audio_dir)])

    st.subheader("2. Import Audio Files")
    st.caption("Browse to the folder where the WAV/FLAC/MP3 files currently live, then copy them into the dataset audio folder.")
    default_source_audio = dataset_audio_dir if dataset_audio_dir.exists() else datasets_dir
    selected_source_path = folder_picker("Source audio folder", default_source_audio, "processing_source_audio")
    source_audio_dir = resolve_selected_audio_dir(selected_source_path)
    if source_audio_dir != selected_source_path:
        st.info(f"Using nested source audio folder: {source_audio_dir}")

    source_count = count_audio_files(source_audio_dir)
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
            st.success(f"Audio import complete: copied {copied}, skipped {skipped}, source files {total}.")

    audio_dir = dataset_audio_dir
    audio_count = count_audio_files(audio_dir)
    if audio_count == 0:
        st.warning("The dataset audio folder is empty. Copy audio files before running analyzers.")
    else:
        st.success(f"Ready to analyze {audio_count} audio files from the dataset audio folder.")

    st.subheader("3. Run Analyzers")

    birdnet_available = (
        shutil.which("birdnet-analyze")
        or shutil.which("birdnet-analyzer")
        or shutil.which("birdnet_analyzer")
    )
    default_birdnet_command = birdnet_available or "birdnet-analyze"
    birdnet_command = st.text_input("BirdNET command", value=default_birdnet_command)
    birdnet_lat, birdnet_lon = render_birdnet_location_picker()
    dataset_location_name = st.text_input("Dataset location name", value=dataset_name)
    if st.button("Save BirdNET location to manifest and database"):
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
        "BirdNET week date",
        value=default_birdnet_date,
        help="Pick a representative recording date. The ISO week number is passed to BirdNET as --week.",
    )
    birdnet_week = int(birdnet_date.isocalendar().week)
    st.caption(f"BirdNET week: {birdnet_week} from {birdnet_date:%Y-%m-%d}")
    birdnet_extra_args = st.text_input("BirdNET extra arguments", value="")
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
        ] + extra_args + [
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
        show_command_result(returncode, stdout, stderr)

    bat_cols = st.columns(2)
    with bat_cols[0]:
        bat_threshold = st.number_input("BatDetect2 threshold", min_value=0.01, max_value=0.99, value=0.30, step=0.05)
    with bat_cols[1]:
        bat_model = st.text_input("BatDetect2 model checkpoint", value="")

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
        st.code(" ".join(f'"{part}"' if " " in part else part for part in command), language="powershell")
        returncode, stdout, stderr = run_monitored_command(
            command,
            PROJECT_ROOT,
            "BatDetect2",
            audio_dir,
            batdetect_output_dir,
            (".csv", ".json"),
        )
        show_command_result(returncode, stdout, stderr)

    st.subheader("4. Update Manifest And Database")
    st.info(
        "After running an analyzer: first regenerate the manifest. "
        "Then use Append for a brand-new dataset that has never been imported. "
        "Use Rebuild listed datasets if you re-ran outputs for an existing dataset. "
        "Use Replace DB from manifest when you want to remove old datasets that are no longer in the manifest."
    )

    action_cols = st.columns(4)
    with action_cols[0]:
        if st.button("Regenerate manifest"):
            command = [
                sys.executable,
                str(SCRIPTS_DIR / "generate_manifest.py"),
                "--data-root",
                str(data_root),
                "--manifest",
                str(manifest_path),
            ]
            with st.spinner("Updating manifest..."):
                returncode, stdout, stderr = run_captured_command(command, PROJECT_ROOT)
            show_command_result(returncode, stdout, stderr)

    with action_cols[1]:
        if st.button("Append to database"):
            command = [
                sys.executable,
                str(SCRIPTS_DIR / "build_db.py"),
                "--data-root",
                str(data_root),
                "--manifest",
                str(manifest_path),
                "--db",
                str(resolved_db_path),
                "--append",
            ]
            with st.spinner("Appending detections and missing metrics..."):
                returncode, stdout, stderr = run_captured_command(command, PROJECT_ROOT)
            show_command_result(returncode, stdout, stderr)

    with action_cols[2]:
        if st.button("Rebuild listed datasets"):
            command = [
                sys.executable,
                str(SCRIPTS_DIR / "build_db.py"),
                "--data-root",
                str(data_root),
                "--manifest",
                str(manifest_path),
                "--db",
                str(resolved_db_path),
            ]
            with st.spinner("Rebuilding datasets from manifest..."):
                returncode, stdout, stderr = run_captured_command(command, PROJECT_ROOT)
            show_command_result(returncode, stdout, stderr)

    with action_cols[3]:
        if st.button("Replace DB from manifest"):
            command = [
                sys.executable,
                str(SCRIPTS_DIR / "build_db.py"),
                "--data-root",
                str(data_root),
                "--manifest",
                str(manifest_path),
                "--db",
                str(resolved_db_path),
                "--replace-db",
            ]
            with st.spinner("Replacing database from manifest..."):
                returncode, stdout, stderr = run_captured_command(command, PROJECT_ROOT)
            show_command_result(returncode, stdout, stderr)

    if manifest_path.exists():
        st.subheader("Manifest Preview")
        try:
            st.dataframe(pd.read_csv(manifest_path), use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not preview manifest: {exc}")


st.set_page_config(page_title=APP_TITLE, page_icon="🎙️", layout="wide")

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=220)
    st.markdown(f"### {APP_TITLE}")
    st.header("Data")
    db_path = st.text_input("Database path", value=os.environ.get("BIRDNET_DB", str(DEFAULT_DB_PATH)))
    data_root = Path(
        st.text_input("External data root", value=os.environ.get("BIRDNET_DATA_ROOT", str(PROJECT_ROOT)))
    ).expanduser().resolve()
    ensure_local_data_folders(data_root)
    view_mode = st.radio("View", ["Dashboard", "Processing", "Bird highlights", "Bat highlights"])

if view_mode == "Processing":
    render_processing_page(data_root, db_path)
    st.stop()

resolved_db_path = resolve_db_path(db_path, data_root)
if not resolved_db_path.exists():
    render_first_run_page(data_root, resolved_db_path)
    st.stop()

con = connect_database(resolved_db_path)

if has_platform_schema(con):
    datasets, df = load_platform_data(con)
else:
    datasets, df = load_legacy_data(con)

if df.empty:
    st.error("No detections found in database.")
    st.stop()

df["DateTime"] = pd.to_datetime(df["DateTime"])
all_detections = df.copy()
datasets = datasets.copy()
datasets["_sort_key"] = datasets["dataset_id"].map(dataset_sort_key)
datasets = datasets.sort_values("_sort_key").drop(columns=["_sort_key"])

points = location_points(datasets)

if view_mode == "Bird highlights":
    detector_values = all_detections["Detector"].fillna("birdnet").astype(str).str.lower()
    render_gallery(all_detections[detector_values == "birdnet"], data_root, "birdnet")
    st.stop()

if view_mode == "Bat highlights":
    detector_values = all_detections["Detector"].fillna("birdnet").astype(str).str.lower()
    render_gallery(all_detections[detector_values == "batdetect2"], data_root, "batdetect2")
    st.stop()

dataset_labels = {}
for _, row in datasets.iterrows():
    label = dataset_label(row)
    if label in dataset_labels:
        label = f"{label} ({row['dataset_id']})"
    dataset_labels[label] = row["dataset_id"]

selected_dataset_label = st.sidebar.selectbox("Dataset", list(dataset_labels.keys()))
selected_dataset_id = dataset_labels[selected_dataset_label]

with st.sidebar:
    if not points.empty:
        st.subheader("Locations")
        sidebar_deck = map_deck(points, selected_dataset_id=selected_dataset_id, satellite=False)
        st.pydeck_chart(sidebar_deck, use_container_width=True)
    else:
        st.caption("No dataset coordinates saved yet.")

df = df[df["dataset_id"] == selected_dataset_id].copy()
dataset_row = datasets[datasets["dataset_id"] == selected_dataset_id].iloc[0]

start_time = df["DateTime"].min()
end_time = df["DateTime"].max()

render_sticky_dashboard_header(
    dataset_row.get("location_name") or dataset_row.get("dataset_id"),
    f"{start_time:%d %b %Y} - {end_time:%d %b %Y}",
    f"{len(df):,}",
)

selected_points = points[points["dataset_id"] == selected_dataset_id]
if not selected_points.empty:
    st.subheader("Location")
    map_style = st.radio("Map style", ["Satellite", "Standard"], horizontal=True)
    if map_style == "Satellite":
        st.plotly_chart(satellite_map_figure(selected_points), use_container_width=True)
    else:
        selected_deck = map_deck(
            selected_points,
            selected_dataset_id=selected_dataset_id,
            satellite=False,
            zoom=12,
        )
        st.pydeck_chart(selected_deck, use_container_width=True)

species_counts = df["CommonName"].value_counts().reset_index()
species_counts.columns = ["Common Name", "Count"]
species_counts = species_counts.sort_values("Count", ascending=False)

st.header("1. Species Overview")

col1, col2 = st.columns([1, 2])

with col1:
    st.dataframe(species_counts, hide_index=True, use_container_width=True)

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
    fig.update_layout(showlegend=False, height=500)
    fig.update_xaxes(
        type="log",
        tickvals=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
        ticktext=["1", "2", "5", "10", "20", "50", "100", "200", "500", "1k", "2k", "5k", "10k"],
    )
    st.plotly_chart(fig, use_container_width=True)

render_detection_count_note()

st.header("2. Species Occurrence Matrix")

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

matrix_height = max(420, min(1200, 26 * len(occurrence_matrix)))
matrix_fig = px.imshow(
    occurrence_matrix,
    labels={"x": matrix_settings["label"], "y": "Species", "color": "Detections"},
    color_continuous_scale="YlGnBu",
    aspect="auto",
    text_auto=len(occurrence_matrix.columns) <= 60,
)
matrix_fig.update_layout(height=matrix_height, margin=dict(l=10, r=10, t=20, b=10))
matrix_fig.update_xaxes(side="top")
st.plotly_chart(matrix_fig, use_container_width=True)

st.header("3. Species Selection")

species_options = [f"{row['Common Name']} ({row['Count']})" for _, row in species_counts.iterrows()]
species_lookup = {f"{row['Common Name']} ({row['Count']})": row["Common Name"] for _, row in species_counts.iterrows()}

selected_display = st.selectbox("Select species", species_options)
selected_species = species_lookup[selected_display]

species_df = df[df["CommonName"] == selected_species]

if species_df.empty:
    st.warning("No detections for this species.")
    st.stop()

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
                st.image(profile["image_url"], use_container_width=True)
        with text_col:
            st.subheader(profile["title"])
            if profile["extract"]:
                st.write(profile["extract"])
            if profile["page_url"]:
                st.markdown(f"[Wikipedia]({profile['page_url']})")
        with map_col:
            if profile["range_map_url"]:
                st.image(profile["range_map_url"], use_container_width=True)

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

st.header("4. Temporal Pattern")

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
fig2 = px.bar(
    temporal_counts,
    x="TimeBin",
    y="Count",
    title=f"{temporal_title} detections of {selected_species}",
)
fig2.update_layout(xaxis_title=temporal_settings["axis"], yaxis_title="Detections")
st.plotly_chart(fig2, use_container_width=True)

st.header("5. Detection Inspector")

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
    "relative_audio_path",
    "BeginTime",
    "EndTime",
    "SpeciesCode",
]
visible_columns = [column for column in visible_columns if column in filtered_df.columns]

st.dataframe(filtered_df[visible_columns], use_container_width=True)

if filtered_df.empty:
    st.warning("No detections match the selected filters.")
    st.stop()

filtered_df["Display"] = (
    filtered_df.index.astype(str)
    + " | "
    + selected_species
    + " | "
    + filtered_df["Confidence"].round(3).astype(str)
)

selected_detection = st.selectbox("Select detection", filtered_df["Display"])
selected_index = int(selected_detection.split(" | ")[0])
row = filtered_df.loc[selected_index]
render_audio_inspector(
    row,
    data_root,
    str(row.get("Detector") or "birdnet").lower(),
    key_prefix=f"inspector_{selected_index}",
)
