# Data Structure

AudioMoth Dashboard expects a data root with a `datasets/` folder. The app defaults to the project root, but you can point it at an external drive with `python run_dashboard.py --data-root "D:\AudioMoth Projects"`.

## Recommended Layout

```text
datasets/
  datasets_manifest.csv
  audiomoth.duckdb
  01_site_name/
    audio/
    birda/
    birdnet/
    batdetect2_outputs/
```

Folder purpose:

- `audio/`: original WAV, FLAC, or MP3 recordings.
- `birda/`: Birda CSV outputs.
- `birdnet/`: BirdNET Analyzer selection-table outputs.
- `batdetect2_outputs/`: BatDetect2 CSV outputs.
- `datasets_manifest.csv`: generated dataset manifest.
- `audiomoth.duckdb`: generated dashboard database.

## BirdNET Output

The dashboard accepts original BirdNET Analyzer selection tables with columns like:

```text
Selection  View  Channel  Begin Time (s)  End Time (s)  Low Freq (Hz)  High Freq (Hz)  Species Code  Common Name  Confidence
```

The original BirdNET files do not need audio paths. `scripts/build_db.py` links each table to the matching audio file by filename or timestamp.

## Birda Output

Birda outputs should stay in `birda/`. The builder imports Birda CSV files with columns like:

```text
Start (s)  End (s)  Scientific name  Common name  Confidence  File
```

Birda is the recommended backend when performance matters because its Windows CUDA build can use NVIDIA GPUs and it skips existing outputs unless `--force` is used. Install Birda separately from [tphakala/birda](https://github.com/tphakala/birda), then use the Processing page to run it.

## BatDetect2 Output

BatDetect2 outputs should stay in `batdetect2_outputs/`. The builder imports these as detector `batdetect2`, so bird and bat detections remain traceable to the model that produced them.

When run through the dashboard or `scripts\run_batdetect2.py`, BatDetect2 skips audio files that already have matching `.csv` or `.json` outputs in `batdetect2_outputs/`. Audio subfolders are supported. The helper stages pending files into a temporary flat folder for BatDetect2, so filenames must be unique across subfolders. For the older BatDetect2 CLI, empty prediction files are saved too, which keeps progress tracking aligned with all analyzed recordings. Use `--rerun-existing` only when you want to recompute all BatDetect2 outputs.

## Manifest

Generate the manifest from the Processing page or with:

```powershell
python scripts\generate_manifest.py --data-root "."
```

The manifest includes:

- `dataset_id`
- `dataset_type`
- `location_name`
- `latitude`
- `longitude`
- `period_label`
- `birdnet_dir`
- `batdetect2_dir`
- `birda_dir`
- `audio_dir`

`location_name`, `latitude`, and `longitude` can be edited from the Processing page. Existing manual values are preserved when regenerating the manifest and the `dataset_id` still matches.

## Database

Build or update the database from the Processing page, or run:

```powershell
python scripts\build_db.py --data-root "."
```

Use `--append` for brand-new datasets. Use the normal rebuild command when analyzer outputs changed or new audio files were added for datasets already in the database.

## GitHub Safety

The repository ignores everything below `datasets/` except `.gitkeep`. This prevents real audio files, coordinates, analyzer outputs, and generated databases from being published accidentally.
