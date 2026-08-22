# Data Structure

AudioMoth Dashboard expects a data root with a `datasets/` folder. The app defaults to the project root, but you can point it at an external drive with `python run_dashboard.py --data-root "D:\AudioMoth Projects"`.

## Recommended Layout

```text
datasets/
  datasets_manifest.csv
  birdnet.duckdb
  01_site_name/
    audio/
    birdnet/
    batdetect2_outputs/
```

Folder purpose:

- `audio/`: original WAV, FLAC, or MP3 recordings.
- `birdnet/`: BirdNET Analyzer selection-table outputs.
- `batdetect2_outputs/`: BatDetect2 CSV outputs.
- `datasets_manifest.csv`: generated dataset manifest.
- `birdnet.duckdb`: generated dashboard database.

## BirdNET Output

The dashboard accepts original BirdNET Analyzer selection tables with columns like:

```text
Selection  View  Channel  Begin Time (s)  End Time (s)  Low Freq (Hz)  High Freq (Hz)  Species Code  Common Name  Confidence
```

The original BirdNET files do not need audio paths. `scripts/build_db.py` links each table to the matching audio file by filename or timestamp.

## BatDetect2 Output

BatDetect2 outputs should stay in `batdetect2_outputs/`. The builder imports these as detector `batdetect2`, so bird and bat detections remain traceable to the model that produced them.

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
- `audio_dir`

`location_name`, `latitude`, and `longitude` can be edited from the Processing page. Existing manual values are preserved when regenerating the manifest and the `dataset_id` still matches.

## Database

Build or update the database from the Processing page, or run:

```powershell
python scripts\build_db.py --data-root "."
```

Use `--append` for brand-new datasets. Use the normal rebuild command when analyzer outputs changed for datasets already in the database.

## GitHub Safety

The repository ignores everything below `datasets/` except `.gitkeep`. This prevents real audio files, coordinates, analyzer outputs, and generated databases from being published accidentally.
