# Troubleshooting

## The Dashboard Opens But Says There Is No Database

This is expected on a clean clone. Open `Processing` in the sidebar, create or select a dataset folder, run analyzers, regenerate the manifest, then build or append to the database.

## The Manifest Is Missing

The manifest is generated from dataset folders. It is not required before first launch.

From the app, use `Processing` -> `Regenerate manifest`.

From PowerShell:

```powershell
python scripts\generate_manifest.py --data-root "."
```

If there are no dataset folders yet, the command creates an empty manifest with headers.

## Install Fails With Access Denied

Use the default installer:

```powershell
python install_requirements.py
```

This creates `.venv/` inside the project and avoids writing to the global Python installation. Avoid `--system` unless you intentionally want a global install.

## Streamlit Shows Stale JavaScript Errors

Restart the dashboard and hard-refresh the browser. The launcher disables Streamlit's file watcher to reduce stale frontend chunks:

```powershell
python run_dashboard.py
```

## BirdNET Command Is Not Found

Run:

```powershell
python install_requirements.py
python run_dashboard.py
```

The app launcher uses the local `.venv/` when it exists. The BirdNET command is usually `birdnet-analyze`.

## Audio Playback Fails

Check that:

- the app's data root points to the folder that contains `datasets/`
- the matching audio file exists under the dataset's `audio/` folder
- the analyzer output filename matches the audio filename or timestamp

For projects on an external drive, start with:

```powershell
python run_dashboard.py --data-root "D:\AudioMoth Projects"
```
