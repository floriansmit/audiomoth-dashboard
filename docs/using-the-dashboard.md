# Using the Dashboard

Select a project in the sidebar and press **Open Selected Project**. Bird and bat views remain separate even when they share the same source audio folder.

## Dashboard Tabs

- **Overview** shows the project location, date range, observation count, and an interactive map.
- **Histogram** compares species detection counts. Counts represent detector activity, not animal abundance.
- **Matrix** shows species activity across hours, days, or weeks. Images are loaded when a Wikipedia thumbnail is available.
- **Species** combines the selected species with a short Wikipedia profile, image, and distribution map when available.
- **Temporal Pattern** plots the selected species at an independently chosen time resolution, including minute-level detail.
- **Detection Inspector** filters individual detections by confidence. Select a table row to load the linked recording, audio controls, and spectrogram.

## Audio And Clips

The inspector prefers a generated clip when one covers the selected detection. Otherwise, it reads only the required interval from the original recording. Bird audio plays at its recorded speed. Bat audio can be time-expanded by 10x, lowering both pitch and playback speed so ultrasonic calls become audible.

Generated clips are optional. The Processing page ranks candidates using classifier confidence, estimated signal-to-noise ratio, and background level, then reports the expected storage reduction before writing files. Original recordings are never deleted automatically.

## Reading Bat Annotations

For BatDetect2 rows, the spectrogram adds small colored triangles above every predicted pulse in the visible time window. The legend maps each color to the model's predicted species. Predictions at nearly the same time are separated vertically so they remain visible.

The gold brace groups nearby detections in the selected pulse's frequency band. The red arrow identifies the selected pulse. Text below the figure reports the number of grouped pulses, median spacing, and a plain-language description of the selected pulse shape.

These annotations are acoustic clues, not confirmed behavior or individual-animal counts. Several predictions may describe one passing bat, simultaneous model labels may disagree, and search, approach, feeding, and social behavior cannot be established from pulse spacing alone.

## Highlights

Bird and bat highlights can be generated for the current project, selected projects, or all projects. The galleries rank clear, confident detections and preload their spectrograms with a progress indicator. Selecting an item opens its audio and detailed spectrogram view.

## Processing

The Processing page guides project setup, audio import, analyzer selection, manifest generation, database updates, audio-quality analysis, and clip creation. Use **Update selected project** after adding or replacing analyzer outputs; it refreshes only the selected folder's bird and bat rows. Whole-database operations remain under advanced actions.
