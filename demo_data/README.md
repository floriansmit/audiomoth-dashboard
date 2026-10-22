# Copenhagen Demo Data

This optional demo contains six short excerpts from the project author's AudioMoth field recordings:

- Three audible bird recordings: Barn Swallow, Tawny Owl, and Common Woodpigeon.
- Three ultrasonic bat recordings: Nathusius' Pipistrelle, Soprano Pipistrelle, and Serotine.

The analyzer tables are clipped from the corresponding Birda and BatDetect2 results and use timestamps relative to each excerpt. Bird excerpts are downsampled to 48 kHz to keep the download compact; ultrasonic bat excerpts retain the original 192 kHz sample rate. The location is represented as Copenhagen at city-level coordinates; original file paths are not included.

Run `python scripts/install_demo.py` from the repository root, or use **Install demo projects** on the dashboard's first-run screen. The installer copies these files into the normal `datasets/` folder, generates the manifest, builds the database, and calculates detection-level audio metrics.

## Recording Rights

Audio recordings are copyright 2026 Florian W.H. Smit. They are included for demonstration in AudioMoth Dashboard and may be redistributed unchanged as part of a fork or release of this project. Other reuse requires permission from the copyright holder. The software's MIT license does not grant separate rights to the recordings.

