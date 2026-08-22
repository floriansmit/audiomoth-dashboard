import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DASHBOARD_APP = PROJECT_ROOT / "dashboard" / "app.py"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
APP_TITLE = "AudioMoth Dashboard"


def dashboard_python():
    if VENV_PYTHON.exists():
        return VENV_PYTHON
    return Path(sys.executable)


def main():
    parser = argparse.ArgumentParser(description=f"Start {APP_TITLE}.")
    parser.add_argument("--port", type=int, default=8502, help="Local Streamlit port.")
    parser.add_argument("--data-root", help="Folder containing datasets, manifest, and audio files.")
    parser.add_argument("--db", help="DuckDB database path.")
    args = parser.parse_args()

    env = os.environ.copy()
    if args.data_root:
        env["AUDIOMOTH_DATA_ROOT"] = str(Path(args.data_root).resolve())
    if args.db:
        env["AUDIOMOTH_DB"] = str(Path(args.db).resolve())

    python_path = dashboard_python()
    command = [
        str(python_path),
        "-m",
        "streamlit",
        "run",
        str(DASHBOARD_APP),
        "--server.port",
        str(args.port),
        "--server.address",
        "localhost",
        "--server.fileWatcherType",
        "none",
        "--browser.gatherUsageStats",
        "false",
    ]

    (PROJECT_ROOT / "datasets").mkdir(parents=True, exist_ok=True)

    print(f"Starting {APP_TITLE} on http://localhost:{args.port}")
    print(f"Project folder: {PROJECT_ROOT}")
    print(f"Using Python: {python_path}")
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
