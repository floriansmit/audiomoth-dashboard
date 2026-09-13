import argparse
import shutil
import subprocess
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
VENV_DIR = APP_DIR / ".venv"


def venv_python():
    if sys.platform.startswith("win"):
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def create_venv(installer):
    python_path = venv_python()
    if python_path.exists():
        return python_path

    print(f"Creating virtual environment: {VENV_DIR}")
    if installer in {"auto", "uv"} and shutil.which("uv") is not None:
        subprocess.run(["uv", "venv", str(VENV_DIR), "--python", sys.executable], cwd=APP_DIR, check=True)
    else:
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], cwd=APP_DIR, check=True)

    if not python_path.exists():
        raise RuntimeError(f"Virtual environment Python was not created: {python_path}")
    return python_path


def uv_command(requirements_path, python_path):
    return [
        "uv",
        "--system-certs",
        "pip",
        "install",
        "--python",
        str(python_path),
        "-r",
        str(requirements_path),
    ]


def pip_command(requirements_path, python_path):
    return [
        str(python_path),
        "-m",
        "pip",
        "install",
        "-r",
        str(requirements_path),
    ]


def install(requirements_path, installer, use_system):
    if not requirements_path.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_path}")

    python_path = Path(sys.executable) if use_system else create_venv(installer)
    uv_available = shutil.which("uv") is not None
    if installer == "auto":
        installer = "uv" if uv_available else "pip"
    if installer == "uv" and not uv_available:
        raise RuntimeError("uv was requested, but the 'uv' command was not found on PATH.")

    command = uv_command(requirements_path, python_path) if installer == "uv" else pip_command(requirements_path, python_path)
    print(f"Installing packages from {requirements_path.name} with {installer}")
    print(f"Target Python: {python_path}")
    subprocess.run(command, cwd=APP_DIR, check=True)


def main():
    parser = argparse.ArgumentParser(description="Install AudioMoth Dashboard requirements.")
    parser.add_argument(
        "--installer",
        choices=("auto", "uv", "pip"),
        default="auto",
        help="Package installer to use. Auto prefers uv when available.",
    )
    parser.add_argument(
        "--system",
        action="store_true",
        help="Install into the current Python instead of the local .venv. Not recommended.",
    )
    args = parser.parse_args()

    print(f"Using Python: {sys.executable}")
    install(APP_DIR / "requirements.txt", args.installer, args.system)
    print("Requirements installed.")
    if not args.system:
        print(f"Dashboard environment: {venv_python()}")


if __name__ == "__main__":
    main()
