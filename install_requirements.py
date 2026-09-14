import argparse
import shutil
import subprocess
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
VENV_DIR = APP_DIR / ".venv"
BIRDA_RELEASES_URL = "https://github.com/tphakala/birda/releases/latest"
BIRDA_WINDOWS_GUIDE_URL = "https://github.com/tphakala/birda/blob/main/docs/windows-guide.md"


def run_birda_check(command, label):
    try:
        result = subprocess.run(
            command,
            cwd=APP_DIR,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  {label}: check failed ({exc})")
        return False

    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if output:
        for line in output.splitlines():
            print(f"    {line}")
    if result.returncode != 0:
        print(f"  {label}: command exited with code {result.returncode}")
        return False
    return True


def print_birda_install_instructions():
    print("\nBirda is optional and was not found on PATH.")
    print("The dashboard and BirdNET Analyzer still work without it, but Run Birda will be unavailable.")
    print("\nRecommended Windows setup:")
    print(f"  1. Download the Windows CUDA installer or ZIP: {BIRDA_RELEASES_URL}")
    print("  2. Install it, or extract the ZIP to a permanent folder such as C:\\Tools\\birda.")
    print("  3. Add the folder containing birda.exe to your user PATH, then open a new PowerShell window.")
    print("  4. Verify the executable: birda --version")
    print("  5. Install a model: birda models install birdnet-v24")
    print("  6. Inspect acceleration support: birda providers")
    print(f"\nDetailed Windows, CUDA, and TensorRT guide: {BIRDA_WINDOWS_GUIDE_URL}")


def check_birda():
    print("\nChecking optional Birda integration...")
    birda_path = shutil.which("birda")
    if not birda_path:
        print_birda_install_instructions()
        return False

    print(f"Birda executable: {birda_path}")
    version_ok = run_birda_check([birda_path, "--version"], "Version")

    print("  Configured model files:")
    models_ok = run_birda_check([birda_path, "models", "check"], "Models")
    if not models_ok:
        print("  Install the recommended model with: birda models install birdnet-v24")

    print("  Available inference providers:")
    providers_ok = run_birda_check([birda_path, "providers"], "Providers")
    if not providers_ok:
        print("  Run 'birda providers' in a new PowerShell window to diagnose GPU support.")

    if version_ok and models_ok:
        print("Birda is ready for use from the dashboard.")
    else:
        print(f"Birda needs attention. See: {BIRDA_WINDOWS_GUIDE_URL}")
    return version_ok and models_ok


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
    parser.add_argument(
        "--skip-birda-check",
        action="store_true",
        help="Skip the optional Birda executable, model, and provider checks.",
    )
    args = parser.parse_args()

    print(f"Using Python: {sys.executable}")
    install(APP_DIR / "requirements.txt", args.installer, args.system)
    print("Requirements installed.")
    if not args.system:
        print(f"Dashboard environment: {venv_python()}")
    if not args.skip_birda_check:
        check_birda()


if __name__ == "__main__":
    main()
