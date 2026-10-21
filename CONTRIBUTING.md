# Contributing

Contributions, bug reports, and documentation improvements are welcome.

## Before Opening an Issue

1. Check the existing issues and `docs/troubleshooting.md`.
2. Reproduce the problem with the latest `main` branch.
3. Remove private recordings, exact coordinates, usernames, and local paths from screenshots and logs.

For a bug report, include the operating system, Python version, analyzer and model used, the command you ran, the full error message, and a minimal description of the dataset layout. Do not upload field recordings unless you have permission to share them.

## Development Setup

```powershell
git clone https://github.com/floriansmit/audiomoth-dashboard.git
cd audiomoth-dashboard
python install_requirements.py
python run_dashboard.py
```

Keep generated datasets, databases, analyzer outputs, Wikipedia cache files, and audio outside Git. The repository `.gitignore` covers the standard locations and common audio formats.

## Pull Requests

- Keep changes focused and explain the user-facing behavior.
- Follow the existing Python and Streamlit patterns.
- Run the Python compile checks before submitting:

```powershell
python -m py_compile dashboard\app.py scripts\build_db.py scripts\generate_manifest.py scripts\run_batdetect2.py install_requirements.py run_dashboard.py
```

- Update the README or relevant document when behavior, setup, or the data layout changes.
- Never commit real recordings, generated databases, private coordinates, credentials, or model files.

By contributing, you agree that your contribution is licensed under the repository's MIT License.
