# Contributing to neokine

Thanks for your interest! neokine is a small pose-visualization demo with two
independent front-ends to the same idea:

- **Web app** — `docs/` (static HTML/CSS/JS; MediaPipe loaded from a CDN).
- **Streamlit app** — `app/` (local Python; MediaPipe on-device).

Please keep the two front-ends independent and avoid coupling them. Shared
landmark constants live in `shared/skeleton.py`.

> **Scope reminder.** neokine is a *visualization* demo, not a measurement or
> diagnostic tool, and makes no clinical claims. Please keep contributions and
> wording consistent with that scope (see the disclaimer in the README).

## Development setup

### Web app (`docs/`)

No install. ES modules require `http://`, so serve the folder rather than
opening the file directly:

```bash
python3 -m http.server 8000 --directory docs
# open http://localhost:8000
```

### Streamlit app (`app/`)

```bash
conda env create -f environment.yml && conda activate neokine
# or: python3 -m venv .venv && .venv/bin/pip install -r requirements-app.txt

streamlit run app/streamlit_app.py
```

Python 3.9–3.12 only (`mediapipe==0.10.14` has no wheels for 3.13+).

## Pull requests

1. Fork and create a topic branch.
2. Keep changes focused; describe *what* and *why* in the PR description.
3. Manually verify the front-end(s) you touched still run (steps above).
4. If you change a dependency, update both `environment.yml` and
   `requirements-app.txt`.

## Reporting issues

Open a GitHub issue with steps to reproduce, your OS/Python/browser, and — for
the Streamlit app — the MediaPipe model variant and inputs used.

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE), the same license that covers this project.
