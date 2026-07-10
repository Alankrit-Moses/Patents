# Pattern Mining Results Viewer

Static viewer for the framework, pattern cards, experiment construction, prompt/pipeline skeletons, robustness results, score statistics, and evaluation drilldowns.

## Open locally

```powershell
cd experiments\results_viewer
python -m http.server 8765 --bind 127.0.0.1
```

Then open `http://127.0.0.1:8765/index.html`.

## Regenerate data

Run this from the repository root after result artifacts change:

```powershell
python experiments\results_viewer\build_viewer_data.py
```

The generated `viewer_data.js` is intentionally committed/stored with the viewer so the app can run as a static page without a backend.
