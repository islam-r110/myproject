# Structural Clean-up Guide

This guide explains how to simplify your VS Code workspace so the production app is cleanly deployed from the repository root.

Steps (safe, reversible):

1. Backup current workspace (recommended)
   - Copy your project folder somewhere safe if you want a rollback.

2. Keep only production files at the repository root
   - Required production files (move these to the root):
     - `main.py`  (this is the Streamlit app entrypoint)
     - `dubai_real_estate_data_realistic_500.csv` (data file)
     - `requirements.txt` (created alongside this guide)

3. Remove or relocate confusing sub-folders and validation scripts
   - If you have a sub-folder named `ADADAD` or similar that contains copies of `main.py`/`app.py`/`validate_main.py`, move any *development-only* scripts into a `dev/` or `tools/` folder.
   - Example commands (PowerShell):

```powershell
# create a dev folder, move validation scripts there
mkdir .\dev
move .\validate_main.py .\dev\validate_main.py
move .\app.py .\dev\app.py
# if ADADAD folder exists and contains non-production files
move .\ADADAD .\dev\ADADAD
```

4. Remove duplicate/conflicting imports
   - Ensure only the root `main.py` is used by Streamlit. Delete or rename any alternate `main.py` copies under subfolders.

5. Confirm top-level file layout
   - After clean-up, your workspace root should show at minimum:
     - `main.py`
     - `dubai_real_estate_data_realistic_500.csv`
     - `requirements.txt`

6. Test locally (no venv required)
   - From repository root run:

```powershell
streamlit run main.py
```

7. Streamlit Cloud notes
   - Streamlit Cloud runs `pip install -r requirements.txt` from the repo root. Ensure `requirements.txt` is at root and lists all runtime packages.
   - Do not commit/keep a virtual environment folder in the repo (like `.venv/`); add it to `.gitignore`.

8. Quick troubleshooting
   - If you see `ModuleNotFoundError: No module named seaborn` on Streamlit Cloud, confirm `seaborn` is present in `requirements.txt` and committed.
   - If there are multiple `main.py` files, Streamlit may pick the wrong one; keep only the root `main.py` or rename dev copies.

That's it — after these steps your repository root is clean and ready for Streamlit Cloud.
