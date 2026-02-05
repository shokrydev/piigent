---
description: Run a Python script using the project's virtual environment
---

1. Check if `.venv` exists.
2. If it exists, run the command using `.venv/bin/python`.
   `Run: .venv/bin/python [script_path] [args]`
3. If it does not exist, ask the user or try to create it (if authorized).
   **DO NOT** run with system `python3` unless explicitly instructed to ignore the venv.

// turbo
4. If the previous command failed with "ModuleNotFoundError", check if you need to install dependencies.
