"""Execute the laboratory 3 notebook in place and verify its text."""

import os
from pathlib import Path

import nbformat
from check_lab3_claims import check_claims
from nbclient import NotebookClient

from emlab.lab3_config import DEFAULT_CONFIG, KERNEL_NAME, NOTEBOOK_PATH

ROOT = Path(__file__).resolve().parents[1]


def report(cell, cell_index, **kwargs):
    print(f"Completed cell {cell_index}", flush=True)


def main():
    os.environ["JUPYTER_PATH"] = str(ROOT / ".venv/share/jupyter")
    path = ROOT / NOTEBOOK_PATH
    notebook = nbformat.read(path, as_version=4)
    try:
        NotebookClient(
            notebook,
            timeout=DEFAULT_CONFIG.notebook_timeout_seconds,
            resources={"metadata": {"path": str(ROOT)}},
            kernel_name=KERNEL_NAME,
            on_cell_executed=report,
        ).execute()
    finally:
        # Also after a failure: the notebook then shows the cell that broke
        # and its traceback. The original exception still propagates.
        nbformat.write(notebook, path)
    print("Notebook executed successfully")
    check_claims(ROOT)
    print("Statements of the notebook text agree with the results")


if __name__ == "__main__":
    main()
