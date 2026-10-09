"""Execute one group-split trainer and save outputs after every cell."""

import argparse
from pathlib import Path
import time

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("year", type=int, choices=(2018, 2022))
    args = parser.parse_args()
    matches = list(ROOT.glob(f"Classfier-LR55-CAPNetLite-QPSKlike-blind-RsCFO-11x15-{args.year}train20.ipynb"))
    notebook = nbformat.read(matches[0], as_version=4)
    output = ROOT / "results" / f"training_{args.year}_executed.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(notebook, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(PROJECT)}})
    with client.setup_kernel():
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {args.year} cell {i} START", flush=True)
            try:
                client.execute_cell(cell, i)
            finally:
                nbformat.write(notebook, output)
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {args.year} cell {i} DONE", flush=True)


if __name__ == "__main__":
    main()
