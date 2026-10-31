"""Convertit les sources au format « percent » (fichiers .py de sources/) en notebooks Jupyter.

Usage :
    python tools/build_notebooks.py            # convertit tous les fichiers de sources/
    python tools/build_notebooks.py --execute  # convertit puis exécute chaque notebook
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "sources"
OUT = ROOT / "notebooks"

CELL_MARKER = re.compile(r"^#\s*%%\s*(\[markdown\])?\s*$")


def to_markdown(lines: list[str]) -> list[str]:
    """Retire le préfixe « # » utilisé pour écrire du Markdown dans un fichier .py."""
    out: list[str] = []
    for line in lines:
        if line.startswith("# "):
            out.append(line[2:])
        elif line.strip() == "#":
            out.append("")
        else:
            out.append(line)
    return out


def strip_trailing_blank(lines: list[str]) -> list[str]:
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def parse(source: str) -> list[dict]:
    cells: list[dict] = []
    current_kind = "code"
    current: list[str] = []
    seen_marker = False

    for line in source.splitlines():
        match = CELL_MARKER.match(line)
        if match:
            if seen_marker:
                cells.append({"kind": current_kind, "lines": current})
            seen_marker = True
            current_kind = "markdown" if match.group(1) else "code"
            current = []
            continue
        current.append(line)

    if seen_marker:
        cells.append({"kind": current_kind, "lines": current})

    if not cells:
        raise ValueError("Aucune cellule trouvée : le fichier doit contenir '# %%'.")

    nb_cells = []
    for index, cell in enumerate(cells):
        lines = strip_trailing_blank(cell["lines"])
        while lines and not lines[0].strip():
            lines.pop(0)
        if not lines:
            continue
        if cell["kind"] == "markdown":
            lines = to_markdown(lines)
        text = "\n".join(lines) + "\n"
        nb_cells.append(
            {
                "cell_type": cell["kind"],
                "id": f"cell-{index:02d}",
                "metadata": {},
                "source": text.splitlines(keepends=True),
                **({"outputs": [], "execution_count": None} if cell["kind"] == "code" else {}),
            }
        )
    return nb_cells


def build(source_file: Path) -> Path:
    cells = parse(source_file.read_text(encoding="utf-8"))
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.mkdir(exist_ok=True)
    target = OUT / f"{source_file.stem}.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return target


def execute(path: Path) -> None:
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbformat.write(nb, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()

    sources = [SRC / f for f in args.files] if args.files else sorted(SRC.glob("*.py"))
    if not sources:
        print(f"Aucune source trouvée dans {SRC}")
        return 1

    for source in sources:
        target = build(source)
        print(f"construit   {target.relative_to(ROOT)}")
        if args.execute:
            execute(target)
            print(f"exécuté     {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
