"""Second patch: rename the saved/loaded checkpoint from the old
capnet_lite_c{C1}_{C2}_{SPLIT_MODE}.pt (parameterized by width, now
misleading since the narrow-deep architecture no longer matches its own
C1/C2 literally embedded elsewhere) to a fixed capnet_lite_narrowdeep_
{SPLIT_MODE}.pt name, matching the new "capnet_lite_narrowdeep" cap_model_key
added to helper_function/fully_online_pipeline_inference.py. Also updates the
matching FullyOnline-Inference notebooks to request that key and filename.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

OLD_NAME_EXPR = 'f"capnet_lite_c{C1}_{C2}_{SPLIT_MODE}.pt"'
NEW_NAME_EXPR = 'f"capnet_lite_narrowdeep_{SPLIT_MODE}.pt"'


def source(cell):
    return "".join(cell["source"])


def set_source(cell, value):
    cell["source"] = value.splitlines(keepends=True)


def replace_once(text, old, new, where):
    n = text.count(old)
    if n != 1:
        raise AssertionError(f"{where}: expected exactly one occurrence, found {n}")
    return text.replace(old, new)


def patch_trainer(path):
    nb = json.loads(path.read_text())
    cells = nb["cells"]
    for idx in (5, 7):
        c = source(cells[idx])
        c = replace_once(c, OLD_NAME_EXPR, NEW_NAME_EXPR, f"{path.name} cell{idx}")
        set_source(cells[idx], c)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    print("patched", path)


def patch_inference(path, year):
    nb = json.loads(path.read_text())
    cells = nb["cells"]
    c1 = source(cells[1])
    c1 = replace_once(c1, 'cap_model_key="capnet_lite"', 'cap_model_key="capnet_lite_narrowdeep"',
                       f"{path.name} cap_model_key")
    old_assert = f'assert pipeline.cap_path.name == "capnet_lite_c32_64_{year}train20.pt"'
    new_assert = f'assert pipeline.cap_path.name == "capnet_lite_narrowdeep_{year}train20.pt"'
    c1 = replace_once(c1, old_assert, new_assert, f"{path.name} cap_path assertion")
    set_source(cells[1], c1)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    print("patched", path)


def main():
    for year in (2018, 2022):
        patch_trainer(HERE / f"Classfier-LR55-CAPNetLite-QPSKlike-blind-RsCFO-11x15-{year}train20.ipynb")
        patch_inference(HERE / f"FullyOnline-Inference-LR55-CAPNetLite-{year}train20.ipynb", year)


if __name__ == "__main__":
    main()
