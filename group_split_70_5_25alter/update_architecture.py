"""One-shot patch: swap the group-split trainer notebooks' CAPNetLite for the
narrow-deep + SiLU variant found by the seed/architecture search (see
results/final2_all.csv). Seed handling is untouched -- these notebooks still
do not fix an RNG seed, matching the original design intent.

Architecture change: c1=32,c2=64,ReLU (111,716 params) ->
c1=16,c2=52,+1 extra ResBlock,SiLU (110,412 params).
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

OLD_WIDTHS = "C1 = 32\nC2 = 64"
NEW_WIDTHS = "C1 = 16\nC2 = 52"

OLD_MODEL_BLOCK = '''# ---------------------------------------------------------------------------
# CAPNetLite: a small FPGA-friendly ResNet-style CNN.
#
# Design choices for ZCU104 PL friendliness:
#   - a single shared trunk + one small FC head
#   - plain 3x3 convs with static padding=1
#   - ReLU activation
#   - BatchNorm folds into the preceding conv at inference
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out, inplace=True)


class CAPNetLite(nn.Module):
    def __init__(self, n_classes=4, c1=32, c2=64):
        super().__init__()
        self.stem = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(c1)
        self.block1 = ResBlock(c1)
        self.down = nn.Conv2d(c1, c2, kernel_size=3, stride=(1, 2), padding=1, bias=False)
        self.down_bn = nn.BatchNorm2d(c2)
        self.block2 = ResBlock(c2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c2, n_classes)

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem(x)), inplace=True)
        x = self.block1(x)
        x = F.relu(self.down_bn(self.down(x)), inplace=True)
        x = self.block2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)'''

NEW_MODEL_BLOCK = '''# ---------------------------------------------------------------------------
# CAPNetLite: a small FPGA-friendly ResNet-style CNN.
#
# Design choices for ZCU104 PL friendliness:
#   - a single shared trunk + one small FC head
#   - plain 3x3 convs with static padding=1
#   - SiLU activation
#   - BatchNorm folds into the preceding conv at inference
#
# Narrow-deep variant (c1=16, c2=52, +1 extra ResBlock at c2, SiLU) found by
# the seed/architecture search in response_to_spl/r1c1/group_split_70_5_25;
# 110,412 params, under the original 111,716-param budget.
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        identity = x
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.silu(out)


class CAPNetLite(nn.Module):
    def __init__(self, n_classes=4, c1=16, c2=52):
        super().__init__()
        self.stem = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(c1)
        self.block1 = ResBlock(c1)
        self.down = nn.Conv2d(c1, c2, kernel_size=3, stride=(1, 2), padding=1, bias=False)
        self.down_bn = nn.BatchNorm2d(c2)
        self.block2 = ResBlock(c2)
        self.block3 = ResBlock(c2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c2, n_classes)

    def forward(self, x):
        x = F.silu(self.stem_bn(self.stem(x)))
        x = self.block1(x)
        x = F.silu(self.down_bn(self.down(x)))
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)'''


def source(cell):
    return "".join(cell["source"])


def set_source(cell, value):
    cell["source"] = value.splitlines(keepends=True)


def replace_once(text, old, new, where):
    n = text.count(old)
    if n != 1:
        raise AssertionError(f"{where}: expected exactly one occurrence, found {n}")
    return text.replace(old, new)


def patch(path):
    nb = json.loads(path.read_text())
    cells = nb["cells"]

    c1 = source(cells[1])
    c1 = replace_once(c1, OLD_WIDTHS, NEW_WIDTHS, "cell1 widths")
    set_source(cells[1], c1)

    c5 = source(cells[5])
    c5 = replace_once(c5, OLD_MODEL_BLOCK, NEW_MODEL_BLOCK, "cell5 model block")
    set_source(cells[5], c5)

    intro = source(cells[0])
    marker = "\n\nThis R1-C1 copy assigns **signal samples** at 70/5/25"
    if marker in intro and "narrow-deep" not in intro:
        intro = intro.replace(
            marker,
            "\n\nThis notebook uses the narrow-deep SiLU CAPNetLite variant "
            "(c1=16, c2=52, +1 extra ResBlock, 110,412 params) found by the "
            "seed/architecture search; see `results/final2_all.csv`."
            + marker,
        )
        set_source(cells[0], intro)

    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    print("patched", path)


def main():
    for year in (2018, 2022):
        patch(HERE / f"Classfier-LR55-CAPNetLite-QPSKlike-blind-RsCFO-11x15-{year}train20.ipynb")


if __name__ == "__main__":
    main()
