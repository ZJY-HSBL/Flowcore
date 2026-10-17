"""Guard the accuracy and renderability of the README CI visualizations."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
FIGURES = [
    "assets/flowcore-pipeline.svg",
    "assets/v05-delivery.svg",
    "assets/v06-stress.svg",
    "assets/v09-controller.svg",
    "assets/v10-generalization.svg",
]


def test_readme_referenced_svg_figures_are_valid_xml():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for relative in FIGURES:
        path = ROOT / relative
        assert path.exists(), relative
        assert relative in readme
        root = ET.parse(path).getroot()
        assert root.tag.endswith("svg")
        assert root.attrib.get("viewBox")
        assert any(node.tag.endswith("title") for node in root)


def test_reference_dataset_has_consistent_provenance_and_values():
    data = json.loads(
        (ROOT / "results" / "ci_reference_summary.json").read_text(encoding="utf-8")
    )
    assert data["schema_version"] == 1
    for release in ("v05", "v06", "v07", "v08", "v09", "v10"):
        assert data["provenance"][release]["run"].startswith(
            "https://github.com/ZJY-HSBL/Flowcore/actions/runs/"
        )

    v05 = {item["name"]: item for item in data["v05"]["backend"]}
    assert v05["Circulant-Kronecker"]["delivery_pct"] == 99.88

    v06 = {item["name"]: item for item in data["v06"]["family"]}
    assert v06["Circulant-Kronecker"]["arbitrary_slot_accuracy_pct"] == 27.47

    v09 = data["v09"]
    assert v09["ranks"][0]["params"] == 30
    assert all(r["task_exact_accuracy_pct"] == 100 for r in v09["ranks"])

    v10 = data["v10"]
    average = sum(v10["soft_mse_by_seed"]) / 3
    assert abs(average - v10["soft_mse_mean"]) < 1e-6
    assert v10["hard_mse_mean"] == 0
    assert v10["whole_permutation_accuracy_pct"] == 100
    assert v10["double_noise_slot_accuracy_pct"] == 100
