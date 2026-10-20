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
    "assets/v11-unseen-views.svg",
    "assets/v12-structural-metric.svg",
    "assets/v13-loss-ablation.svg",
    "assets/v14-global-margin.svg",
    "assets/v15-graph-routing.svg",
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

    v11 = data["v11"]
    assert v11["learned_fixed_view"]["unseen_slot_accuracy_pct"][0] == 23.20
    assert v11["invariant_distance"]["unseen_slot_accuracy_pct"][0] == 100
    assert v11["invariant_distance"]["unseen_slot_accuracy_pct"][-1] == 81.68
    assert v11["invariant_distance"]["trainable_parameters"] == 0

    v12 = data["v12"]
    assert v12["trained_metric_parameters"] == 8
    assert v12["baseline_metric_parameters"] == 0
    six = v12["module_count"]["6"]
    eight = v12["module_count"]["8"]
    assert six["learned_quantile"]["slot_accuracy_pct"][2] == 74.22
    assert six["uniform_quantile"]["slot_accuracy_pct"][2] == 77.86
    assert eight["learned_quantile"]["slot_accuracy_pct"][2] == 69.27
    assert eight["uniform_quantile"]["slot_accuracy_pct"][2] == 76.63

    v13 = data["v13"]
    assert v13["learned_parameters_per_controller"] == 8
    assert v13["zero_parameter_baselines"] == 2
    assert v13["conditions"]["6"]["margin"]["slot_accuracy_pct"][2] == 77.08
    assert v13["conditions"]["8"]["margin"]["slot_accuracy_pct"][2] == 74.48
    assert v13["conditions"]["8"]["uniform_quantile"]["slot_accuracy_pct"][2] == 76.63

    v14 = data["v14"]
    assert v14["learned_parameter_count"] == 8
    assert v14["zero_parameter_baselines"] == 2
    assert v14["conditions"]["6"]["global_margin"]["slot_accuracy_pct"][2] == 77.43
    assert v14["conditions"]["8"]["global_margin"]["slot_accuracy_pct"][2] == 74.41
    assert v14["conditions"]["8"]["uniform_quantile"]["slot_accuracy_pct"][2] == 76.63


def test_v15_graph_result_provenance():
    data = json.loads((ROOT / "results" / "ci_reference_summary.json").read_text())
    assert data["v15"]["conditions"]["8"]["fingerprint"]["slot_accuracy_pct"][2] == 75.72
    assert data["v15"]["conditions"]["8"]["graph_unweighted"]["slot_accuracy_pct"][2] == 92.06
    assert data["v15"]["conditions"]["8"]["graph_unweighted"]["permutation_exact_pct"][2] == 83.85
    assert data["v15"]["conditions"]["8"]["graph_unweighted"]["top50_exact_pct"][2] == 98.44
