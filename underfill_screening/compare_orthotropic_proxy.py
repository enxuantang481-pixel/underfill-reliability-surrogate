from __future__ import annotations

import json
from pathlib import Path

import mph

from comsol_corner_model import PORT, _flatten_real


MODELS = {
    "reference_A": Path(
        r"E:\COMSEL\official_compliant_redesign_ortho\reference_A\proxy\coarse"
        r"\reference_A_proxy_coarse_ortho_solved.mph"
    ),
    "nominal_B": Path(
        r"E:\COMSEL\official_compliant_redesign_ortho\nominal_B\proxy\coarse"
        r"\nominal_B_proxy_coarse_ortho_solved.mph"
    ),
}

OUTPUT = Path(__file__).with_name("orthotropic_proxy_comparison.json")


def evaluate(model, selection_name: str, tag: str) -> list[float]:
    numerical = model.java.result().numerical().create(tag, "AvVolume")
    numerical.selection().named(selection_name)
    numerical.set("expr", "solid.mises")
    numerical.set("data", "dset1")
    return _flatten_real(numerical.getReal())


def main() -> None:
    client = mph.Client(port=PORT, host="localhost")
    results: dict[str, dict] = {}
    try:
        for case_id, model_path in MODELS.items():
            model = client.load(str(model_path))
            try:
                joint_metrics = []
                for ix in range(5):
                    for iy in range(5):
                        values = evaluate(
                            model,
                            f"sel_ball_{ix}_{iy}",
                            f"av_ortho_proxy_{case_id}_{ix}_{iy}",
                        )
                        joint_metrics.append(
                            {
                                "joint": f"({ix},{iy})",
                                "max_mises_pa": max(values),
                            }
                        )
                ranked = sorted(
                    joint_metrics,
                    key=lambda item: item["max_mises_pa"],
                    reverse=True,
                )
                corner = next(item for item in ranked if item["joint"] == "(0,0)")
                results[case_id] = {
                    "corner_metric_pa": corner["max_mises_pa"],
                    "corner_rank": next(
                        index
                        for index, item in enumerate(ranked, start=1)
                        if item["joint"] == "(0,0)"
                    ),
                    "ranked_joints": ranked,
                    "model": str(model_path),
                }
            finally:
                client.remove(model)
    finally:
        client.disconnect()

    ratio = (
        results["nominal_B"]["corner_metric_pa"]
        / results["reference_A"]["corner_metric_pa"]
    )
    payload = {
        "metric": "max of volume-average corner-joint solid.mises at 25/-40/125 degC",
        "results": results,
        "comparison": {
            "damage_ratio_B_to_A": ratio,
            "reduction_percent": 100.0 * (1.0 - ratio),
            "inverse_proxy_factor": 1.0 / ratio,
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
