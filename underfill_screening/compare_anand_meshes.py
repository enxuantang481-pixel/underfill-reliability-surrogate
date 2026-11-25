from __future__ import annotations

import json
from pathlib import Path

import mph

from comsol_corner_model import PORT, _evaluate_anand_metric


MODELS = {
    "coarse": {
        "reference_A": Path(
            r"E:\COMSEL\official_compliant_redesign_rms\reference_A\anand_c1\coarse"
            r"\official_corner_reference_A_anand_coarse_dt15_batchsolved.mph"
        ),
        "nominal_B": Path(
            r"E:\COMSEL\official_compliant_redesign_rms\nominal_B\anand_c1\coarse"
            r"\official_corner_nominal_B_anand_coarse_dt15_batchsolved.mph"
        ),
    },
    "medium": {
        "reference_A": Path(
            r"E:\COMSEL\official_compliant_redesign_rms\reference_A\anand_c1\medium"
            r"\official_corner_reference_A_anand_medium_dt15_batchsolved.mph"
        ),
        "nominal_B": Path(
            r"E:\COMSEL\official_compliant_redesign_rms\nominal_B\anand_c1\medium"
            r"\official_corner_nominal_B_anand_medium_dt15_batchsolved.mph"
        ),
    },
}

OUTPUT = Path(__file__).with_name("anand_mesh_sensitivity.json")


def main() -> None:
    client = mph.Client(port=PORT, host="localhost")
    results: dict[str, dict[str, dict]] = {}
    try:
        for mesh_level, cases in MODELS.items():
            results[mesh_level] = {}
            for case_id, model_path in cases.items():
                model = client.load(str(model_path))
                try:
                    times = [float(value) for value in model.java.sol("sol1").getPVals()]
                    if not times or abs(times[-1] - 3810.0) > 1e-6:
                        raise RuntimeError(
                            f"{mesh_level}/{case_id} incomplete at "
                            f"{times[-1] if times else None} s"
                        )
                    expression, delta_evpe, _ = _evaluate_anand_metric(
                        model,
                        times,
                        f"{mesh_level}_{case_id}",
                        cycles=1,
                    )
                    results[mesh_level][case_id] = {
                        "field_expression": expression,
                        "delta_evpe": delta_evpe,
                        "model": str(model_path),
                    }
                finally:
                    client.remove(model)
    finally:
        client.disconnect()

    for mesh_level, cases in results.items():
        reference = cases["reference_A"]["delta_evpe"]
        design = cases["nominal_B"]["delta_evpe"]
        ratio = design / reference
        cases["comparison"] = {
            "damage_ratio_B_to_A": ratio,
            "evpe_reduction_percent": 100.0 * (1.0 - ratio),
            "inverse_damage_factor": 1.0 / ratio,
        }

    coarse = results["coarse"]
    medium = results["medium"]
    payload = {
        "metric": "one-cycle increment of corner-ball volume-average solid.evpe",
        "results": results,
        "mesh_change_percent": {
            "reference_A": 100.0
            * (medium["reference_A"]["delta_evpe"] / coarse["reference_A"]["delta_evpe"] - 1.0),
            "nominal_B": 100.0
            * (medium["nominal_B"]["delta_evpe"] / coarse["nominal_B"]["delta_evpe"] - 1.0),
            "inverse_damage_factor": 100.0
            * (
                medium["comparison"]["inverse_damage_factor"]
                / coarse["comparison"]["inverse_damage_factor"]
                - 1.0
            ),
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
