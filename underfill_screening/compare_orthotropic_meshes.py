from __future__ import annotations

import json
from pathlib import Path

import mph

from comsol_corner_model import PORT, _evaluate_anand_metric


MODELS = {
    "coarse": {
        "reference_A": Path(
            r"E:\COMSEL\official_compliant_redesign_ortho\reference_A\anand_c2\coarse"
            r"\reference_A_anand_c2_coarse_ortho_dt15_solved.mph"
        ),
        "nominal_B": Path(
            r"E:\COMSEL\official_compliant_redesign_ortho\nominal_B\anand_c2\coarse"
            r"\nominal_B_anand_c2_coarse_ortho_dt15_solved.mph"
        ),
    },
    "medium": {
        "reference_A": Path(
            r"E:\COMSEL\official_compliant_redesign_ortho\reference_A\anand_c1\medium"
            r"\reference_A_anand_c1_medium_ortho_dt15_ooc_solved.mph"
        ),
        "nominal_B": Path(
            r"E:\COMSEL\official_compliant_redesign_ortho\nominal_B\anand_c1\medium"
            r"\nominal_B_anand_c1_medium_ortho_dt15_ooc_solved.mph"
        ),
    },
}

OUTPUT = Path(__file__).with_name("orthotropic_mesh_sensitivity.json")


def main() -> None:
    client = mph.Client(port=PORT, host="localhost")
    results: dict[str, dict] = {}
    try:
        for mesh_level, cases in MODELS.items():
            results[mesh_level] = {}
            for case_id, model_path in cases.items():
                model = client.load(str(model_path))
                try:
                    times = [float(value) for value in model.java.sol("sol1").getPVals()]
                    expected_end = 3810.0 if mesh_level == "medium" else 7230.0
                    if not times or abs(times[-1] - expected_end) > 1e-6:
                        raise RuntimeError(
                            f"{mesh_level}/{case_id} incomplete at "
                            f"{times[-1] if times else None} s"
                        )
                    cycles = 1 if mesh_level == "medium" else 2
                    expression, delta_evpe, _ = _evaluate_anand_metric(
                        model,
                        times,
                        f"ortho_{mesh_level}_{case_id}",
                        cycles=cycles,
                    )
                    if mesh_level == "coarse":
                        values = _evaluate_anand_metric(
                            model,
                            times,
                            f"ortho_{mesh_level}_{case_id}_cycle1",
                            cycles=1,
                        )[2]
                        start = min(
                            range(len(times)), key=lambda index: abs(times[index] - 390.0)
                        )
                        end = min(
                            range(len(times)), key=lambda index: abs(times[index] - 3810.0)
                        )
                        delta_evpe = values[end] - values[start]
                    results[mesh_level][case_id] = {
                        "field_expression": expression,
                        "cycle_1_delta_evpe": delta_evpe,
                        "model": str(model_path),
                    }
                finally:
                    client.remove(model)
    finally:
        client.disconnect()

    for mesh_level, cases in results.items():
        reference = cases["reference_A"]["cycle_1_delta_evpe"]
        design = cases["nominal_B"]["cycle_1_delta_evpe"]
        ratio = design / reference
        cases["comparison"] = {
            "damage_ratio_B_to_A": ratio,
            "evpe_reduction_percent": 100.0 * (1.0 - ratio),
            "inverse_damage_factor": 1.0 / ratio,
        }

    payload = {
        "metric": "cycle-1 increment of corner-ball volume-average solid.evpe",
        "results": results,
        "mesh_change_percent": {
            "reference_A": 100.0
            * (
                results["medium"]["reference_A"]["cycle_1_delta_evpe"]
                / results["coarse"]["reference_A"]["cycle_1_delta_evpe"]
                - 1.0
            ),
            "nominal_B": 100.0
            * (
                results["medium"]["nominal_B"]["cycle_1_delta_evpe"]
                / results["coarse"]["nominal_B"]["cycle_1_delta_evpe"]
                - 1.0
            ),
            "inverse_damage_factor": 100.0
            * (
                results["medium"]["comparison"]["inverse_damage_factor"]
                / results["coarse"]["comparison"]["inverse_damage_factor"]
                - 1.0
            ),
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
