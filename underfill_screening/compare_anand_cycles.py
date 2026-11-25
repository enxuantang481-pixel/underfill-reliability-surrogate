from __future__ import annotations

import json
from pathlib import Path

import mph

from comsol_corner_model import PORT, _evaluate_anand_metric


MODELS = {
    "reference_A": Path(
        r"E:\COMSEL\official_compliant_redesign_rms\reference_A\anand_c2\coarse"
        r"\official_corner_reference_A_anand_coarse_dt15_batchsolved.mph"
    ),
    "nominal_B": Path(
        r"E:\COMSEL\official_compliant_redesign_rms\nominal_B\anand_c2\coarse"
        r"\official_corner_nominal_B_anand_coarse_dt15_batchsolved.mph"
    ),
}

OUTPUT = Path(__file__).with_name("anand_cycle_comparison.json")


def main() -> None:
    client = mph.Client(port=PORT, host="localhost")
    results: dict[str, dict] = {}
    try:
        for case_id, model_path in MODELS.items():
            model = client.load(str(model_path))
            try:
                times = [float(value) for value in model.java.sol("sol1").getPVals()]
                if not times or abs(times[-1] - 7230.0) > 1e-6:
                    raise RuntimeError(
                        f"{case_id} is incomplete: final saved time is "
                        f"{times[-1] if times else None} s"
                    )
                expression, cycle_2_delta, values = _evaluate_anand_metric(
                    model,
                    times,
                    case_id,
                    cycles=2,
                )
                first_cycle_start = min(
                    range(len(times)), key=lambda index: abs(times[index] - 390.0)
                )
                first_cycle_end = min(
                    range(len(times)), key=lambda index: abs(times[index] - 3810.0)
                )
                cycle_1_delta = values[first_cycle_end] - values[first_cycle_start]
                results[case_id] = {
                    "field_expression": expression,
                    "cycle_1_delta_evpe": cycle_1_delta,
                    "cycle_2_delta_evpe": cycle_2_delta,
                    "cycle_2_to_cycle_1": cycle_2_delta / cycle_1_delta,
                    "model": str(model_path),
                }
            finally:
                client.remove(model)
    finally:
        client.disconnect()

    for cycle_key in ["cycle_1_delta_evpe", "cycle_2_delta_evpe"]:
        reference = results["reference_A"][cycle_key]
        design = results["nominal_B"][cycle_key]
        ratio = design / reference
        results[f"{cycle_key}_comparison"] = {
            "damage_ratio_B_to_A": ratio,
            "evpe_reduction_percent": 100.0 * (1.0 - ratio),
            "inverse_damage_factor": 1.0 / ratio,
        }

    payload = {
        "metric": "cycle-wise increment of corner-ball volume-average solid.evpe",
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
