from __future__ import annotations

import json
from pathlib import Path

import mph

from comsol_corner_model import PORT, _evaluate_anand_metric


CASES = {
    "reference_A": Path(
        r"E:\COMSEL\official_compliant_redesign_rms\reference_A\anand_c1\coarse"
        r"\official_corner_reference_A_anand_coarse_dt15_batchsolved.mph"
    ),
    "nominal_B": Path(
        r"E:\COMSEL\official_compliant_redesign_rms\nominal_B\anand_c1\coarse"
        r"\official_corner_nominal_B_anand_coarse_dt15_batchsolved.mph"
    ),
    "optimized_B": Path(
        r"E:\COMSEL\official_compliant_redesign_rms\al_006_01_B\anand_c1\coarse"
        r"\official_corner_al_006_01_B_anand_coarse_dt15_batchsolved.mph"
    ),
}

OUTPUT = Path(__file__).with_name("anand_comparison.json")


def main() -> None:
    client = mph.Client(port=PORT, host="localhost")
    results: dict[str, dict] = {}
    try:
        for case_id, model_path in CASES.items():
            model = client.load(str(model_path))
            try:
                times = [float(value) for value in model.java.sol("sol1").getPVals()]
                if not times or abs(times[-1] - 3810.0) > 1e-6:
                    raise RuntimeError(
                        f"{case_id} is incomplete: final saved time is {times[-1] if times else None} s"
                    )
                expression, damage_metric, values = _evaluate_anand_metric(
                    model,
                    times,
                    case_id,
                    cycles=1,
                )
                results[case_id] = {
                    "field_expression": expression,
                    "delta_evpe": damage_metric,
                    "field_times_s": times,
                    "field_series": values,
                    "model": str(model_path),
                }
            finally:
                client.remove(model)
    finally:
        client.disconnect()

    reference = results["reference_A"]["delta_evpe"]
    for case_id, result in results.items():
        ratio = result["delta_evpe"] / reference
        result["damage_ratio_to_reference_A"] = ratio
        result["evpe_reduction_percent"] = 100.0 * (1.0 - ratio)
        result["inverse_damage_factor"] = 1.0 / ratio

    payload = {
        "metric": "one-cycle increment of corner-ball volume-average solid.evpe",
        "comparison_boundary": (
            "Fixed 5x5 local geometry derived from official dimensions, full local underfill, "
            "identical mesh and BLR-TCC-1 history; "
            "Anand viscoplasticity restricted to the 2x2 critical corner joints."
        ),
        "claim_boundary": (
            "inverse_damage_factor is an EVPE reduction factor, not a calibrated fatigue-life factor"
        ),
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
