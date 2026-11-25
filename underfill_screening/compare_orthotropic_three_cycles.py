from __future__ import annotations

import json
from pathlib import Path

import mph

from comsol_corner_model import PORT, _evaluate_anand_metric


MODELS = {
    "reference_A": Path(
        r"E:\COMSEL\official_compliant_redesign_ortho\reference_A\anand_c3\coarse"
        r"\reference_A_anand_c3_coarse_ortho_dt15_solved.mph"
    ),
    "nominal_B": Path(
        r"E:\COMSEL\official_compliant_redesign_ortho\nominal_B\anand_c3\coarse"
        r"\nominal_B_anand_c3_coarse_ortho_dt15_solved.mph"
    ),
}

OUTPUT = Path(__file__).with_name("orthotropic_three_cycle_comparison.json")
CYCLE_BOUNDARIES_S = [390.0, 3810.0, 7230.0, 10650.0]


def nearest_index(times: list[float], target: float) -> int:
    return min(range(len(times)), key=lambda index: abs(times[index] - target))


def main() -> None:
    client = mph.Client(port=PORT, host="localhost")
    results: dict[str, dict] = {}
    try:
        for case_id, model_path in MODELS.items():
            model = client.load(str(model_path))
            try:
                times = [float(value) for value in model.java.sol("sol1").getPVals()]
                expected_end = CYCLE_BOUNDARIES_S[-1]
                if not times or abs(times[-1] - expected_end) > 1e-6:
                    raise RuntimeError(
                        f"{case_id} is incomplete: final saved time is "
                        f"{times[-1] if times else None} s"
                    )

                expression, _, values = _evaluate_anand_metric(
                    model,
                    times,
                    f"orthotropic_c3_{case_id}",
                    cycles=3,
                )
                indices = {
                    target: nearest_index(times, target)
                    for target in CYCLE_BOUNDARIES_S
                }
                deltas = [
                    values[indices[end]] - values[indices[start]]
                    for start, end in zip(
                        CYCLE_BOUNDARIES_S[:-1], CYCLE_BOUNDARIES_S[1:]
                    )
                ]
                results[case_id] = {
                    "field_expression": expression,
                    "cycle_delta_evpe": deltas,
                    "successive_cycle_ratio": [
                        deltas[index] / deltas[index - 1]
                        for index in range(1, len(deltas))
                    ],
                    "final_time_s": times[-1],
                    "saved_time_count": len(times),
                    "model": str(model_path),
                }
            finally:
                client.remove(model)
    finally:
        client.disconnect()

    comparisons = []
    for cycle_index in range(3):
        reference = results["reference_A"]["cycle_delta_evpe"][cycle_index]
        design = results["nominal_B"]["cycle_delta_evpe"][cycle_index]
        ratio = design / reference
        comparisons.append(
            {
                "cycle": cycle_index + 1,
                "damage_ratio_B_to_A": ratio,
                "evpe_reduction_percent": 100.0 * (1.0 - ratio),
                "inverse_damage_factor": 1.0 / ratio,
            }
        )

    payload = {
        "metric": "cycle-wise increment of corner-ball volume-average solid.evpe",
        "interpretation": (
            "Inverse damage factor is a relative strain-accumulation indicator, "
            "not a calibrated fatigue-life multiplier."
        ),
        "fr4_model": (
            "Competition orthotropic E, Poisson ratios and CTE; "
            "shear moduli derived from E and nu."
        ),
        "results": results,
        "comparisons": comparisons,
        "minimum_inverse_damage_factor": min(
            item["inverse_damage_factor"] for item in comparisons
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
