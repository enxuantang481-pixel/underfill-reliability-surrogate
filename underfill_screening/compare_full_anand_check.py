from __future__ import annotations

import json
import math
from pathlib import Path

import mph

from comsol_corner_model import PORT, _flatten_real


MODELS = {
    "reference_A": Path(
        r"E:\COMSEL\official_compliant_redesign_ortho_full_anand\reference_A"
        r"\anand_c1\coarse\reference_A_full25_anand_c1_ortho_dt15_solved.mph"
    ),
    "nominal_B": Path(
        r"E:\COMSEL\official_compliant_redesign_ortho_full_anand\nominal_B"
        r"\anand_c1\coarse\nominal_B_full25_anand_c1_ortho_dt15_solved.mph"
    ),
}

CRITICAL_ZONE_RESULTS = {
    "reference_A": 0.017812319795827963,
    "nominal_B": 0.00526843846200486,
}

OUTPUT = Path(__file__).with_name("full_anand_25_joint_check.json")


def nearest_index(times: list[float], target: float) -> int:
    return min(range(len(times)), key=lambda index: abs(times[index] - target))


def evaluate_joint(model, selection_name: str, tag: str) -> list[float]:
    numerical = model.java.result().numerical().create(tag, "AvVolume")
    numerical.selection().named(selection_name)
    numerical.set("expr", "solid.evpe")
    numerical.set("data", "dset1")
    return _flatten_real(numerical.getReal())


def main() -> None:
    client = mph.Client(port=PORT, host="localhost")
    results: dict[str, dict] = {}
    try:
        for case_id, model_path in MODELS.items():
            model = client.load(str(model_path))
            try:
                times = [float(value) for value in model.java.sol("sol1").getPVals()]
                if not times or abs(times[-1] - 3810.0) > 1e-6:
                    raise RuntimeError(
                        f"{case_id} incomplete at "
                        f"{times[-1] if times else None} s"
                    )
                start_index = nearest_index(times, 390.0)
                end_index = nearest_index(times, 3810.0)
                joints = []
                for ix in range(5):
                    for iy in range(5):
                        values = evaluate_joint(
                            model,
                            f"sel_ball_{ix}_{iy}",
                            f"full25_{case_id}_{ix}_{iy}",
                        )
                        joints.append(
                            {
                                "joint": f"({ix},{iy})",
                                "dnp_mm": math.hypot(12.5 - ix, 12.5 - iy),
                                "cycle_1_delta_evpe": (
                                    values[end_index] - values[start_index]
                                ),
                            }
                        )
                ranked = sorted(
                    joints,
                    key=lambda item: item["cycle_1_delta_evpe"],
                    reverse=True,
                )
                corner_delta = next(
                    item["cycle_1_delta_evpe"]
                    for item in joints
                    if item["joint"] == "(0,0)"
                )
                results[case_id] = {
                    "model": str(model_path),
                    "final_time_s": times[-1],
                    "anand_selected_domain_count": len(
                        model.java.component("comp1")
                        .physics("solid")
                        .feature("lemm1")
                        .feature("vp1")
                        .selection()
                        .entities()
                    ),
                    "corner_cycle_1_delta_evpe": corner_delta,
                    "corner_change_from_2x2_anand_percent": 100.0
                    * (corner_delta / CRITICAL_ZONE_RESULTS[case_id] - 1.0),
                    "critical_joint_ranking": [
                        item["joint"] for item in ranked
                    ],
                    "joints": ranked,
                }
            finally:
                client.remove(model)
    finally:
        client.disconnect()

    reference = results["reference_A"]["corner_cycle_1_delta_evpe"]
    design = results["nominal_B"]["corner_cycle_1_delta_evpe"]
    ratio = design / reference
    payload = {
        "metric": (
            "cycle-1 increment of each solder joint's volume-average solid.evpe; "
            "Anand active in all 25 joints"
        ),
        "checks": {
            "anand_active_in_all_25_joints": all(
                result["anand_selected_domain_count"] == 25
                for result in results.values()
            ),
            "outer_corner_ranked_first_in_both": all(
                result["critical_joint_ranking"][0] == "(0,0)"
                for result in results.values()
            ),
        },
        "comparison": {
            "damage_ratio_B_to_A": ratio,
            "evpe_reduction_percent": 100.0 * (1.0 - ratio),
            "inverse_damage_factor": 1.0 / ratio,
        },
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
