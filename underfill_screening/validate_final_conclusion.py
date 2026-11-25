from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parent
OUTPUT = ROOT / "final_compliance_verdict.json"


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main() -> None:
    cycles = load("orthotropic_three_cycle_comparison.json")
    full_anand = load("full_anand_25_joint_check.json")
    mesh = load("orthotropic_mesh_sensitivity.json")
    model_audit = load("final_model_audit.json")

    minimum_three_cycle_factor = min(
        item["inverse_damage_factor"] for item in cycles["comparisons"]
    )
    full_anand_factor = full_anand["comparison"]["inverse_damage_factor"]
    mesh_factor_change_percent = abs(
        mesh["mesh_change_percent"]["inverse_damage_factor"]
    )

    checks = {
        "minimum_three_cycle_factor_exceeds_2x": (
            minimum_three_cycle_factor >= 2.0
        ),
        "full_25_joint_anand_factor_exceeds_2x": full_anand_factor >= 2.0,
        "anand_active_in_all_25_joint_check": full_anand["checks"][
            "anand_active_in_all_25_joints"
        ],
        "outer_corner_is_critical": full_anand["checks"][
            "outer_corner_ranked_first_in_both"
        ],
        "fixed_parameters_identical_A_B": model_audit["checks"][
            "fixed_parameters_identical_A_B"
        ],
        "fixed_domain_selections_identical_A_B": model_audit["checks"][
            "fixed_domain_selections_identical_A_B"
        ],
        "three_cycle_models_complete": model_audit["checks"][
            "both_complete_to_10650_s"
        ],
        "mesh_factor_change_below_1_percent": mesh_factor_change_percent < 1.0,
    }

    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "recommended_design": "competition nominal Epoxy B",
        "reference_design": "competition nominal Epoxy A",
        "claim_metric": (
            "inverse ratio of cycle-wise critical-joint volume-average "
            "equivalent viscoplastic strain increment"
        ),
        "minimum_three_cycle_inverse_damage_factor": (
            minimum_three_cycle_factor
        ),
        "full_25_joint_cycle_1_inverse_damage_factor": full_anand_factor,
        "full_25_joint_cycle_1_evpe_reduction_percent": full_anand[
            "comparison"
        ]["evpe_reduction_percent"],
        "mesh_factor_change_percent": mesh_factor_change_percent,
        "checks": checks,
        "claim_boundary": (
            "Relative EVPE damage-indicator improvement, not calibrated "
            "cycles-to-failure."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
