from __future__ import annotations

import csv
import json
from pathlib import Path

from active_learning import propose_cases, read_solved_results
from comsol_corner_model import OUTPUT_ROOT


HERE = Path(__file__).resolve().parent


def main() -> None:
    coarse_rows = read_solved_results(OUTPUT_ROOT, mode="proxy", mesh_level="coarse")
    reference = next(row for row in coarse_rows if row["case_id"] == "reference_A")
    reference_damage = reference["damage_metric"]
    table = []
    for row in sorted(coarse_rows, key=lambda item: item["damage_metric"]):
        ratio = row["damage_metric"] / reference_damage
        nominal = {
            "A": {"elastic_gpa": 10.0, "cte_ppm_k": 24.0, "tg_c": 110.0},
            "B": {"elastic_gpa": 7.0, "cte_ppm_k": 32.0, "tg_c": 145.0},
        }[row["variant"]]
        table.append(
            {
                "case_id": row["case_id"],
                "variant": row["variant"],
                "elastic_scale": row["elastic_scale"],
                "cte_scale": row["cte_scale"],
                "tg_shift_c": row["tg_shift_c"],
                "elastic_gpa": nominal["elastic_gpa"] * row["elastic_scale"],
                "cte_below_tg_ppm_k": nominal["cte_ppm_k"] * row["cte_scale"],
                "tg_c": nominal["tg_c"] + row["tg_shift_c"],
                "damage_mpa": row["damage_metric"] / 1e6,
                "damage_ratio_vs_A": ratio,
                "reduction_vs_A_percent": 100.0 * (1.0 - ratio),
                "source": row["source"],
            }
        )

    best = table[0]
    less_aggressive = min(
        (
            row
            for row in table
            if row["variant"] == "B"
            and row["elastic_scale"] >= 1.0
            and row["tg_shift_c"] >= 0.0
        ),
        key=lambda item: item["damage_ratio_vs_A"],
    )

    mesh_validation = {}
    for case_id in ["reference_A", best["case_id"]]:
        coarse = json.loads(
            (OUTPUT_ROOT / case_id / "proxy" / "coarse" / "result.json").read_text(encoding="utf-8")
        )
        medium = json.loads(
            (OUTPUT_ROOT / case_id / "proxy" / "medium" / "result.json").read_text(encoding="utf-8")
        )
        mesh_validation[case_id] = {
            "coarse_damage_mpa": coarse["damage_metric"] / 1e6,
            "medium_damage_mpa": medium["damage_metric"] / 1e6,
            "relative_change_percent": 100.0
            * (medium["damage_metric"] - coarse["damage_metric"])
            / coarse["damage_metric"],
        }

    reference_medium = mesh_validation["reference_A"]["medium_damage_mpa"]
    best_medium = mesh_validation[best["case_id"]]["medium_damage_mpa"]
    mesh_validation["medium_ratio_best_vs_A"] = best_medium / reference_medium

    screening_life_exponent = 1.96
    uncalibrated_damage_threshold = 2.0 ** (-1.0 / screening_life_exponent)
    next_candidates = propose_cases(coarse_rows, batch_size=4, seed=2028)
    report = {
        "geometry_status": "fixed to dimensions measured from the competition STEP",
        "boundary_condition": "Rigid Motion Suppression; free thermo-mechanical warpage",
        "damage_metric": "max over 25/-40/125 C states of the official corner-ball volume-average von Mises stress",
        "reference_case": "official nominal Epoxy A",
        "coarse_proxy_results": table,
        "mathematical_best_screened_candidate": best,
        "less_aggressive_screened_candidate": less_aggressive,
        "mesh_validation": mesh_validation,
        "uncalibrated_2x_screening": {
            "exponent": screening_life_exponent,
            "damage_ratio_threshold": uncalibrated_damage_threshold,
            "warning": "The exponent is not a calibrated SAC305 life model for this assembly. It is included only as a screening threshold, not a qualified life claim.",
        },
        "anand_status": {
            "full_25_ball_fixed_bottom": "failed local-variable convergence",
            "full_25_ball_rigid_motion": "computationally impractical for batch screening",
            "corner_2x2_batch": "advanced to 79 percent and was saved; stopped only because TU Delft license connectivity was lost",
            "partial_model": str(
                OUTPUT_ROOT
                / "reference_A"
                / "anand_c1"
                / "coarse"
                / "official_corner_reference_A_anand_coarse_batchsolved.mph"
            ),
        },
        "next_active_learning_candidates": next_candidates,
    }

    json_path = HERE / "screening_summary.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = HERE / "screening_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

