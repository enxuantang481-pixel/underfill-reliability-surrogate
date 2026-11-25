from __future__ import annotations

import json
import math
from pathlib import Path

import mph

from comsol_corner_model import PORT, _flatten_real


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

OUTPUT = Path(__file__).with_name("final_model_audit.json")
CYCLE_BOUNDARIES_S = [390.0, 3810.0, 7230.0, 10650.0]
JOINTS = [(0, 0), (0, 1), (1, 0), (1, 1)]

FIXED_PARAMETER_NAMES = [
    "patch_min",
    "patch_max",
    "patch_size",
    "pitch",
    "solder_r",
    "solder_h",
    "pad_r",
    "pad_t",
    "gap_h",
    "pcb_t",
    "sub_t",
    "corner_DNP",
    "rho_sac305",
    "E_sac305",
    "nu_sac305",
    "alpha_sac305",
    "anand_s0",
    "anand_Q",
    "anand_A",
    "anand_xi",
    "anand_m",
    "anand_h0",
    "anand_shat",
    "anand_n",
    "anand_a",
    "E_fr4a_xy",
    "E_fr4a_z",
    "nu_fr4a_xy",
    "nu_fr4a_xz",
    "G_fr4a_xy",
    "G_fr4a_xz",
    "alpha_fr4a_xy",
    "alpha_fr4a_z",
    "E_fr4b_xy",
    "E_fr4b_z",
    "nu_fr4b_xy",
    "nu_fr4b_xz",
    "G_fr4b_xy",
    "G_fr4b_xz",
    "alpha_fr4b_xy",
    "alpha_fr4b_z",
]

UNDERFILL_PARAMETER_NAMES = [
    "rho_uf",
    "E_uf",
    "nu_uf",
    "alpha_uf_below",
    "alpha_jump",
    "Tg_uf",
    "T_smooth",
    "alpha_uf",
]


def parameter_expressions(model, names: list[str]) -> dict[str, str]:
    parameters = model.java.param()
    return {name: str(parameters.get(name)) for name in names}


def selection_entities(selection) -> list[int]:
    return sorted(int(entity) for entity in selection.entities())


def nearest_index(times: list[float], target: float) -> int:
    return min(range(len(times)), key=lambda index: abs(times[index] - target))


def evaluate_joint(model, selection_name: str, tag: str) -> list[float]:
    numerical = model.java.result().numerical().create(tag, "AvVolume")
    numerical.selection().named(selection_name)
    numerical.set("expr", "solid.evpe")
    numerical.set("data", "dset1")
    return _flatten_real(numerical.getReal())


def string_property(feature, name: str) -> str:
    try:
        return str(feature.getString(name))
    except Exception:
        return "<unavailable>"


def audit_case(model, case_id: str, model_path: Path) -> dict:
    java = model.java
    component = java.component("comp1")
    solid = component.physics("solid")
    default_elastic = solid.feature("lemm1")
    pcb_elastic = solid.feature("lemm_pcb")
    substrate_elastic = solid.feature("lemm_substrate")
    viscoplasticity = default_elastic.feature("vp1")
    time_solver = java.sol("sol1").feature("t1")
    times = [float(value) for value in java.sol("sol1").getPVals()]

    indices = {
        target: nearest_index(times, target) for target in CYCLE_BOUNDARIES_S
    }
    joints = []
    for ix, iy in JOINTS:
        values = evaluate_joint(
            model,
            f"sel_ball_{ix}_{iy}",
            f"audit_{case_id}_{ix}_{iy}",
        )
        deltas = [
            values[indices[end]] - values[indices[start]]
            for start, end in zip(
                CYCLE_BOUNDARIES_S[:-1], CYCLE_BOUNDARIES_S[1:]
            )
        ]
        joints.append(
            {
                "joint": f"({ix},{iy})",
                "dnp_mm": math.hypot(12.5 - ix, 12.5 - iy),
                "cycle_delta_evpe": deltas,
            }
        )

    return {
        "model": str(model_path),
        "final_time_s": times[-1],
        "saved_time_count": len(times),
        "fixed_parameters": parameter_expressions(model, FIXED_PARAMETER_NAMES),
        "underfill_parameters": parameter_expressions(
            model, UNDERFILL_PARAMETER_NAMES
        ),
        "domain_selections": {
            tag: selection_entities(component.selection(tag))
            for tag in [
                "sel_pcb",
                "sel_substrate",
                "sel_pads",
                "sel_solder",
                "sel_underfill",
                "sel_anand_zone",
            ]
        },
        "physics": {
            "solid_feature_tags": [
                str(tag) for tag in solid.feature().tags()
            ],
            "default_elastic_subfeatures": [
                str(tag) for tag in default_elastic.feature().tags()
            ],
            "anand": {
                "model": string_property(
                    viscoplasticity, "ViscoplasticityModel"
                ),
                "selected_domains": selection_entities(
                    viscoplasticity.selection()
                ),
                "temperature": string_property(
                    viscoplasticity, "minput_temperature"
                ),
            },
            "pcb_orthotropic": {
                "selected_domains": selection_entities(pcb_elastic.selection()),
                "solid_model": string_property(pcb_elastic, "SolidModel"),
                "Evector": string_property(pcb_elastic, "Evector"),
                "nuvector": string_property(pcb_elastic, "nuvector"),
                "Gvector": string_property(pcb_elastic, "Gvector"),
            },
            "substrate_orthotropic": {
                "selected_domains": selection_entities(
                    substrate_elastic.selection()
                ),
                "solid_model": string_property(
                    substrate_elastic, "SolidModel"
                ),
                "Evector": string_property(substrate_elastic, "Evector"),
                "nuvector": string_property(substrate_elastic, "nuvector"),
                "Gvector": string_property(substrate_elastic, "Gvector"),
            },
            "rigid_motion_suppression": "rms1" in [
                str(tag) for tag in solid.feature().tags()
            ],
        },
        "solver": {
            "relative_tolerance": string_property(time_solver, "rtol"),
            "maximum_order": string_property(time_solver, "maxorder"),
            "initial_step": string_property(time_solver, "initialstepbdf"),
            "maximum_step": string_property(time_solver, "maxstepbdf"),
            "fully_coupled_max_iterations": string_property(
                time_solver.feature("fc1"), "maxiter"
            ),
            "linear_solver": string_property(
                time_solver.feature("d1"), "linsolver"
            ),
        },
        "critical_joint_ranking": [
            item["joint"]
            for item in sorted(
                joints,
                key=lambda item: item["cycle_delta_evpe"][-1],
                reverse=True,
            )
        ],
        "joints": joints,
    }


def main() -> None:
    client = mph.Client(port=PORT, host="localhost")
    results: dict[str, dict] = {}
    try:
        for case_id, model_path in MODELS.items():
            model = client.load(str(model_path))
            try:
                results[case_id] = audit_case(model, case_id, model_path)
            finally:
                client.remove(model)
    finally:
        client.disconnect()

    fixed_parameters_identical = (
        results["reference_A"]["fixed_parameters"]
        == results["nominal_B"]["fixed_parameters"]
    )
    fixed_selections_identical = (
        results["reference_A"]["domain_selections"]
        == results["nominal_B"]["domain_selections"]
    )
    payload = {
        "checks": {
            "fixed_parameters_identical_A_B": fixed_parameters_identical,
            "fixed_domain_selections_identical_A_B": fixed_selections_identical,
            "both_complete_to_10650_s": all(
                abs(result["final_time_s"] - 10650.0) < 1e-6
                for result in results.values()
            ),
            "corner_joint_ranked_first_in_both": all(
                result["critical_joint_ranking"][0] == "(0,0)"
                for result in results.values()
            ),
        },
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
