from __future__ import annotations

import argparse
from pathlib import Path

import jpype
import mph

from comsol_corner_model import PORT


def jstrings(values: list[str]) -> jpype.JArray:
    return jpype.JArray(jpype.JString)(values)


def set_values(feature, **values) -> None:
    for key, value in values.items():
        if isinstance(value, list):
            value = jstrings(value)
        feature.set(key, value)


def add_parameter(model, name: str, expression: str, description: str) -> None:
    model.java.param().set(name, expression, description)


def create_orthotropic_feature(
    solid,
    tag: str,
    selection: str,
    prefix: str,
) -> None:
    feature = solid.create(tag, "LinearElasticModel", 3)
    feature.selection().named(selection)
    set_values(
        feature,
        SolidModel="Orthotropic",
        OrthotropicOption="OrthotropicStd",
        Evector_mat="userdef",
        Evector=[f"E_{prefix}_xy", f"E_{prefix}_xy", f"E_{prefix}_z"],
        nuvector_mat="userdef",
        nuvector=[f"nu_{prefix}_xy", f"nu_{prefix}_xz", f"nu_{prefix}_xz"],
        Gvector_mat="userdef",
        Gvector=[f"G_{prefix}_xy", f"G_{prefix}_xz", f"G_{prefix}_xz"],
    )
    expansion = feature.create(f"te_{tag}", "ThermalExpansion", 3)
    set_values(
        expansion,
        alpha_mat="userdef",
        alpha=[f"alpha_{prefix}_xy", f"alpha_{prefix}_xy", f"alpha_{prefix}_z"],
        minput_temperature_src="userdef",
        minput_temperature="Tblr(t)",
        minput_strainreferencetemperature_src="userdef",
        minput_strainreferencetemperature="Tref",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace isotropic FR4 physics with competition orthotropic inputs."
    )
    parser.add_argument("input_model", type=Path)
    parser.add_argument("output_model", type=Path)
    args = parser.parse_args()

    client = mph.Client(port=PORT, host="localhost")
    model = None
    try:
        model = client.load(str(args.input_model))
        java = model.java
        component = java.component("comp1")

        parameters = {
            "E_fr4a_xy": ("20[GPa]", "competition FR4-A in-plane modulus"),
            "E_fr4a_z": ("15[GPa]", "competition FR4-A through-thickness modulus"),
            "nu_fr4a_xy": ("0.10", "competition FR4-A in-plane Poisson ratio"),
            "nu_fr4a_xz": ("0.14", "competition FR4-A transverse Poisson ratio"),
            "G_fr4a_xy": ("E_fr4a_xy/(2*(1+nu_fr4a_xy))", "derived in-plane shear modulus"),
            "G_fr4a_xz": ("E_fr4a_z/(2*(1+nu_fr4a_xz))", "derived transverse shear modulus"),
            "alpha_fr4a_xy": ("16e-6[1/K]", "competition FR4-A in-plane CTE"),
            "alpha_fr4a_z": ("82e-6[1/K]", "competition FR4-A through-thickness CTE"),
            "E_fr4b_xy": ("18[GPa]", "competition FR4-B in-plane modulus"),
            "E_fr4b_z": ("14[GPa]", "competition FR4-B through-thickness modulus"),
            "nu_fr4b_xy": ("0.10", "competition FR4-B in-plane Poisson ratio"),
            "nu_fr4b_xz": ("0.14", "competition FR4-B transverse Poisson ratio"),
            "G_fr4b_xy": ("E_fr4b_xy/(2*(1+nu_fr4b_xy))", "derived in-plane shear modulus"),
            "G_fr4b_xz": ("E_fr4b_z/(2*(1+nu_fr4b_xz))", "derived transverse shear modulus"),
            "alpha_fr4b_xy": ("11e-6[1/K]", "competition FR4-B in-plane CTE"),
            "alpha_fr4b_z": ("70e-6[1/K]", "competition FR4-B through-thickness CTE"),
        }
        for name, (expression, description) in parameters.items():
            add_parameter(model, name, expression, description)

        selections = component.selection()
        if "sel_non_fr4" not in [str(tag) for tag in selections.tags()]:
            non_fr4 = selections.create("sel_non_fr4", "Union")
            non_fr4.set(
                "input",
                jstrings(["sel_pads", "sel_solder", "sel_underfill"]),
            )

        solid = component.physics("solid")
        solid.feature("lemm1")
        create_orthotropic_feature(solid, "lemm_pcb", "sel_pcb", "fr4a")
        create_orthotropic_feature(
            solid,
            "lemm_substrate",
            "sel_substrate",
            "fr4b",
        )

        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        java.save(str(args.output_model))
        print(args.output_model)
    finally:
        if model is not None:
            client.remove(model)
        client.disconnect()


if __name__ == "__main__":
    main()
