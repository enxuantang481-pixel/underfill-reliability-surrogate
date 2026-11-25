from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent


def main() -> None:
    cycle_data = json.loads(
        (ROOT / "anand_cycle_comparison.json").read_text(encoding="utf-8")
    )["results"]
    mesh_data = json.loads(
        (ROOT / "anand_mesh_sensitivity.json").read_text(encoding="utf-8")
    )["results"]

    cycles = ["Cycle 1", "Cycle 2"]
    reference = [
        cycle_data["reference_A"]["cycle_1_delta_evpe"],
        cycle_data["reference_A"]["cycle_2_delta_evpe"],
    ]
    design = [
        cycle_data["nominal_B"]["cycle_1_delta_evpe"],
        cycle_data["nominal_B"]["cycle_2_delta_evpe"],
    ]
    factors = [
        cycle_data["cycle_1_delta_evpe_comparison"]["inverse_damage_factor"],
        cycle_data["cycle_2_delta_evpe_comparison"]["inverse_damage_factor"],
    ]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 5.6),
        gridspec_kw={"width_ratios": [1.55, 1]},
    )
    figure.patch.set_facecolor("white")

    positions = np.arange(len(cycles))
    width = 0.34
    axes[0].bar(
        positions - width / 2,
        reference,
        width,
        color="#1f4e79",
        label="Nominal Epoxy A",
    )
    axes[0].bar(
        positions + width / 2,
        design,
        width,
        color="#1f9e89",
        label="Nominal Epoxy B",
    )
    axes[0].set_xticks(positions, cycles)
    axes[0].set_ylabel("Cycle-wise corner-joint ΔEVPE")
    axes[0].set_title("Nominal Epoxy B lowers nonlinear solder damage")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False, loc="upper right")
    for index, factor in enumerate(factors):
        axes[0].text(
            index,
            max(reference[index], design[index]) * 1.04,
            f"{factor:.2f}× inverse-damage factor",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#163a5f",
            weight="bold",
        )
    axes[0].set_ylim(0, max(reference) * 1.22)

    mesh_levels = ["Coarse", "Medium"]
    mesh_factors = [
        mesh_data["coarse"]["comparison"]["inverse_damage_factor"],
        mesh_data["medium"]["comparison"]["inverse_damage_factor"],
    ]
    axes[1].plot(
        mesh_levels,
        mesh_factors,
        marker="o",
        linewidth=2.5,
        markersize=8,
        color="#e07a1f",
    )
    axes[1].axhline(2.0, color="#b3261e", linestyle="--", linewidth=1.5)
    axes[1].text(0.5, 2.04, "2× target", ha="center", color="#b3261e")
    axes[1].set_ylim(1.8, 3.5)
    axes[1].set_ylabel("Inverse ΔEVPE factor (A/B)")
    axes[1].set_title("The >2× result is mesh-stable")
    axes[1].grid(axis="y", alpha=0.22)
    for index, factor in enumerate(mesh_factors):
        axes[1].text(
            index,
            factor + 0.05,
            f"{factor:.3f}×",
            ha="center",
            va="bottom",
            weight="bold",
            color="#8a4a12",
        )

    figure.suptitle(
        "Fixed-geometry COMSOL Anand validation",
        fontsize=16,
        weight="bold",
        color="#14213d",
    )
    figure.text(
        0.5,
        0.01,
        "Metric: cycle-wise increment of volume-averaged solid.evpe in the critical corner joint; not calibrated fatigue life.",
        ha="center",
        fontsize=9,
        color="#4d5969",
    )
    figure.tight_layout(rect=[0, 0.05, 1, 0.93])
    figure.savefig(ROOT / "anand_validation.png", dpi=220, bbox_inches="tight")
    figure.savefig(ROOT / "anand_validation.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
