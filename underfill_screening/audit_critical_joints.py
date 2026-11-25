from __future__ import annotations

import json
import math
from pathlib import Path

import mph

from comsol_corner_model import PORT, _flatten_real


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

JOINTS = [(0, 0), (0, 1), (1, 0), (1, 1)]
OUTPUT = Path(__file__).with_name("critical_joint_audit.json")


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
                indices = {
                    target: min(
                        range(len(times)), key=lambda index: abs(times[index] - target)
                    )
                    for target in [390.0, 3810.0, 7230.0]
                }
                joints: list[dict] = []
                for ix, iy in JOINTS:
                    values = evaluate_joint(
                        model,
                        f"sel_ball_{ix}_{iy}",
                        f"av_{case_id}_{ix}_{iy}",
                    )
                    dnp_mm = math.hypot(12.5 - ix, 12.5 - iy)
                    joints.append(
                        {
                            "joint": f"({ix},{iy})",
                            "dnp_mm": dnp_mm,
                            "cycle_1_delta_evpe": values[indices[3810.0]]
                            - values[indices[390.0]],
                            "cycle_2_delta_evpe": values[indices[7230.0]]
                            - values[indices[3810.0]],
                        }
                    )
                results[case_id] = {
                    "model": str(model_path),
                    "joints": joints,
                    "cycle_1_rank": [
                        item["joint"]
                        for item in sorted(
                            joints,
                            key=lambda item: item["cycle_1_delta_evpe"],
                            reverse=True,
                        )
                    ],
                    "cycle_2_rank": [
                        item["joint"]
                        for item in sorted(
                            joints,
                            key=lambda item: item["cycle_2_delta_evpe"],
                            reverse=True,
                        )
                    ],
                }
            finally:
                client.remove(model)
    finally:
        client.disconnect()

    payload = {
        "interpretation": (
            "(0,0) is the official outer corner joint at DNP 17.6777 mm; "
            "the other three are its first inward neighbors."
        ),
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
