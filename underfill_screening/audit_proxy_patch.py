from __future__ import annotations

import json
import math
from pathlib import Path

import mph

from comsol_corner_model import PORT, _flatten_real


MODELS = {
    "reference_A": Path(
        r"E:\COMSEL\official_compliant_redesign_rms\reference_A\proxy\coarse"
        r"\official_corner_reference_A_proxy_coarse_solved.mph"
    ),
    "nominal_B": Path(
        r"E:\COMSEL\official_compliant_redesign_rms\nominal_B\proxy\coarse"
        r"\official_corner_nominal_B_proxy_coarse_solved.mph"
    ),
}

OUTPUT = Path(__file__).with_name("proxy_patch_audit.json")


def evaluate_joint(model, selection_name: str, tag: str) -> list[float]:
    numerical = model.java.result().numerical().create(tag, "AvVolume")
    numerical.selection().named(selection_name)
    numerical.set("expr", "solid.mises")
    numerical.set("data", "dset1")
    return _flatten_real(numerical.getReal())


def main() -> None:
    client = mph.Client(port=PORT, host="localhost")
    results: dict[str, dict] = {}
    try:
        for case_id, model_path in MODELS.items():
            model = client.load(str(model_path))
            try:
                joints: list[dict] = []
                for ix in range(5):
                    for iy in range(5):
                        values = evaluate_joint(
                            model,
                            f"sel_ball_{ix}_{iy}",
                            f"av_proxy_{case_id}_{ix}_{iy}",
                        )
                        joints.append(
                            {
                                "joint": f"({ix},{iy})",
                                "dnp_mm": math.hypot(12.5 - ix, 12.5 - iy),
                                "max_volume_average_mises_pa": max(values),
                                "series_pa": values,
                            }
                        )
                ranked = sorted(
                    joints,
                    key=lambda item: item["max_volume_average_mises_pa"],
                    reverse=True,
                )
                results[case_id] = {
                    "model": str(model_path),
                    "ranked_joints": ranked,
                    "corner_rank": next(
                        index
                        for index, item in enumerate(ranked, start=1)
                        if item["joint"] == "(0,0)"
                    ),
                }
            finally:
                client.remove(model)
    finally:
        client.disconnect()

    payload = {
        "metric": "max over 25/-40/125 degC of volume-average solder von Mises stress",
        "role": "patch-wide critical-location screen only; nonlinear EVPE ranking is reported separately",
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
