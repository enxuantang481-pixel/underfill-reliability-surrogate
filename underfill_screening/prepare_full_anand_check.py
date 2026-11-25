from __future__ import annotations

import argparse
from pathlib import Path

import mph

from comsol_corner_model import PORT, _output_times


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a one-cycle validation model with Anand active in all "
            "25 solder joints."
        )
    )
    parser.add_argument("input_model", type=Path)
    parser.add_argument("output_model", type=Path)
    args = parser.parse_args()

    client = mph.Client(port=PORT, host="localhost")
    model = None
    try:
        model = client.load(str(args.input_model))
        java = model.java
        viscoplasticity = (
            java.component("comp1")
            .physics("solid")
            .feature("lemm1")
            .feature("vp1")
        )
        viscoplasticity.selection().named("sel_solder")

        time_list = " ".join(
            str(value) for value in _output_times("anand", cycles=1)
        )
        java.study("std1").feature("time").set("tlist", time_list)
        java.sol("sol1").feature("t1").set("tlist", time_list)

        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        java.modelPath(str(args.output_model.parent))
        java.label(args.output_model.name)
        java.save(str(args.output_model))
        print(args.output_model)
    finally:
        if model is not None:
            client.remove(model)
        client.disconnect()


if __name__ == "__main__":
    main()
