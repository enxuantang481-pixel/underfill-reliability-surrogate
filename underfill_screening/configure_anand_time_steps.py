from __future__ import annotations

import argparse
from pathlib import Path

import mph

from comsol_corner_model import PORT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply conservative time-step controls to a meshed Anand model."
    )
    parser.add_argument("input_model", type=Path)
    parser.add_argument("output_model", type=Path)
    parser.add_argument("--initial-step", default="0.1[s]")
    parser.add_argument("--max-step", default="15[s]")
    parser.add_argument("--max-iterations", type=int, default=30)
    args = parser.parse_args()

    client = mph.Client(port=PORT, host="localhost")
    model = None
    try:
        model = client.load(str(args.input_model))
        time_solver = model.java.sol("sol1").feature("t1")
        time_solver.set("initialstepbdfactive", True)
        time_solver.set("initialstepbdf", args.initial_step)
        time_solver.set("maxstepconstraintbdf", "const")
        time_solver.set("maxstepbdf", args.max_step)
        time_solver.feature("fc1").set("maxiter", str(args.max_iterations))

        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        model.java.save(str(args.output_model))
        print(args.output_model)
    finally:
        if model is not None:
            client.remove(model)
        client.disconnect()


if __name__ == "__main__":
    main()
