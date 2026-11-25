from __future__ import annotations

import argparse
from pathlib import Path

import mph

from comsol_corner_model import PORT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configure PARDISO out-of-core operation for memory-limited solves."
    )
    parser.add_argument("input_model", type=Path)
    parser.add_argument("output_model", type=Path)
    args = parser.parse_args()

    client = mph.Client(port=PORT, host="localhost")
    model = None
    try:
        model = client.load(str(args.input_model))
        direct = model.java.sol("sol1").feature("t1").feature("d1")
        direct.set("ooc", "on")
        direct.set("memfracooc", "0.50")
        direct.set("usetotmemory", "0.50")
        direct.set("oocmemory", "2048")

        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        model.java.save(str(args.output_model))
        print(args.output_model)
    finally:
        if model is not None:
            client.remove(model)
        client.disconnect()


if __name__ == "__main__":
    main()
