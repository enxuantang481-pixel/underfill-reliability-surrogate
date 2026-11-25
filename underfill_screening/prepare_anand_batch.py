from __future__ import annotations

import argparse
import json
from pathlib import Path

import mph

from comsol_corner_model import PORT, build_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and mesh an Anand case for a subsequent COMSOL Batch solve."
    )
    parser.add_argument("case_file", type=Path)
    parser.add_argument("--cycles", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument(
        "--mesh-level",
        choices=["coarse", "medium", "fine"],
        default="coarse",
    )
    args = parser.parse_args()

    case = json.loads(args.case_file.read_text(encoding="utf-8"))
    client = mph.Client(port=PORT, host="localhost")
    model = None
    try:
        model, case_dir, _ = build_model(
            client,
            case,
            mode="anand",
            mesh_level=args.mesh_level,
            cycles=args.cycles,
        )
        model.java.component("comp1").mesh("mesh1").run()
        meshed_path = (
            case_dir
            / f"official_corner_{case['case_id']}_anand_{args.mesh_level}_meshed.mph"
        )
        model.java.save(str(meshed_path))
        print(meshed_path)
    finally:
        if model is not None:
            client.remove(model)
        client.disconnect()


if __name__ == "__main__":
    main()
