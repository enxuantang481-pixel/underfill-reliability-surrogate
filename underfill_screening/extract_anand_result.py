from __future__ import annotations

import argparse
import json
from pathlib import Path

import mph

from comsol_corner_model import PORT, _evaluate_anand_metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--cycles", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    client = mph.Client(port=PORT, host="localhost")
    model = None
    try:
        model = client.load(str(args.model))
        times = [float(value) for value in model.java.sol("sol1").getPVals()]
        expression, damage_metric, values = _evaluate_anand_metric(
            model,
            times,
            args.case_id,
            args.cycles,
        )
        result = {
            "case_id": args.case_id,
            "status": "solved",
            "mode": "anand",
            "cycles": args.cycles,
            "field_expression": expression,
            "damage_metric": damage_metric,
            "metric_name": "last-cycle increment of corner-ball volume-average equivalent viscoplastic strain",
            "time_count": len(values),
            "field_times_s": times,
            "field_series": values,
            "solved_model": str(args.model),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        if model is not None:
            client.remove(model)
        client.disconnect()


if __name__ == "__main__":
    main()

