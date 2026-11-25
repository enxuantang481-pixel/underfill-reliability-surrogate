from __future__ import annotations

import argparse
import csv
from pathlib import Path
import json
import traceback

import mph

from active_learning import initial_cases, propose_cases, read_solved_results
from comsol_corner_model import OUTPUT_ROOT, PORT, solve_case


def _write_case_table(cases: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for case in cases for key in case})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(cases)


def _write_summary(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _add_reference_ratios(rows: list[dict]) -> None:
    reference = next(
        (row for row in rows if row.get("case_id") == "reference_A" and row.get("status") == "solved"),
        None,
    )
    if reference is None:
        return
    reference_damage = reference["damage_metric"]
    for row in rows:
        if row.get("status") == "solved" and row.get("damage_metric", 0) > 0:
            row["damage_ratio_vs_A"] = row["damage_metric"] / reference_damage
            row["reduction_vs_A_percent"] = 100.0 * (1.0 - row["damage_ratio_vs_A"])


def _proxy_cases(round_index: int, batch_size: int) -> list[dict]:
    solved = read_solved_results(OUTPUT_ROOT, mode="proxy", mesh_level="coarse")
    if round_index == 0:
        return initial_cases()
    return propose_cases(solved, batch_size=batch_size, seed=2026 + round_index)


def _anand_cases(limit: int) -> list[dict]:
    solved_proxy = read_solved_results(OUTPUT_ROOT, mode="proxy", mesh_level="coarse")
    reference = next((row for row in solved_proxy if row["case_id"] == "reference_A"), None)
    if reference is None:
        raise RuntimeError("Proxy reference_A must be solved before Anand validation.")
    candidates = sorted(
        [row for row in solved_proxy if row["case_id"] != "reference_A"],
        key=lambda row: row["damage_metric"],
    )[: max(1, limit - 1)]
    selected = [reference] + candidates
    return [
        {
            key: row[key]
            for key in ["case_id", "variant", "elastic_scale", "cte_scale", "tg_shift_c", "source"]
        }
        for row in selected
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["proxy", "anand"], default="proxy")
    parser.add_argument("--mesh", choices=["coarse", "medium", "fine"], default="coarse")
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--case-id")
    parser.add_argument("--cycles", type=int, choices=[1, 2, 3], default=3)
    args = parser.parse_args()

    if args.mode == "proxy":
        cases = _proxy_cases(args.round, args.batch_size)
    else:
        cases = _anand_cases(args.max_cases or 4)
    if args.case_id:
        cases = [case for case in cases if case["case_id"] == args.case_id]
    if args.max_cases:
        cases = cases[: args.max_cases]
    if not cases:
        raise RuntimeError("No cases selected.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_case_table(cases, OUTPUT_ROOT / f"submitted_{args.mode}_round_{args.round}.csv")
    client = mph.Client(port=PORT, host="localhost")
    rows = []
    try:
        for case in cases:
            print(f"[{args.mode}] {case['case_id']}", flush=True)
            try:
                rows.append(solve_case(client, case, args.mode, args.mesh, args.cycles))
            except Exception as exc:
                rows.append({**case, "status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
            _add_reference_ratios(rows)
            _write_summary(rows, OUTPUT_ROOT / f"summary_{args.mode}_round_{args.round}.json")
    finally:
        client.disconnect()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

