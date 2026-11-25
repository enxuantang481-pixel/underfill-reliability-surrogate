from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path


CSV_PATH = Path(__file__).with_name("official_audit") / "step_domains.csv"
OUTPUT = Path(__file__).with_name("official_audit") / "bga_layout.txt"


def main() -> None:
    with CSV_PATH.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    substrate = next(row for row in rows if int(row["domain"]) == 2)
    center_x = (float(substrate["xmin_mm"]) + float(substrate["xmax_mm"])) / 2.0
    center_y = (float(substrate["ymin_mm"]) + float(substrate["ymax_mm"])) / 2.0

    pad_rows = [
        row
        for row in rows
        if float(row["dx_mm"]) < 0.46 and float(row["dz_mm"]) < 0.04
    ]
    positions: set[tuple[float, float]] = set()
    for row in pad_rows:
        x = (float(row["xmin_mm"]) + float(row["xmax_mm"])) / 2.0
        y = (float(row["ymin_mm"]) + float(row["ymax_mm"])) / 2.0
        dx = x - center_x
        dy = y - center_y
        u = (dx + dy) / math.sqrt(2.0)
        v = (dy - dx) / math.sqrt(2.0)
        positions.add((round(u * 2.0) / 2.0, round(v * 2.0) / 2.0))

    solder = next(row for row in rows if int(row["domain"]) == 4)
    solder_x = (float(solder["xmin_mm"]) + float(solder["xmax_mm"])) / 2.0
    solder_y = (float(solder["ymin_mm"]) + float(solder["ymax_mm"])) / 2.0
    solder_u = ((solder_x - center_x) + (solder_y - center_y)) / math.sqrt(2.0)
    solder_v = ((solder_y - center_y) - (solder_x - center_x)) / math.sqrt(2.0)

    u_counts = Counter(u for u, _ in positions)
    v_counts = Counter(v for _, v in positions)
    u_values = sorted(u_counts)
    v_values = sorted(v_counts)

    corner_u = min(u_values)
    corner_v = max(v_values)
    local_rows: list[str] = []
    for u in [corner_u + offset for offset in range(5)]:
        line = []
        for v in [corner_v - offset for offset in range(5)]:
            line.append("X" if (u, v) in positions else ".")
        local_rows.append(f"u={u:5.1f}: " + " ".join(line))

    report = [
        f"substrate_center_mm=({center_x:.6f}, {center_y:.6f})",
        f"unique_joint_positions={len(positions)}",
        f"u_range_mm=({min(u_values):.1f}, {max(u_values):.1f})",
        f"v_range_mm=({min(v_values):.1f}, {max(v_values):.1f})",
        f"logical_pitch_mm={min(b-a for a,b in zip(u_values,u_values[1:])):.1f}",
        f"single_solder_center_global_mm=({solder_x:.6f}, {solder_y:.6f})",
        f"single_solder_center_local_mm=({solder_u:.6f}, {solder_v:.6f})",
        f"single_solder_DNP_mm={math.hypot(solder_u, solder_v):.6f}",
        "",
        "occupied_positions_per_u_row:",
        ", ".join(f"{u:.1f}:{u_counts[u]}" for u in u_values),
        "",
        "5x5 logical corner occupancy from official pads (X=present):",
        *local_rows,
    ]
    OUTPUT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
