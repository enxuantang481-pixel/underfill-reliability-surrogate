from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import mph

from comsol_corner_model import PORT


MODEL = Path(r"E:\COMSEL\estc_baseline_geometry.mph")
OUTPUT = Path(__file__).with_name("official_audit") / "step_domains.csv"


def rounded_mm(value_m: float) -> float:
    return round(value_m * 1000.0, 6)


def main() -> None:
    client = mph.Client(port=PORT, host="localhost")
    model = None
    try:
        model = client.load(str(MODEL))
        measure = model.java.component("comp1").geom("geom1").measureFinal()
        domain_count = int(measure.getNEntities()[-1])
        measure.selection().geom(3)
        rows: list[dict[str, float | int]] = []
        signatures: Counter[tuple[float, float, float]] = Counter()
        for domain in range(1, domain_count + 1):
            measure.selection().set([domain])
            xmin, xmax, ymin, ymax, zmin, zmax = [
                float(value) for value in measure.getBoundingBox()
            ]
            dx = rounded_mm(xmax - xmin)
            dy = rounded_mm(ymax - ymin)
            dz = rounded_mm(zmax - zmin)
            signature = (dx, dy, dz)
            signatures[signature] += 1
            rows.append(
                {
                    "domain": domain,
                    "xmin_mm": rounded_mm(xmin),
                    "xmax_mm": rounded_mm(xmax),
                    "ymin_mm": rounded_mm(ymin),
                    "ymax_mm": rounded_mm(ymax),
                    "zmin_mm": rounded_mm(zmin),
                    "zmax_mm": rounded_mm(zmax),
                    "dx_mm": dx,
                    "dy_mm": dy,
                    "dz_mm": dz,
                    "volume_mm3": float(measure.getVolume()) * 1e9,
                }
            )

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        print(f"domains={domain_count}")
        print(f"output={OUTPUT}")
        for signature, count in signatures.most_common(20):
            print(f"count={count:4d} bbox_mm={signature}")
    finally:
        if model is not None:
            client.remove(model)
        client.disconnect()


if __name__ == "__main__":
    main()
