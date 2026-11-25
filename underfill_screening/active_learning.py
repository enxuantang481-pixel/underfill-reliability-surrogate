from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


@dataclass(frozen=True)
class DesignBounds:
    elastic_scale: tuple[float, float] = (0.75, 1.10)
    cte_scale: tuple[float, float] = (0.60, 1.00)
    tg_shift_c: tuple[float, float] = (-5.0, 15.0)


BOUNDS = DesignBounds()


def _case(
    case_id: str,
    variant: str,
    elastic_scale: float,
    cte_scale: float,
    tg_shift_c: float,
    source: str,
) -> dict:
    return {
        "case_id": case_id,
        "variant": variant,
        "elastic_scale": float(elastic_scale),
        "cte_scale": float(cte_scale),
        "tg_shift_c": float(tg_shift_c),
        "source": source,
    }


def initial_cases(seed: int = 2026, lhs_count: int = 8) -> list[dict]:
    cases = [
        _case("reference_A", "A", 1.0, 1.0, 0.0, "official nominal reference"),
        _case("nominal_B", "B", 1.0, 1.0, 0.0, "official nominal alternative"),
    ]
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    samples = qmc.scale(
        sampler.random(lhs_count),
        [BOUNDS.elastic_scale[0], BOUNDS.cte_scale[0], BOUNDS.tg_shift_c[0]],
        [BOUNDS.elastic_scale[1], BOUNDS.cte_scale[1], BOUNDS.tg_shift_c[1]],
    )
    for index, values in enumerate(samples):
        variant = "A" if index < max(2, lhs_count // 4) else "B"
        cases.append(
            _case(
                f"lhs_{index + 1:02d}_{variant}",
                variant,
                values[0],
                values[1],
                values[2],
                "screening DOE; bounds require supplier-property confirmation",
            )
        )
    return cases


def encode_case(case: dict) -> np.ndarray:
    return np.array(
        [
            0.0 if case["variant"] == "A" else 1.0,
            case["elastic_scale"],
            case["cte_scale"],
            case["tg_shift_c"],
        ],
        dtype=float,
    )


def read_solved_results(
    results_dir: Path,
    mode: str | None = None,
    mesh_level: str | None = None,
) -> list[dict]:
    rows_by_case: dict[str, dict] = {}
    for path in sorted(results_dir.rglob("result.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "solved" or not np.isfinite(data.get("damage_metric", np.nan)):
            continue
        if mode is not None and data.get("mode") != mode:
            continue
        if mesh_level is not None and data.get("mesh_level") != mesh_level:
            continue
        rows_by_case[data["case_id"]] = data
    return list(rows_by_case.values())


def _candidate_pool(seed: int, count: int = 1024) -> list[dict]:
    sampler = qmc.Sobol(d=4, scramble=True, seed=seed)
    exponent = int(np.ceil(np.log2(count)))
    samples = sampler.random_base2(exponent)[:count]
    pool = []
    for index, values in enumerate(samples):
        variant = "A" if values[0] < 0.25 else "B"
        elastic_scale = BOUNDS.elastic_scale[0] + values[1] * (
            BOUNDS.elastic_scale[1] - BOUNDS.elastic_scale[0]
        )
        cte_scale = BOUNDS.cte_scale[0] + values[2] * (
            BOUNDS.cte_scale[1] - BOUNDS.cte_scale[0]
        )
        tg_shift_c = BOUNDS.tg_shift_c[0] + values[3] * (
            BOUNDS.tg_shift_c[1] - BOUNDS.tg_shift_c[0]
        )
        pool.append(
            _case(
                f"gp_pool_{index + 1:04d}_{variant}",
                variant,
                elastic_scale,
                cte_scale,
                tg_shift_c,
                "Gaussian-process active-learning proposal",
            )
        )
    return pool


def propose_cases(
    solved_rows: list[dict],
    batch_size: int = 4,
    seed: int = 2026,
) -> list[dict]:
    if len(solved_rows) < 5:
        solved_ids = {row["case_id"] for row in solved_rows}
        return [case for case in initial_cases(seed) if case["case_id"] not in solved_ids][:batch_size]

    features = np.vstack([encode_case(row) for row in solved_rows])
    targets = np.log(np.array([row["damage_metric"] for row in solved_rows], dtype=float))
    feature_min = np.array([0.0, BOUNDS.elastic_scale[0], BOUNDS.cte_scale[0], BOUNDS.tg_shift_c[0]])
    feature_max = np.array([1.0, BOUNDS.elastic_scale[1], BOUNDS.cte_scale[1], BOUNDS.tg_shift_c[1]])
    normalized_features = (features - feature_min) / (feature_max - feature_min)
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=np.ones(4),
        length_scale_bounds=(5e-2, 20.0),
        nu=2.5,
    ) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-9, 1e-2))
    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=seed,
    )
    model.fit(normalized_features, targets)

    pool = _candidate_pool(seed + len(solved_rows))
    pool_features = np.vstack([encode_case(case) for case in pool])
    normalized_pool = (pool_features - feature_min) / (feature_max - feature_min)
    mean, std = model.predict(normalized_pool, return_std=True)
    lower_confidence_bound = mean - 1.5 * std
    ordered_indices = np.argsort(lower_confidence_bound)

    solved_normalized = normalized_features
    selected_indices: list[int] = []
    for index in ordered_indices:
        candidate = normalized_pool[index]
        distance_to_solved = np.linalg.norm(solved_normalized - candidate, axis=1).min()
        distance_to_selected = min(
            (np.linalg.norm(normalized_pool[chosen] - candidate) for chosen in selected_indices),
            default=np.inf,
        )
        if distance_to_solved >= 0.06 and distance_to_selected >= 0.12:
            selected_indices.append(int(index))
        if len(selected_indices) == batch_size:
            break

    proposals = []
    for rank, index in enumerate(selected_indices, start=1):
        proposal = dict(pool[index])
        proposal["case_id"] = f"al_{len(solved_rows):03d}_{rank:02d}_{proposal['variant']}"
        proposal["predicted_log_damage"] = float(mean[index])
        proposal["predicted_log_damage_std"] = float(std[index])
        proposals.append(proposal)
    return proposals

