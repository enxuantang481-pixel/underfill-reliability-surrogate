from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt


@dataclass(frozen=True)
class OfficialGeometry:
    pcb_length_mm: float = 100.0
    pcb_width_mm: float = 100.0
    pcb_thickness_mm: float = 1.51
    substrate_length_mm: float = 38.184
    substrate_width_mm: float = 38.184
    substrate_thickness_mm: float = 1.09
    bga_count_per_axis: int = 26
    bga_pitch_mm: float = 1.0
    solder_radius_mm: float = 0.30
    solder_height_mm: float = 0.40
    pad_diameter_mm: float = 0.45
    pad_thickness_mm: float = 0.03
    corner_patch_count: int = 5

    @property
    def bga_half_span_mm(self) -> float:
        return 0.5 * (self.bga_count_per_axis - 1) * self.bga_pitch_mm

    @property
    def corner_dnp_mm(self) -> float:
        return sqrt(2.0) * self.bga_half_span_mm

    @property
    def substrate_edge_margin_mm(self) -> float:
        return 0.5 * self.substrate_length_mm - self.bga_half_span_mm

    @property
    def interconnect_gap_mm(self) -> float:
        return 2.0 * self.pad_thickness_mm + self.solder_height_mm

    @property
    def patch_min_mm(self) -> float:
        return -self.substrate_edge_margin_mm

    @property
    def patch_max_mm(self) -> float:
        return (self.corner_patch_count - 0.5) * self.bga_pitch_mm

    @property
    def patch_size_mm(self) -> float:
        return self.patch_max_mm - self.patch_min_mm

    def to_dict(self) -> dict[str, float | int]:
        values = asdict(self)
        values.update(
            bga_half_span_mm=self.bga_half_span_mm,
            corner_dnp_mm=self.corner_dnp_mm,
            substrate_edge_margin_mm=self.substrate_edge_margin_mm,
            interconnect_gap_mm=self.interconnect_gap_mm,
            patch_min_mm=self.patch_min_mm,
            patch_max_mm=self.patch_max_mm,
            patch_size_mm=self.patch_size_mm,
        )
        return values


OFFICIAL_GEOMETRY = OfficialGeometry()

