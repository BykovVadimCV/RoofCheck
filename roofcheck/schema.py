"""Data model for the "photo from a taller building" roof check.

Layout on disk (one folder per object, one sub-folder per vantage point):

    data/objects/<object_id>/object.json
    data/objects/<object_id>/vantages/<vantage_id>/reference.jpg
    data/objects/<object_id>/vantages/<vantage_id>/reference.json
    data/objects/<object_id>/photos/<photo_id>.jpg          (incoming reports)
    data/objects/<object_id>/photos/<photo_id>.result.json  (module output)

A vantage point is "where the worker stands" (a roof of a taller building). The reference
is one frame from that point with the target roof marked once. Every new photo from the same
point is registered to the reference and the markup is carried over.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- classes
# Pixel classes inside the target-roof polygon (segmentation, stage 3).
SEG_CLASSES = [
    ("background", "Вне проверяемой кровли"),
    ("snow", "Снег на скате"),
    ("metal_clean", "Чистое покрытие, сухое"),
    ("metal_wet", "Мокрое покрытие, блик (не снег)"),
    ("ice", "Наледь на покрытии"),
    ("structure", "Трубы, вентшахты, надстройки, антенны"),
    ("fence", "Ограждение, снегозадержатели"),
    ("overhang", "Снежный свес над карнизом"),
    ("ignore", "Не размечается: наложения, перекрытие"),
]
# Eave-strip state per segment (stage 3, also used for icicles later).
EAVE_STATES = ["clean", "snow_on_edge", "overhang", "icicles", "not_visible"]

# Polygon roles used in references and in CVAT.
POLY_ROLES = {
    "roof": "Проверяемая кровля (видимая часть, как рисует заказчик)",
    "slope": "Скат: 4 угла по порядку карниз-лево, карниз-право, конёк-право, конёк-лево",
    "exclude": "Исключить из расчёта (своя кровля снимающего, посторонние предметы)",
    "overlay": "Наложения на кадре (метка времени, значок GPS)",
    "other_roof": "Другие кровли в кадре (для синтетики: на них тоже рисуется снег)",
    "structure": "Трубы, вентшахты, слуховые окна, антенны, тарелки на скате (не несут снег на скате, на верх ложится шапка)",
}


@dataclass
class Polygon:
    role: str                       # key of POLY_ROLES
    points: list[list[float]]       # [[x, y], ...] in reference-image pixels
    label: str = ""                 # free text (e.g. "скат во двор")


@dataclass
class Reference:
    object_id: str
    vantage_id: str
    image: str                      # file name next to reference.json
    width: int
    height: int
    polygons: list[Polygon] = field(default_factory=list)
    season: str = "autumn"          # autumn | winter: winter references are added at first snow
    geometry: dict = field(default_factory=dict)   # per-roof prior for synthesis: length_m, depth_m, pitch_deg
    source: str = ""                # where the markup came from (customer polygon, manual, ...)
    notes: str = ""

    def by_role(self, role: str) -> list[Polygon]:
        return [p for p in self.polygons if p.role == role]

    # ---- io
    def save(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / "reference.json"
        out.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1))
        return out

    @staticmethod
    def load(path: Path) -> "Reference":
        d = json.loads(Path(path).read_text())
        d["polygons"] = [Polygon(**p) for p in d.get("polygons", [])]
        return Reference(**d)


@dataclass
class ObjectInfo:
    object_id: str
    address: str = ""
    roof_type: str = "pitched_metal"   # pitched_metal | flat_soft | mixed
    vantages: list[str] = field(default_factory=list)
    notes: str = ""

    def save(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / "object.json"
        out.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1))
        return out
