"""CVAT interchange: label set for the project and conversion of references to/from
"CVAT for images 1.1" XML (polygons). Masks of snow are labelled in CVAT with the same
label set and SAM-assisted tools; export them as "Segmentation mask 1.1".
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from .schema import POLY_ROLES, SEG_CLASSES, Polygon, Reference

PALETTE = {
    "roof": "#e2603a", "slope": "#f2b134", "exclude": "#5b5f7a", "overlay": "#8f95bd", "other_roof": "#1f8a72",
    "snow": "#ffffff", "metal_clean": "#6b7bc4", "metal_wet": "#2fa3c7", "ice": "#9fe3ff",
    "structure": "#8c6d4f", "fence": "#3c3c3c", "overhang": "#f4f3ee", "ignore": "#000000",
}


def label_spec() -> list[dict]:
    """JSON for CVAT: Project -> Constructor -> Raw. Polygon roles first, then pixel classes."""
    labels = [{"name": k, "color": PALETTE[k], "type": "polygon", "attributes": [
        {"name": "comment", "input_type": "text", "mutable": False, "values": [""]}]}
        for k in POLY_ROLES]
    labels += [{"name": k, "color": PALETTE[k], "type": "mask", "attributes": []}
               for k, _ in SEG_CLASSES if k != "background"]
    return labels


def write_label_spec(path: Path) -> Path:
    path.write_text(json.dumps(label_spec(), ensure_ascii=False, indent=1))
    return path


def references_to_cvat_xml(refs: list[Reference], path: Path) -> Path:
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    for i, r in enumerate(refs):
        im = ET.SubElement(root, "image", id=str(i), name=f"{r.object_id}/{r.vantage_id}/{r.image}",
                           width=str(r.width), height=str(r.height))
        for p in r.polygons:
            pts = ";".join(f"{x:.2f},{y:.2f}" for x, y in p.points)
            el = ET.SubElement(im, "polygon", label=p.role, occluded="0", source="manual", points=pts, z_order="0")
            if p.label:
                ET.SubElement(el, "attribute", name="comment").text = p.label
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def polygons_from_cvat_xml(path: Path) -> dict[str, list[Polygon]]:
    """{image name: [Polygon]} from a CVAT for images 1.1 export."""
    out: dict[str, list[Polygon]] = {}
    for im in ET.parse(path).getroot().iter("image"):
        polys = []
        for el in im.iter("polygon"):
            pts = [[float(a) for a in xy.split(",")] for xy in el.get("points").split(";")]
            comment = el.find("attribute[@name='comment']")
            polys.append(Polygon(role=el.get("label"), points=pts,
                                 label=comment.text if comment is not None and comment.text else ""))
        out[im.get("name")] = polys
    return out
