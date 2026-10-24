#!/usr/bin/env python3
"""LILA camera-trap datasets with published MegaDetector v5a results.

Inputs:
  data/raw/lila_md_results/<dataset>_mdv5a.0.0_results.json.zip  MegaDetector output (from
      lila.science "MegaDetector results for camera trap datasets"; added to the repo by hand)
  hand-drawn bounding boxes, downloaded from LILA's GCP mirror into ~/.cache/countcal/lila

True count = number of boxes whose category is an animal (not empty / human / vehicle);
images with any human or vehicle box are dropped. Predicted count = MegaDetector animal
detections with confidence >= 0.2 (MegaDetector's recommended v5 threshold). MegaDetector's
training data includes these LILA datasets, so it is better here than on new cameras.

Populations (one CSV each in data/processed/):
  wcs_md_t20   WCS Camera Traps: the five sites with the largest mean counts (herds of
               white-lipped peccary etc.); the detector over-counts at some sites and
               under-counts at others.
  cict_md_t20  Channel Islands Camera Traps: 2,000 images sampled at random over all sites,
               including empty frames (mostly 0-1 foxes per image).
Caltech Camera Traps was checked and left out: almost every boxed image holds exactly one
animal and MegaDetector counts it correctly, so there is nothing to calibrate.
"""

import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from countcal.data import PROCESSED_DIR, REPO_ROOT  # noqa: E402

LILA = "https://storage.googleapis.com/public-datasets-lila"
MD_DIR = REPO_ROOT / "data" / "raw" / "lila_md_results"
CACHE = Path.home() / ".cache" / "countcal" / "lila"
THRESHOLD = 0.2
NON_ANIMAL = {"empty", "human", "homo sapiens", "person", "vehicle", "car"}
DROP_IF_PRESENT = {"human", "homo sapiens", "person", "vehicle", "car"}

DATASETS = {
    "wcs_md_t20": {
        "boxes": "wcs/wcs_20220205_bboxes_with_classes.zip",
        "md": "wcs-camera-traps_animals_mdv5a.0.0_results.json.zip",
        "select": {"sites": ["4313", "1198", "743", "739", "654"]},
    },
    "cict_md_t20": {
        "boxes": "channel-islands-camera-traps/channel-islands-camera-traps.json.zip",
        "md": "channel-islands-camera-traps_mdv5a.0.0_results.json.zip",
        "select": {"random": 2000, "seed": 0},
    },
}


def _load_json(path: Path) -> dict:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith(".json"))
            return json.loads(z.read(name))
    return json.loads(path.read_text())


def _boxes(rel: str) -> dict:
    path = CACHE / Path(rel).name
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"downloading {LILA}/{rel}")
        urllib.request.urlretrieve(f"{LILA}/{rel}", path)
    return _load_json(path)


def build(name: str, spec: dict) -> pd.DataFrame:
    ann = _boxes(spec["boxes"])
    cats = {c["id"]: c["name"].lower() for c in ann["categories"]}
    per_image: dict = {}
    for a in ann["annotations"]:
        if "bbox" in a:
            per_image.setdefault(a["image_id"], []).append(cats[a["category_id"]])
    md = _load_json(MD_DIR / spec["md"])
    det = {im["file"]: im.get("detections") or [] for im in md["images"] if "failure" not in im}

    rows = []
    for im in ann["images"]:
        b = per_image.get(im["id"])
        if b is None or im["file_name"] not in det or any(c in DROP_IF_PRESENT for c in b):
            continue
        rows.append({
            "image_id": im["file_name"],
            "site": str(im.get("location")),
            "true_count": sum(c not in NON_ANIMAL for c in b),
            "pred_count": sum(1 for x in det[im["file_name"]] if x["category"] == "1" and x["conf"] >= THRESHOLD),
        })
    df = pd.DataFrame(rows)
    sel = spec["select"]
    if "sites" in sel:
        df = df[df.site.isin(sel["sites"])]
    if "random" in sel:
        df = df.sample(n=sel["random"], random_state=sel["seed"])
    return df.sort_values(["site", "image_id"]).reset_index(drop=True)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, spec in DATASETS.items():
        df = build(name, spec)
        df.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} images, {df.site.nunique()} sites, mean true {df.true_count.mean():.2f}, "
              f"mean pred {df.pred_count.mean():.2f}, max true {df.true_count.max()}, "
              f"empty {np.mean(df.true_count == 0):.0%} -> data/processed/{name}.csv")


if __name__ == "__main__":
    main()
