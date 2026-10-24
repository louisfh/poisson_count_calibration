#!/usr/bin/env python3
"""Snapshot Serengeti camera-trap counts with MegaDetector predictions.

Ground truth: hand-drawn animal bounding boxes from the Snapshot Serengeti bounding-box
release (Beery / Yang, 2019-09-03, seasons 1-6), hosted by LILA BC. The true count of an
image is its number of "animal" boxes. Images with person, vehicle or "group" boxes are
dropped because their counts are ambiguous.

Prediction: MegaDetector v5a run on each image; the predicted count is the number of
animal detections above a confidence threshold. Several thresholds give several
"detectors" with different error profiles (low threshold -> false positives, high
threshold -> misses). Note MegaDetector's training data includes Snapshot Serengeti, so
this detector is better here than it would be on a new camera network.

The survey: all season-1 images from nine camera sites with relatively large groups.

Stages (run in order; each is resumable):
  python scripts/prepare_serengeti.py select      # pick images, write selection CSV
  python scripts/prepare_serengeti.py download    # fetch images from LILA's GCP mirror
  python scripts/prepare_serengeti.py detect      # run MegaDetector (CPU is fine, ~1-2 s/image)
  python scripts/prepare_serengeti.py build       # write data/processed/serengeti_md_t*.csv
"""

import argparse
import collections
import io
import json
import sys
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from countcal.data import PROCESSED_DIR, REPO_ROOT  # noqa: E402

LILA = "https://storage.googleapis.com/public-datasets-lila"
BBOX_URL = f"{LILA}/snapshotserengeti-v-2-0/SnapshotSerengetiBboxes_20190903.json.zip"
IMAGE_URL = f"{LILA}/snapshotserengeti-unzipped/{{file_name}}"

SEASON = "S1"
SITES = ["I13", "F10", "E06", "F11", "R10", "F08", "U11", "H08", "L10"]
THRESHOLDS = [0.1, 0.2, 0.5]
MIN_CONF_STORED = 0.05
MD_MODEL = "MDV5A"

RAW_DIR = REPO_ROOT / "data" / "raw" / "serengeti"
SELECTION = RAW_DIR / "selected_images.csv"
DETECTIONS = RAW_DIR / "megadetector_v5a_detections.json"
UNAVAILABLE = RAW_DIR / "unavailable_images.txt"  # listed in the bbox release but missing from the mirror
CACHE = Path.home() / ".cache" / "countcal" / "serengeti"


def select():
    print(f"downloading {BBOX_URL}")
    raw = urllib.request.urlopen(BBOX_URL).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        bb = json.loads(z.read(z.namelist()[0]))
    cats = {c["id"]: c["name"] for c in bb["categories"]}
    boxes = collections.defaultdict(list)
    for a in bb["annotations"]:
        boxes[a["image_id"]].append(cats[a["category_id"]])
    rows = []
    for im in bb["images"]:
        if im["season"] != SEASON or im["location"] not in SITES:
            continue
        b = boxes.get(im["id"], [])
        if any(c != "animal" for c in b):
            continue
        rows.append({"image_id": im["id"], "file_name": im["file_name"], "site": im["location"],
                     "seq_id": im.get("seq_id"), "datetime": im.get("datetime"), "true_count": len(b)})
    df = pd.DataFrame(rows).sort_values("image_id")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SELECTION, index=False)
    print(f"selected {len(df)} images from {df.site.nunique()} sites -> {SELECTION}")
    print(df.groupby("site").true_count.agg(["size", "mean", "max"]))


def _fetch(file_name):
    """Returns bytes downloaded, or None if the mirror does not have the image."""
    dest = CACHE / file_name
    if dest.exists():
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = urllib.request.urlopen(IMAGE_URL.format(file_name=file_name), timeout=120).read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.rename(dest)
    return len(data)


def _available(sel: pd.DataFrame) -> pd.DataFrame:
    if UNAVAILABLE.exists():
        bad = set(UNAVAILABLE.read_text().split())
        sel = sel[~sel.file_name.isin(bad)]
    return sel


def download():
    files = pd.read_csv(SELECTION).file_name.tolist()
    with ThreadPoolExecutor(16) as ex:
        sizes = list(ex.map(_fetch, files))
    missing = sorted(f for f, n in zip(files, sizes) if n is None)
    UNAVAILABLE.write_text("".join(f + "\n" for f in missing))
    total = sum(n for n in sizes if n)
    print(f"downloaded {total / 1e6:.0f} MB; {len(files) - len(missing)} images in {CACHE}; "
          f"{len(missing)} not on the mirror (listed in {UNAVAILABLE.name})")


def detect():
    from megadetector.detection.run_detector_batch import load_and_run_detector_batch

    sel = _available(pd.read_csv(SELECTION))
    done = json.loads(DETECTIONS.read_text()) if DETECTIONS.exists() else {}
    todo = [f for f in sel.file_name if f not in done]
    print(f"{len(todo)} images to run ({len(done)} done)")
    chunk = 100
    for i in range(0, len(todo), chunk):
        batch = [str(CACHE / f) for f in todo[i:i + chunk]]
        results = load_and_run_detector_batch(MD_MODEL, batch, confidence_threshold=MIN_CONF_STORED, quiet=True)
        for r in results:
            rel = str(Path(r["file"]).relative_to(CACHE))
            done[rel] = [[round(d["conf"], 4), d["category"]] for d in r.get("detections") or []]
        DETECTIONS.write_text(json.dumps(done, separators=(",", ":"), sort_keys=True))
        print(f"  {len(done)}/{len(sel)}", flush=True)


def build():
    sel = _available(pd.read_csv(SELECTION))
    det = json.loads(DETECTIONS.read_text())
    missing = [f for f in sel.file_name if f not in det]
    if missing:
        raise SystemExit(f"{len(missing)} images have no detections yet; run `detect` first")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for t in THRESHOLDS:
        pred = sel.file_name.map(lambda f: sum(1 for conf, cat in det[f] if cat == "1" and conf >= t))
        out = pd.DataFrame({"image_id": sel.image_id, "site": sel.site, "true_count": sel.true_count,
                            "pred_count": pred})
        path = PROCESSED_DIR / f"serengeti_md_t{int(t * 100):02d}.csv"
        out.to_csv(path, index=False)
        print(f"threshold {t}: mean true {out.true_count.mean():.2f}, mean pred {out.pred_count.mean():.2f} -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["select", "download", "detect", "build"])
    {"select": select, "download": download, "detect": detect, "build": build}[ap.parse_args().stage]()
