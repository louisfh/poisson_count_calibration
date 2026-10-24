#!/usr/bin/env python3
"""Caltech Fish Counting (CFC; Kay et al., ECCV 2022): per-clip sonar fish counts.

Inputs, from https://data.caltech.edu/records/1y23m-j8r69, placed in data/raw/cfc/:
  fish_counting_annotations.tar.gz   ground-truth tracks (MOTChallenge gt.txt per clip)
  fish_counting_results.tar.gz       Baseline / Baseline++ tracker output (MOTChallenge)
  fish_counting_metadata.tar.gz      clip width/height per location

A clip's count is computed exactly as CFC's evaluate.py does: a track counts if its centre
moves from one half of the frame to the other between its first and last box, ignoring
tracks that move less than 0.05 (normalised). Left and right crossings are summed. The
same rule applied to ground-truth tracks gives the true count and to tracker output the
predicted count. One "image" here is one video clip; the site is the river location.

Writes data/processed/cfc_<tracker>.csv for each tracker found in the results archive.
"""

import io
import json
import math
import re
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from countcal.data import PROCESSED_DIR, REPO_ROOT  # noqa: E402

RAW = REPO_ROOT / "data" / "raw" / "cfc"
FILTER_DIST = 0.05


def count_crossings(rows, w, h):
    """rows: iterable of (frame, track_id, left, top, width, height), MOT 1-indexed pixels."""
    tracks = defaultdict(list)
    for frame, tid, x, y, bw, bh in sorted(rows):
        tracks[tid].append(((x - 1) / w + bw / w / 2, (y - 1) / h + bh / h / 2))
    n = 0
    for pts in tracks.values():
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        if math.hypot(x1 - x0, y1 - y0) < FILTER_DIST:
            continue
        if (x0 < 0.5) != (x1 < 0.5):
            n += 1
    return n


def read_mot(f):
    rows = []
    for line in io.TextIOWrapper(f):
        p = [v.strip() for v in line.split(",")]
        if len(p) >= 6:
            rows.append((int(float(p[0])), int(float(p[1])), *map(float, p[2:6])))
    return rows


def load_metadata():
    dims = {}
    with tarfile.open(RAW / "fish_counting_metadata.tar.gz") as t:
        for m in t.getmembers():
            if m.isfile() and m.name.endswith(".json"):
                location = Path(m.name).stem
                for clip in json.load(t.extractfile(m)):
                    dims[(location, clip["clip_name"])] = (clip["width"], clip["height"])
    return dims


def load_counts(archive, pattern):
    """Map (location, tracker_or_None, clip) -> list of MOT rows, for members matching pattern."""
    out = {}
    with tarfile.open(archive) as t:
        for m in t.getmembers():
            mt = pattern.search(m.name) if m.isfile() else None
            if mt:
                out[mt.groupdict().get("tracker"), mt["location"], mt["clip"]] = read_mot(t.extractfile(m))
    return out


def main():
    dims = load_metadata()
    gt = load_counts(RAW / "fish_counting_annotations.tar.gz",
                     re.compile(r"(?P<location>[^/]+)/(?P<clip>[^/]+)/gt\.txt$"))
    pred = load_counts(RAW / "fish_counting_results.tar.gz",
                       re.compile(r"(?P<location>[^/]+)/(?P<tracker>[^/]+)/data/(?P<clip>[^/]+)\.txt$"))
    print(f"{len(gt)} ground-truth clips, {len(pred)} tracker outputs, {len(dims)} clips with metadata")

    truth = {}
    for (_, location, clip), rows in gt.items():
        if (location, clip) in dims:
            truth[location, clip] = count_crossings(rows, *dims[location, clip])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for tracker in sorted({k[0] for k in pred}):
        recs = []
        for (tr, location, clip), rows in pred.items():
            if tr == tracker and (location, clip) in truth:
                recs.append({"image_id": clip, "site": location, "true_count": truth[location, clip],
                             "pred_count": count_crossings(rows, *dims[location, clip])})
        # Clips the tracker output nothing for have predicted count 0.
        seen = {(r["site"], r["image_id"]) for r in recs}
        locations = {r["site"] for r in recs}
        for (location, clip), f in truth.items():
            if location in locations and (location, clip) not in seen:
                recs.append({"image_id": clip, "site": location, "true_count": f, "pred_count": 0})
        df = pd.DataFrame(recs).sort_values(["site", "image_id"])
        name = "cfc_" + re.sub(r"[^a-z0-9]+", "", tracker.lower().replace("++", "pp"))
        df.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        print(f"{tracker}: {len(df)} clips, {df.site.nunique()} locations, mean true {df.true_count.mean():.2f}, "
              f"mean pred {df.pred_count.mean():.2f} -> data/processed/{name}.csv")
        print(df.groupby("site")[["true_count", "pred_count"]].agg(["size", "mean"]).round(2))


if __name__ == "__main__":
    main()
