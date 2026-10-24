#!/usr/bin/env python3
"""Build data/processed/{fish,rays}.csv from the raw files already in the repo.

fish: data/raw/fish_discount_f_g_2025-11-19.json  (f = true count, g = detector count, clip id)
rays: data/raw/rays_maria_frame_counts.csv         (one row per annotated video frame)

Rays site names are lower-cased: the raw file has both "Pescador" and "pescador" (and
"Rincon"/"rincon") for different videos. video_id is kept as image_id prefix.
"""

import json

import pandas as pd

from countcal.data import PROCESSED_DIR, REPO_ROOT

RAW = REPO_ROOT / "data" / "raw"


def fish():
    raw = json.loads((RAW / "fish_discount_f_g_2025-11-19.json").read_text())
    clip = pd.Series(raw["clip"])
    return pd.DataFrame({
        "image_id": clip,
        "site": clip.str.slice(0, 10),  # clip ids are YYYY-MM-DD_HHMMSS; use the date as a block
        "true_count": raw["f"],
        "pred_count": raw["g"],
    })


def rays():
    raw = pd.read_csv(RAW / "rays_maria_frame_counts.csv")
    return pd.DataFrame({
        "image_id": raw["video_id"] + "_" + raw["frame_id"].astype(str),
        "site": raw["site"].str.lower(),
        "true_count": raw["true_count"],
        "pred_count": raw["pred_count"],
    })


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, build in [("fish", fish), ("rays", rays)]:
        df = build()
        df.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} images -> {PROCESSED_DIR / (name + '.csv')}")
