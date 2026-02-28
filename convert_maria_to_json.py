#!/usr/bin/env python3
"""Convert maria_frame_counts.csv to JSON format like discount f/g counts."""

import csv
import json
import sys
from pathlib import Path


def convert(csv_path: str, json_path: str | None = None) -> list[dict]:
    """Convert Maria's frame counts CSV to JSON, preserving all columns."""
    csv_path = Path(csv_path)
    if json_path is None:
        json_path = csv_path.with_suffix(".json")

    rows = []
    f_list = []
    g_list = []
    with open(csv_path) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append({
                "video_id": row["video_id"],
                "site": row["site"],
                "frame_id": int(row["frame_id"]),
                "pred_count": int(row["pred_count"]),
                "true_count": int(row["true_count"]),
            })
            f_list.append(int(row["true_count"]))
            g_list.append(int(row["pred_count"]))

    data = {
        "frames": rows,
        "f": f_list,
        "g": g_list,
    }

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

    return data


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/maria_frame_counts.csv"
    json_path = sys.argv[2] if len(sys.argv) > 2 else None
    convert(csv_path, json_path)
    out = Path(json_path) if json_path else Path(csv_path).with_suffix(".json")
    print(f"Wrote {out}")
