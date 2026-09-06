"""Scratch script: Verifies fields and structure of track_attributes*.json files."""
import json
from pathlib import Path

paths = [
    "reports/tracking/track_attributes.json",
    "reports/tracking/track_attributes_store_aisle.json",
    "reports/tracking/track_attributes_vtest.json"
]

for path in paths:
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"{path}: {len(data)} tracks.")
        first_key = list(data.keys())[0]
        item = data[first_key]
        print(f"  Track {first_key} fields: {list(item.keys())}")
        if "raw_probabilities_40" in item:
            print(f"  raw_probabilities_40 len: {len(item['raw_probabilities_40'])}")
        if "embedding" in item:
            print(f"  embedding len: {len(item['embedding'])}")
    else:
        print(f"{path}: NOT FOUND")
