"""Scratch script: Audits time_penalty activation across positive re-entry events and negative pairs."""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from tracking.hybrid_matching import load_video_hybrid_data, time_penalty

pos_events, neg_pairs = load_video_hybrid_data()

print(f"Total Positive Events: {len(pos_events)}")
pos_penalties = [e["penalty"] for e in pos_events]
pos_nonzero = [p for p in pos_penalties if p > 0]
print(f"  - Positive events with penalty > 0: {len(pos_nonzero)} / {len(pos_events)}")
for idx, e in enumerate(pos_events, 1):
    print(f"    Event {idx:2d} ({e['video_name']:<30}): penalty = {e['penalty']:.4f}, ReID = {e['s_reid']:.4f}, Attr = {e['s_attr']:.4f}")

print(f"\nTotal Negative Pairs: {len(neg_pairs)}")
neg_penalties = [e["penalty"] for e in neg_pairs]
neg_nonzero = [p for p in neg_penalties if p > 0]
print(f"  - Negative pairs with penalty > 0: {len(neg_nonzero)} / {len(neg_pairs)}")

# Inspect delta_t gaps in JSON for each video
print("\n--- Track Time Range Breakdown per Video ---")
for v_name, json_path in [
    ("store-aisle-detection", "reports/tracking/track_attributes_store_aisle.json"),
    ("people-detection", "reports/tracking/track_attributes_people_detection.json"),
    ("person-bicycle-car-detection", "reports/tracking/track_attributes_person_bicycle_car.json"),
    ("vtest", "reports/tracking/track_attributes_vtest.json")
]:
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        print(f"\n=== {v_name} ({len(data)} tracks) ===")
        for tid, d in sorted(data.items(), key=lambda x: int(x[0])):
            f_t = d.get("first_seen_time", -1.0)
            l_t = d.get("last_seen_time", -1.0)
            print(f"  Track {tid:>2}: first_seen_time = {f_t:6.2f}s, last_seen_time = {l_t:6.2f}s (duration: {l_t - f_t:5.2f}s)")
