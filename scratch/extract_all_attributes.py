"""Scratch script: Utility script to extract UPAR 40 attributes and Re-ID embeddings for 4 test videos."""
import os
import sys
import json
import numpy as np
from pathlib import Path

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tracking.track_attributes import load_par_model, predict_crop_probabilities, aggregate_track_predictions, load_track_metadata
from tracking.reid_embedding import ReIDExtractor, DEFAULT_CHECKPOINT_PATH
from datasets.upar.loader import get_upar_transforms

VIDEOS = [
    {
        "name": "store-aisle-detection",
        "crops_dir": "reports/tracking/crops/store_aisle",
        "tracks_csv": "reports/tracking/store-aisle-detection_tracks.csv",
        "out_json": "reports/tracking/track_attributes_store_aisle.json"
    },
    {
        "name": "people-detection",
        "crops_dir": "reports/tracking/crops/people_detection",
        "tracks_csv": "reports/tracking/people-detection_tracks.csv",
        "out_json": "reports/tracking/track_attributes_people_detection.json"
    },
    {
        "name": "person-bicycle-car-detection",
        "crops_dir": "reports/tracking/crops/person_bicycle_car",
        "tracks_csv": "reports/tracking/person-bicycle-car-detection_tracks.csv",
        "out_json": "reports/tracking/track_attributes_person_bicycle_car.json"
    },
    {
        "name": "vtest",
        "crops_dir": "reports/tracking/crops/vtest",
        "tracks_csv": "reports/tracking/vtest_tracks.csv",
        "out_json": "reports/tracking/track_attributes_vtest.json"
    }
]

def extract_for_all():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading UPAR model on {device}...")
    par_model = load_par_model("checkpoints/hydraplus_upar_best.pth", device)
    _, val_transform = get_upar_transforms(256, 128)

    print(f"[INFO] Loading Re-ID Extractor on {device}...")
    reid_extractor = ReIDExtractor(model_name="osnet_x1_0", checkpoint_path=DEFAULT_CHECKPOINT_PATH)

    for v_cfg in VIDEOS:
        v_name = v_cfg["name"]
        crops_dir = Path(v_cfg["crops_dir"])
        tracks_csv = Path(v_cfg["tracks_csv"])
        out_json_path = Path(v_cfg["out_json"])

        print(f"\n--- Extracting attributes & embeddings for {v_name} ---")
        if not crops_dir.exists() or not tracks_csv.exists():
            print(f"[SKIP] Missing crops or csv for {v_name}")
            continue

        tracks_meta = load_track_metadata(str(tracks_csv))
        track_folders = sorted(
            [d for d in crops_dir.iterdir() if d.is_dir() and d.name.startswith("track_")],
            key=lambda x: int(x.name.split("_")[1]) if x.name.split("_")[1].isdigit() else x.name
        )

        json_dict = {}

        for t_dir in track_folders:
            t_id = int(t_dir.name.split("_")[1])
            crop_files = sorted([str(f) for f in t_dir.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg"]])

            if not crop_files:
                continue

            # UPAR Attributes
            head_probs = predict_crop_probabilities(par_model, crop_files, val_transform, device)
            top1_summary, multi_label_summary, raw_40_probs = aggregate_track_predictions(head_probs, threshold=0.5)

            # Re-ID Embedding (Mean pooled 512-dim)
            crop_feats = reid_extractor.extract_features(crop_files)
            mean_emb = np.mean(crop_feats, axis=0)
            mean_emb = mean_emb / np.linalg.norm(mean_emb)
            emb_list = [round(float(val), 6) for val in mean_emb]

            meta = tracks_meta.get(t_id, {
                "first_seen_frame": -1, "last_seen_frame": -1,
                "first_seen_time": -1.0, "last_seen_time": -1.0
            })

            # Vector of raw probabilities in order (list of floats for cosine sim)
            raw_40_vector = [float(v) for v in raw_40_probs.values()]

            json_dict[str(t_id)] = {
                "track_id": t_id,
                "n_frames_used": len(crop_files),
                "first_seen_frame": meta["first_seen_frame"],
                "last_seen_frame": meta["last_seen_frame"],
                "first_seen_time": meta["first_seen_time"],
                "last_seen_time": meta["last_seen_time"],
                "top1_summary": top1_summary,
                "multi_label_heads": multi_label_summary,
                "raw_probabilities_40": raw_40_probs,
                "raw_probabilities_vector": raw_40_vector,
                "embedding": emb_list
            }

        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, indent=2)
        print(f"[DONE] Saved {len(json_dict)} tracks to '{out_json_path}'")

if __name__ == "__main__":
    import torch
    extract_for_all()
