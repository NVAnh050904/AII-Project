#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tracking/run_pipeline.py
========================
End-to-End Orchestrator Pipeline for Video Person Tracking, Attribute Recognition & Re-ID Embedding.

Runs steps 1 to 5 automatically without human intervention:
1. track.py (Level 1: YOLOv8 + ByteTrack)
2. extract_crops.py (Level 2: Crop extraction)
3. track_attributes.py (Level 2: UPAR attribute aggregation)
4. reid_embedding.py (Level 3: OSNet 512-dim embedding extraction)
5. demo_combined.py (Combined Demo Video Generation)

Usage:
    python tracking/run_pipeline.py --video-name real_pedestrians
    python tracking/run_pipeline.py --video-name classroom --conf 0.35 --every-n-frames 5
"""

import os
import sys
import time
import argparse
import subprocess
import shutil
from pathlib import Path

# Set encoding for console output
sys.stdout.reconfigure(encoding='utf-8')

def parse_args():
    parser = argparse.ArgumentParser(description="Automated Video Tracking, Attribute & Re-ID Pipeline Orchestrator")
    parser.add_argument("--video-name", type=str, required=True,
                        help="Tên video (vd: 'real_pedestrians', 'classroom', 'people-detection')")
    parser.add_argument("--video-path", type=str, default=None,
                        help="Đường dẫn file video đầu vào (vd: 'tracking/test_videos/real_pedestrians.mp4')")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Confidence threshold cho YOLOv8 person detection (mặc định: 0.35)")
    parser.add_argument("--every-n-frames", type=int, default=5,
                        help="Tần suất lấy crop (mặc định: 5 frame lấy 1 crop)")
    return parser.parse_args()

def run_command(cmd, step_name):
    print(f"\n" + "=" * 70)
    print(f"PIPELINE STEP: {step_name}")
    print("Command:", " ".join(cmd))
    print("=" * 70)
    
    start_t = time.time()
    res = subprocess.run(cmd, check=False)
    elapsed = time.time() - start_t
    
    if res.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed with exit code {res.returncode}.")
        sys.exit(res.returncode)
    
    print(f"[SUCCESS] Step '{step_name}' completed in {elapsed:.2f} seconds.")
    return elapsed

def main():
    args = parse_args()
    video_name = args.video_name
    python_exe = sys.executable

    base_dir = Path(__file__).resolve().parent.parent

    # 0. Resolve Video Path
    if args.video_path:
        video_path = Path(args.video_path)
    else:
        mp4_path = base_dir / "tracking" / "test_videos" / f"{video_name}.mp4"
        avi_path = base_dir / "tracking" / "test_videos" / f"{video_name}.avi"
        if mp4_path.exists():
            video_path = mp4_path
        elif avi_path.exists():
            video_path = avi_path
        else:
            video_path = mp4_path

    if not video_path.exists():
        raise FileNotFoundError(f"[ERROR] Không tìm thấy file video tại: {video_path}")

    print("=" * 70)
    print(f"AUTOMATED TRACKING & ATTRIBUTE PIPELINE FOR: {video_name}")
    print(f"Video Source: {video_path}")
    print(f"YOLO Conf  : {args.conf}")
    print(f"Crop Step  : Every {args.every_n_frames} frames")
    print("=" * 70)

    total_pipeline_start = time.time()

    # Directories
    tracking_out_dir = base_dir / "reports" / "tracking" / video_name
    tracking_out_dir.mkdir(parents=True, exist_ok=True)

    crops_out_dir = base_dir / "reports" / "tracking" / "crops" / video_name
    crops_out_dir.mkdir(parents=True, exist_ok=True)

    tracks_csv = tracking_out_dir / "tracks.csv"

    # Step 1: track.py
    cmd_1 = [
        python_exe, str(base_dir / "tracking" / "track.py"),
        "--source", str(video_path),
        "--output-dir", str(tracking_out_dir),
        "--conf", str(args.conf),
        "--save-video"
    ]
    run_command(cmd_1, "1. Tracking (YOLOv8 + ByteTrack)")

    # Ensure tracks.csv exists (sync from <video_name>_tracks.csv if needed)
    raw_tracks_csv = tracking_out_dir / f"{video_name}_tracks.csv"
    if raw_tracks_csv.exists() and not tracks_csv.exists():
        shutil.copy(str(raw_tracks_csv), str(tracks_csv))

    # Step 2: extract_crops.py
    cmd_2 = [
        python_exe, str(base_dir / "tracking" / "extract_crops.py"),
        "--video", str(video_path),
        "--csv", str(tracks_csv),
        "--output-dir", str(crops_out_dir),
        "--every-n-frames", str(args.every_n_frames),
        "--clean"
    ]
    run_command(cmd_2, "2. Person Crop Extraction")

    # Step 3: track_attributes.py
    cmd_3 = [
        python_exe, str(base_dir / "tracking" / "track_attributes.py"),
        "--crops-dir", str(crops_out_dir),
        "--tracks-csv", str(tracks_csv),
        "--output-dir", str(tracking_out_dir),
        "--checkpoint", str(base_dir / "checkpoints" / "hydraplus_upar_best.pth")
    ]
    run_command(cmd_3, "3. UPAR Attribute Aggregation")

    # Step 4: reid_embedding.py
    cmd_4 = [
        python_exe, str(base_dir / "tracking" / "reid_embedding.py"),
        "--video-name", video_name,
        "--skip-market-validation"
    ]
    run_command(cmd_4, "4. Re-ID OSNet 512-dim Embedding Extraction")

    # Step 5: demo_combined.py
    cmd_5 = [
        python_exe, str(base_dir / "tracking" / "demo_combined.py"),
        "--video-name", video_name
    ]
    run_command(cmd_5, "5. Demo Combined Video Generation")

    total_pipeline_elapsed = time.time() - total_pipeline_start

    # Determine output demo path
    if video_name == "store-aisle-detection":
        demo_output_file = "reports/tracking/demo/demo_combined_v2_full_attributes.mp4"
    else:
        demo_output_file = f"reports/tracking/demo/demo_{video_name}_v2.mp4"

    # Terminal Instructions & Completion Banner
    print("\n" + "=" * 70)
    print(f"=== HOAN TAT Level 1+2+3 (embedding) tu dong (Tong thoi gian: {total_pipeline_elapsed:.1f}s) ===")
    print(f"Video demo (chua co Re-ID that vi chua xac dinh ground-truth):")
    print(f"{demo_output_file}")
    print()
    print("NEU video nay CO nguoi bi che khuat/mat khung hinh roi quay lai va ban")
    print("muon co Re-ID that (khong chi Tracking don thuan), can lam THEM cac buoc")
    print("SAU (KHONG THE tu dong hoa vi can mat nguoi xac nhan):")
    print(f"1. Xem tracked.mp4 vua tao (reports/tracking/{video_name}/tracked.mp4), quan sat co ai bi mat track roi xuat hien")
    print("   voi track_id khac khong")
    print("2. Tao anh strip so sanh cac track nghi ngo (co the dung script co san")
    print("   tu buoc lam voi store-aisle-detection lam mau)")
    print(f"3. Tao thu cong file reports/tracking/{video_name}/reentry_ground_truth.csv")
    print(f"4. Chay lai: python tracking/demo_combined.py --video-name {video_name}")
    print("   (lan nay se tu dong nhan ra co ground-truth va bat che do Re-ID day du)")
    print("=" * 70)

if __name__ == "__main__":
    main()
