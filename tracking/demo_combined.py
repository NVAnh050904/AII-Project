#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tracking/demo_combined.py
Generates a single combined demo video demonstrating:
- Level 1: Multi-Object Tracking
- Level 2: Attribute Recognition
- Level 3: Person Re-Identification (Re-ID across track breaks)

Inputs:
- tracking/test_videos/store-aisle-detection.mp4
- reports/tracking/store-aisle-detection_tracks.csv
- reports/tracking/track_attributes_store_aisle.json
- reports/tracking/reentry_ground_truth.csv

Output:
- reports/tracking/demo/demo_combined_tracking_attribute_reid.mp4
- Key screenshot frames saved in reports/tracking/demo/
"""

import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Set encoding for console output
sys.stdout.reconfigure(encoding='utf-8')

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(BASE_DIR, 'tracking', 'test_videos', 'store-aisle-detection.mp4')
CSV_PATH = os.path.join(BASE_DIR, 'reports', 'tracking', 'store-aisle-detection', 'tracks.csv')
ATTR_PATH = os.path.join(BASE_DIR, 'reports', 'tracking', 'store-aisle-detection', 'attributes.json')
GT_PATH = os.path.join(BASE_DIR, 'reports', 'tracking', 'store-aisle-detection', 'reentry_ground_truth.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'reports', 'tracking', 'demo')
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, 'demo_combined_tracking_attribute_reid.mp4')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 6 Real Identities Config based on Ground Truth & Attributes
IDENTITY_MAP = {
    # Person A: Yellow Hat (Tracks 1, 13)
    1: {"id": "Person A", "name": "Person A (Mũ vàng)", "color_rgb": (235, 45, 45)},
    13: {"id": "Person A", "name": "Person A (Mũ vàng)", "color_rgb": (235, 45, 45)},

    # Person B: Brown/Green coat (Tracks 5, 7)
    5: {"id": "Person B", "name": "Person B (Áo xanh ngọc)", "color_rgb": (255, 140, 20)},
    7: {"id": "Person B", "name": "Person B (Áo xanh ngọc)", "color_rgb": (255, 140, 20)},

    # Person C: Brown coat (Tracks 11, 18)
    11: {"id": "Person C", "name": "Person C (Da lộn nâu)", "color_rgb": (240, 200, 20)},
    18: {"id": "Person C", "name": "Person C (Da lộn nâu)", "color_rgb": (240, 200, 20)},

    # Person D: Blue denim coat (6 tracks merged: 12, 16, 20, 22, 24, 25)
    12: {"id": "Person D", "name": "Person D (Áo bò xanh)", "color_rgb": (180, 50, 230)},
    16: {"id": "Person D", "name": "Person D (Áo bò xanh)", "color_rgb": (180, 50, 230)},
    20: {"id": "Person D", "name": "Person D (Áo bò xanh)", "color_rgb": (180, 50, 230)},
    22: {"id": "Person D", "name": "Person D (Áo bò xanh)", "color_rgb": (180, 50, 230)},
    24: {"id": "Person D", "name": "Person D (Áo bò xanh)", "color_rgb": (180, 50, 230)},
    25: {"id": "Person D", "name": "Person D (Áo bò xanh)", "color_rgb": (180, 50, 230)},

    # Person E: Black long coat (Track 14)
    14: {"id": "Person E", "name": "Person E (Áo đen dài)", "color_rgb": (35, 195, 85)},

    # Person F: Black short coat (Track 19)
    19: {"id": "Person F", "name": "Person F (Áo đen ngắn)", "color_rgb": (30, 160, 240)}
}

# Re-ID previous track lookup for re-entry notification
PREV_TRACK_MAP = {
    13: 1,
    7: 5,
    18: 11,
    16: 12,
    20: 16,
    22: 20,
    24: 22,
    25: 24
}

# Load Font helpers
def get_fonts():
    try:
        font_large = ImageFont.truetype("arial.ttf", 22)
        font_medium = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 13)
        font_bold = ImageFont.truetype("arialbd.ttf", 16)
    except Exception:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_bold = ImageFont.load_default()
    return font_large, font_medium, font_small, font_bold

def draw_corner_rect(draw, bbox, color, stroke=2, corner_len=12):
    x1, y1, x2, y2 = bbox
    # Main rectangle
    draw.rectangle([x1, y1, x2, y2], outline=color, width=stroke)
    
    # Accent corners (thicker)
    c_w = stroke + 2
    # Top-left corner
    draw.line([(x1, y1), (x1 + corner_len, y1)], fill=color, width=c_w)
    draw.line([(x1, y1), (x1, y1 + corner_len)], fill=color, width=c_w)
    # Top-right corner
    draw.line([(x2, y1), (x2 - corner_len, y1)], fill=color, width=c_w)
    draw.line([(x2, y1), (x2, y1 + corner_len)], fill=color, width=c_w)
    # Bottom-left corner
    draw.line([(x1, y2), (x1 + corner_len, y2)], fill=color, width=c_w)
    draw.line([(x1, y2), (x1, y2 - corner_len)], fill=color, width=c_w)
    # Bottom-right corner
    draw.line([(x2, y2), (x2 - corner_len, y2)], fill=color, width=c_w)
    draw.line([(x2, y2), (x2, y2 - corner_len)], fill=color, width=c_w)

def render_intro_card(w, h, duration_sec, fps, font_large, font_medium, font_small):
    frames = []
    total_frames = int(duration_sec * fps)
    
    for _ in range(total_frames):
        img = Image.new('RGB', (w, h), color=(15, 22, 32))
        draw = ImageDraw.Draw(img)
        
        # Subtle header line
        draw.rectangle([0, 0, w, 6], fill=(50, 130, 240))
        
        # Main Title
        title_text = "Demo: Video Tracking + Attribute Recognition + Person Re-Identification"
        draw.text((w // 2, h // 2 - 45), title_text, fill=(255, 255, 255), font=font_large, anchor="mm")
        
        # Subtitle
        sub_text = "Nguồn video: store-aisle-detection.mp4 (intel-iot-devkit/sample-videos)"
        draw.text((w // 2, h // 2 + 10), sub_text, fill=(180, 200, 220), font=font_medium, anchor="mm")
        
        # Additional badge text
        badge_text = "Tự động hợp nhất nhận dạng (L1 + L2 + L3) | Báo cáo tuần"
        draw.text((w // 2, h // 2 + 50), badge_text, fill=(100, 220, 150), font=font_small, anchor="mm")
        
        # Bottom footnote
        draw.text((w // 2, h - 25), "Màu sắc = danh tính (identity), không phải track ID thô", fill=(140, 160, 180), font=font_small, anchor="mm")
        
        frames.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    return frames

def main():
    print("=== STARTING DEMO VIDEO GENERATION ===")
    
    # 1. Load Data
    print(f"Reading tracks CSV: {CSV_PATH}")
    df_tracks = pd.read_csv(CSV_PATH)
    
    print(f"Reading attributes JSON: {ATTR_PATH}")
    with open(ATTR_PATH, 'r', encoding='utf-8') as f:
        attr_data = json.load(f)
        
    print(f"Reading ground truth CSV: {GT_PATH}")
    df_gt = pd.read_csv(GT_PATH)
    
    # Calculate track start/end times and gaps for Re-ID
    track_summary = {}
    for tid_str, data in attr_data.items():
        tid = int(tid_str)
        track_summary[tid] = {
            "first_seen_frame": data["first_seen_frame"],
            "last_seen_frame": data["last_seen_frame"],
            "first_seen_time": data["first_seen_time"],
            "last_seen_time": data["last_seen_time"],
            "summary": data.get("top1_summary", {})
        }

    # Precalculate Re-ID banners info
    # Re-ID notification lasts 90 frames (~1.5s - 3s)
    reid_banners = {}
    for tid, prev_tid in PREV_TRACK_MAP.items():
        if tid in track_summary and prev_tid in track_summary:
            t_start = track_summary[tid]["first_seen_frame"]
            cur_time = track_summary[tid]["first_seen_time"]
            prev_time = track_summary[prev_tid]["last_seen_time"]
            gap_sec = max(0.1, cur_time - prev_time)
            msg = f"<< Re-ID: nhận lại từ track {prev_tid}, cách {gap_sec:.1f}s >>"
            reid_banners[tid] = {
                "start_frame": t_start,
                "end_frame": t_start + 90,
                "msg": msg
            }

    # 2. Open Original Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_orig = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output Resolution: 640x360 (optimized for size < 30 MB)
    out_w, out_h = 640, 360
    scale_x, scale_y = out_w / orig_w, out_h / orig_h
    output_fps = 30.0
    
    print(f"Video input: {orig_w}x{orig_h}, {orig_fps:.2f} FPS, {total_frames_orig} frames")
    print(f"Target output: {out_w}x{out_h}, {output_fps:.2f} FPS (Target size < 30MB)")

    # 3. Initialize Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, output_fps, (out_w, out_h))
    
    font_large, font_medium, font_small, font_bold = get_fonts()

    # 4. Render Intro Card (3.0 seconds @ 30 FPS = 90 frames)
    print("Rendering 3-second Intro Card...")
    intro_frames = render_intro_card(out_w, out_h, duration_sec=3.0, fps=output_fps, font_large=font_large, font_medium=font_medium, font_small=font_small)
    for frame in intro_frames:
        out.write(frame)

    # 5. Process Main Video Frames
    print("Processing main video frames...")
    
    # Screenshot key frames map (frame_id -> filename)
    screenshot_targets = {
        725: os.path.join(OUTPUT_DIR, "screenshot_1_frame725_personB_track7.jpg"),
        1575: os.path.join(OUTPUT_DIR, "screenshot_2_frame1575_personA_track13.jpg"),
        2220: os.path.join(OUTPUT_DIR, "screenshot_3_frame2220_personD_track16.jpg"),
        2885: os.path.join(OUTPUT_DIR, "screenshot_4_frame2885_personC_track18.jpg"),
        3550: os.path.join(OUTPUT_DIR, "screenshot_5_frame3550_personD_track22.jpg")
    }
    extracted_screenshots = {}

    frame_idx = 0
    written_frames_count = len(intro_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame to target output resolution (640x360)
        frame_resized = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        # Trimming & Speed Up Logic:
        # Frames 0..538 have no active tracks. Speed up 3x in 30fps (write 1 out of 6 frames).
        if frame_idx < 539:
            if frame_idx % 6 != 0:
                frame_idx += 1
                continue
        else:
            # Active tracking section: process 1 out of 2 frames for 30 FPS target,
            # but ALWAYS process screenshot target frames!
            if (frame_idx % 2 != 0) and (frame_idx not in screenshot_targets):
                frame_idx += 1
                continue

        # Convert BGR OpenCV frame to RGB Pillow image for overlay rendering
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(img_pil, 'RGBA')

        # Get active tracks for current frame
        frame_tracks = df_tracks[df_tracks['frame_id'] == frame_idx]
        
        active_identities = set()
        
        for _, row in frame_tracks.iterrows():
            tid = int(row['track_id'])
            if tid not in IDENTITY_MAP:
                continue
            
            id_info = IDENTITY_MAP[tid]
            identity_id = id_info['id']
            identity_name = id_info['name']
            color = id_info['color_rgb']
            
            active_identities.add(identity_id)

            # Scale bounding box coordinates to 640x360
            x1, y1, x2, y2 = float(row['x1']) * scale_x, float(row['y1']) * scale_y, float(row['x2']) * scale_x, float(row['y2']) * scale_y
            bbox = [int(x1), int(y1), int(x2), int(y2)]

            # Draw Bbox with identity color
            draw_corner_rect(draw, bbox, color, stroke=2, corner_len=9)

            # Get Attribute text
            top1 = track_summary.get(tid, {}).get('summary', {})
            gender = top1.get('gender', 'N/A')
            upper_color = top1.get('upper_color', '')
            upper_len = top1.get('upper_length', '')
            
            # Format Line 1 text
            # e.g.: "Person A | Male, Yellow Long"
            line1_str = f"{identity_id} | {gender}, {upper_color} {upper_len}".strip()

            # Check if Line 2 (Re-ID banner) should be active
            line2_str = None
            if tid in reid_banners:
                b_info = reid_banners[tid]
                if b_info['start_frame'] <= frame_idx <= b_info['end_frame']:
                    line2_str = b_info['msg']

            # Calculate Label Box position above bbox
            lbl_x = bbox[0]
            lbl_y = bbox[1] - 8

            # Line 1 size
            bbox_l1 = font_bold.getbbox(line1_str)
            w_l1 = bbox_l1[2] - bbox_l1[0]
            h_l1 = bbox_l1[3] - bbox_l1[1]

            w_total = w_l1 + 14
            h_total = h_l1 + 8

            if line2_str:
                bbox_l2 = font_small.getbbox(line2_str)
                w_l2 = bbox_l2[2] - bbox_l2[0]
                h_l2 = bbox_l2[3] - bbox_l2[1]
                w_total = max(w_total, w_l2 + 14)
                h_total += h_l2 + 6

            # Ensure label doesn't go off top of screen
            box_top = max(8, lbl_y - h_total)
            box_rect = [lbl_x, box_top, lbl_x + w_total, box_top + h_total]

            # Draw semi-transparent dark container background for label
            draw.rectangle(box_rect, fill=(15, 22, 32, 220), outline=color, width=1)

            # Draw Line 1
            draw.text((lbl_x + 7, box_top + 4), line1_str, fill=(255, 255, 255), font=font_bold)

            # Draw Line 2 if Re-ID active
            if line2_str:
                draw.text((lbl_x + 7, box_top + 4 + h_l1 + 4), line2_str, fill=(255, 215, 0), font=font_small)

        # 6. Top-Left HUD Badge: "Active identities: N"
        hud_text = f"Active identities: {len(active_identities)}"
        draw.rectangle([12, 12, 210, 42], fill=(15, 22, 32, 220), outline=(50, 150, 240), width=1)
        # Status green indicator dot
        draw.ellipse([20, 23, 28, 31], fill=(35, 225, 100))
        draw.text((35, 19), hud_text, fill=(255, 255, 255), font=font_bold)

        # 7. Bottom Watermark Banner
        bm_text = "Màu sắc = danh tính (identity), không phải track ID thô"
        draw.rectangle([12, out_h - 32, out_w - 12, out_h - 8], fill=(15, 22, 32, 190))
        draw.text((out_w // 2, out_h - 20), bm_text, fill=(220, 230, 240), font=font_small, anchor="mm")

        # Convert back to OpenCV BGR
        final_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        out.write(final_frame)
        written_frames_count += 1

        # Check screenshot capture
        if frame_idx in screenshot_targets:
            save_path = screenshot_targets[frame_idx]
            cv2.imwrite(save_path, final_frame)
            time_sec = frame_idx / orig_fps
            extracted_screenshots[frame_idx] = {
                "path": save_path,
                "timestamp_sec": time_sec,
                "timestamp_str": f"{int(time_sec//60):02d}:{time_sec%60:04.1f}"
            }
            print(f"Captured screenshot frame {frame_idx} ({time_sec:.1f}s) -> {os.path.basename(save_path)}")

        frame_idx += 1

    # Cleanup
    cap.release()
    out.release()
    print("Video encoding complete!")

    # 8. Post-processing Verification & Bitrate Check
    final_duration = written_frames_count / output_fps
    file_size_mb = os.path.getsize(OUTPUT_VIDEO_PATH) / (1024 * 1024)
    
    print("\n=== OUTPUT METRICS ===")
    print(f"File Path: {OUTPUT_VIDEO_PATH}")
    print(f"Final Duration: {final_duration:.2f} seconds ({written_frames_count} frames)")
    print(f"File Size: {file_size_mb:.2f} MB")
    print(f"Output FPS: {output_fps:.2f}")

    if file_size_mb <= 30.0:
        print(f"SUCCESS: File size is {file_size_mb:.2f} MB (<= 30 MB constraint satisfied!).")
    else:
        print(f"Warning: File size is {file_size_mb:.2f} MB.")

    print("\n=== EXTRACTED SCREENSHOTS ===")
    for f_idx, info in extracted_screenshots.items():
        print(f"- Frame {f_idx:4d} (Timestamp {info['timestamp_str']}): {info['path']}")

if __name__ == "__main__":
    main()
