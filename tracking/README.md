# Module Video Tracking, Attribute Aggregation, Re-ID & Hybrid Matching

Module này mở rộng hệ thống nhận diện thuộc tính người đi bộ UPAR từ ảnh tĩnh thành **Pipeline Phân tích Video Người đi bộ Toàn diện (4 Level)**, kết hợp giữa Tracking ngắn hạn, Gom nhóm Thuộc tính, Re-Identification (OSNet) và Hybrid Matching.

> **Báo cáo Kỹ thuật & Thực nghiệm Chuyên sâu**: Xem chi tiết phương pháp luận, quá trình sửa lỗi pseudo-replication và kết quả đánh giá LOOCV tại [`tracking/TECHNICAL_REPORT.md`](file:///c:/Users/ADMIN/OneDrive/Documents/GitHub/AI-Project/tracking/TECHNICAL_REPORT.md).

---

## 1. Cài đặt Thư viện Dependencies

```powershell
pip install ultralytics opencv-python torchreid networkx pandas
```

* `ultralytics`: Tích hợp sẵn YOLOv8 và ByteTrack/BoT-SORT (`bytetrack.yaml`).
* `torchreid`: Trích xuất đặc trưng Re-ID (OSNet `osnet_x1_0`).
* `networkx`: Xây dựng đồ thị liên thông gom nhóm Identity đối tượng.

---

## 2. Hướng dẫn Vận hành theo 4 Level

### Level 1 — Detection + Tracking (`track.py`)
Phát hiện người đi bộ và duy trì `track_id` ngắn hạn trên video stream:

```powershell
# Tracking từ file video:
python tracking/track.py --source tracking/test_videos/real_pedestrians.mp4 --save-video

# Tracking từ webcam:
python tracking/track.py --source 0 --show
```
* **Output**: `reports/tracking/<ten_video>/tracks.csv` và `reports/tracking/<ten_video>/tracked.mp4`.
* **Model weights**: Tự động lưu/tải tại `checkpoints/yolov8n.pt`.

---

### Level 2 — Trích Crop & Gom Thuộc tính (`extract_crops.py` & `track_attributes.py`)

#### Bước 2.1 — Trích Crop ảnh người đi bộ theo Track ID:
```powershell
python tracking/extract_crops.py \
  --video tracking/test_videos/real_pedestrians.mp4 \
  --csv reports/tracking/real_pedestrians/tracks.csv \
  --output-dir reports/tracking/crops/real_pedestrians \
  --every-n-frames 5 \
  --clean
```
* **Output**: Ảnh crop người theo cấu trúc `reports/tracking/crops/<ten_video>/track_<id>/frame_<n>.jpg`.

#### Bước 2.2 — Gán & Gom nhóm Thuộc tính UPAR qua thời gian:
```powershell
python tracking/track_attributes.py \
  --crops-dir reports/tracking/crops/real_pedestrians \
  --tracks-csv reports/tracking/real_pedestrians/tracks.csv \
  --checkpoint checkpoints/hydraplus_upar_best.pth \
  --output-dir reports/tracking/real_pedestrians \
  --min-frames 3
```
* **Output**: 
  - `reports/tracking/<ten_video>/attributes.csv`: Bảng thuộc tính Top-1 cho mỗi `track_id`.
  - `reports/tracking/<ten_video>/attributes.json`: Chi tiết multi-label active và 40 xác suất raw.
  - `reports/tracking/tracked_persons_summary.csv`: Bảng tổng hợp đối tượng (Track metadata + Attributes).

---

### Level 3 — Re-ID Feature Embedding Extractor & Validation

#### Bước 3.1 — Trích 512-dim Embedding & Validate Market1501 Benchmark (`reid_embedding.py`):
```powershell
python tracking/reid_embedding.py
```
* Đánh giá phân phối Cosine Similarity Intra-ID vs Inter-ID trên dataset Market1501 (Separation Margin: **+34.72%**).
* Tự động tính mean-pooled embedding 512 chiều cho từng `track_id` và cập nhật vào `reports/tracking/<ten_video>/attributes.json`.

#### Bước 3.2 — Đánh giá Domain Gap trên Video Thực (`reid_validate_domain.py`):
```powershell
python tracking/reid_validate_domain.py
```
* Đánh giá phân phối Cosine Similarity ở cấp độ frame-level trên video thực `real_pedestrians.mp4`.

#### Bước 3.3 — Đánh giá Re-entry & Chống Pseudo-Replication (`reid_validate_reentry.py`):
```powershell
python tracking/reid_validate_reentry.py \
  --crops-dir reports/tracking/crops/store_aisle \
  --gt-csv reports/tracking/store-aisle-detection/reentry_ground_truth.csv
```
* Đánh giá EER bài toán Re-entry (người bị occlusion/rời khung hình) kết hợp gom nhóm Identity Độc lập & Bootstrap 1,000 lần.

#### Bước 3.4 — Benchmark Multi-Video Domain (`reid_validate_reentry_combined.py`):
```powershell
python tracking/reid_validate_reentry_combined.py
```
* Đánh giá Re-ID trên 4 video benchmark ($N=12$ sự kiện re-entry độc lập), báo cáo song song **Micro EER (9.67%)** và **Macro EER (13.70%)**.

---

### Level 4 — Hybrid Matching & Cross-Validation (`hybrid_matching.py`)
Kết hợp Re-ID Similarity + Attribute Similarity + Time Penalty theo công thức Hybrid Score:

$$S_{\text{hybrid}} = w_1 S_{\text{reid}} + w_2 S_{\text{attr}} - w_3 P_{\text{time}}$$

```powershell
python tracking/hybrid_matching.py
```
* Sử dụng **Leave-One-Out Cross-Validation (LOOCV)** ở cấp độ video để tìm trọng số tối ưu tránh Data Leakage.
* **Kết quả**: Re-ID 512 chiều đóng vai trò cốt lõi ($w_1=0.80$). Thuộc tính UPAR không làm tăng Re-entry EER do nhiễu nền CCTV, nhưng dùng rất tốt cho bài toán **Person Retrieval / Attribute Querying**.

---

### Demo Combined Video Generation (`demo_combined.py`)
Tạo video demo duy nhất kết hợp trực quan Level 1 (Tracking) + Level 2 (Attribute) + Level 3 (Re-ID):

```powershell
python tracking/demo_combined.py
```
* **Output**:
  - `reports/tracking/demo/demo_combined_tracking_attribute_reid.mp4` (Video demo hợp nhất 62s, 30 FPS, < 30 MB)
  - `reports/tracking/demo/screenshot_*.jpg` (5 ảnh chụp màn hình minh chứng các khoảnh khắc Re-ID)

---

## 3. Cấu trúc File Module `tracking/`

```text
tracking/
├── __init__.py
├── README.md                               # Hướng dẫn sử dụng module tracking
├── TECHNICAL_REPORT.md                     # Báo cáo kỹ thuật & kết quả nghiên cứu 4 Level
├── track.py                                # Level 1: YOLOv8 + ByteTrack tracking pipeline
├── extract_crops.py                        # Level 2: Trích crop ảnh người theo track_id
├── track_attributes.py                     # Level 2: Gom nhóm xác suất 40 thuộc tính UPAR
├── reid_embedding.py                       # Level 3: Trích xuất OSNet 512-dim embedding
├── reid_validate_domain.py                 # Level 3: Validate similarity trên real video
├── reid_validate_reentry.py                # Level 3: Validate Re-entry & Chống Pseudo-replication
├── reid_validate_reentry_combined.py       # Level 3: Benchmark Re-ID trên 4 video
├── hybrid_matching.py                      # Level 4: Hybrid Score & LOOCV Grid Search
├── demo_combined.py                        # Video Demo Hợp nhất Level 1 + Level 2 + Level 3
└── test_videos/                            # Thư mục lưu trữ video thử nghiệm mẫu (.mp4, .avi)
```
