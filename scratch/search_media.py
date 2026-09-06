"""Scratch script: Checks availability and file sizes of candidate public video sample URLs."""
import urllib.request
import json

# Check candidate URLs on github/huggingface
candidates = [
    "https://huggingface.co/datasets/opencv/video-test-data/resolve/main/vtest.avi",
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi",
    "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/driver-action-recognition.mp4",
    "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/one-by-one-person-detection.mp4",
    "https://github.com/intel-iot-devkit/sample-videos/raw/master/store-aisle-detection.mp4"
]

for c in candidates:
    name = c.split('/')[-1]
    try:
        req = urllib.request.Request(c, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            print(f"[FOUND] {name} - {resp.headers.get('Content-Length', 'Unknown')} bytes")
    except Exception as e:
        print(f"[FAILED] {name}: {e}")
