"""Scratch script: Queries Intel IoT sample video repository for pedestrian test videos."""
import urllib.request
import json

url = 'https://api.github.com/repos/intel-iot-devkit/sample-videos/contents/'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as resp:
        files = json.loads(resp.read().decode('utf-8'))
        for f in files:
            if f['name'].endswith('.mp4'):
                print(f"{f['name']:<45} {f['size']/(1024*1024):.2f} MB")
except Exception as e:
    print('Error:', e)
