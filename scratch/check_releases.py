"""Scratch script: Queries GitHub releases for open-source video tracking sample datasets."""
import urllib.request
import json

repos = [
    'roboflow/supervision',
    'ifzhang/ByteTrack',
    'marten-krusemark/tracking',
    'open-mmlab/mmtracking'
]

for r in repos:
    url = f'https://api.github.com/repos/{r}/releases'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as resp:
            releases = json.loads(resp.read().decode('utf-8'))
            print(f'=== {r} ({len(releases)} releases) ===')
            for rel in releases:
                for asset in rel.get('assets', []):
                    if asset['name'].endswith(('.mp4', '.avi', '.mkv', '.zip')):
                        print(f"  - {asset['name']} ({asset['size']/(1024*1024):.2f} MB): {asset['browser_download_url']}")
    except Exception as e:
        print(f'Error querying {r}: {e}')
