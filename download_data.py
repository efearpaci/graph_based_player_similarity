"""
Step 1: Download the Wyscout/Pappalardo Soccer Match Event Dataset.

Source: Pappalardo et al. (2019), "A public data set of spatio-temporal match
events in soccer competitions", Scientific Data.
Figshare collection: https://figshare.com/collections/Soccer_match_event_dataset/4415000

Downloads into data/raw/ :
    events/events_<Country>.json   (per-competition event streams)
    matches/matches_<Country>.json
    players.json, teams.json, competitions.json
    tags2name.csv, eventid2name.csv

Run once:  python download_data.py
"""

import io
import zipfile
from pathlib import Path

import requests

RAW_DIR = Path(__file__).parent / "data" / "raw"

FILES = {
    "events.zip": "https://ndownloader.figshare.com/files/14464685",
    "matches.zip": "https://ndownloader.figshare.com/files/14464622",
    "players.json": "https://ndownloader.figshare.com/files/15073721",
    "teams.json": "https://ndownloader.figshare.com/files/15073697",
    "competitions.json": "https://ndownloader.figshare.com/files/15073685",
    "tags2name.csv": "https://ndownloader.figshare.com/files/21385239",
    "eventid2name.csv": "https://ndownloader.figshare.com/files/21385245",
}


def download(name, url):
    dest = RAW_DIR / name
    if name.endswith(".zip"):
        out_dir = RAW_DIR / name.replace(".zip", "")
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"✓ {name} already extracted -> {out_dir}")
            return
        print(f"↓ downloading {name} …")
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        print(f"  extracting to {out_dir}/")
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            zf.extractall(out_dir)
    else:
        if dest.exists():
            print(f"✓ {name} already downloaded")
            return
        print(f"↓ downloading {name} …")
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        dest.write_bytes(r.content)


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        download(name, url)
    print("\nDone. Contents of data/raw:")
    for p in sorted(RAW_DIR.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(RAW_DIR)}  ({p.stat().st_size / 1e6:.1f} MB)")
