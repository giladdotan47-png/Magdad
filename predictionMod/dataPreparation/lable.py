import sys
sys.path.append(r"/")   # התיקייה שמכילה את bottle_detection_api
from bottle_detection_api.bottle_detection import *
import csv
from pathlib import Path
import cv2

VIDEOS_DIR = Path(r"/predictionMod/seperated")
OUTPUT_CSV = "Labled_Data.csv"

FRAME_RATE = 120  # TODO: set manually

FIELDS = ["xcap", "ycap", "zcap", "xbottle", "ybottle", "zbottle", "frameRate"]


def frame_values(frame):
    """Run the detection on one frame and return the 7 values for the CSV."""
    result = find_bottle(frame)
    # result = {'found': bool, 'bottle_center': (x, z), 'cap_center': (x, z), 'sure': ...}
    if not result or not result["found"]:  # nothing detected -> empty cells (frameRate stays)
        print("ahhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
        return [0, 0, 0, 0, 0, 0, FRAME_RATE]

    xcap, zcap = result["cap_center"]
    xbottle, zbottle = result["bottle_center"]
    return [xcap, 0, zcap, xbottle, 0, zbottle, FRAME_RATE]  # y = 0


rows = []
for video in sorted(VIDEOS_DIR.iterdir()):
    if not video.is_file():
        continue

    row = [video.name]

    cap = cv2.VideoCapture(str(video))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        row.extend(frame_values(frame))   # every frame adds one set of columns
    cap.release()

    rows.append(row)
    print(f"{video.name}: done")

# Header: video_name, then the field set repeated per frame
max_frames = max((len(r) - 1) // len(FIELDS) for r in rows)
header = ["video_name"] + [f"{name}_{i}" for i in range(1, max_frames + 1) for name in FIELDS]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)