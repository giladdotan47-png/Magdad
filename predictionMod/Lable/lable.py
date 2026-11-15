"""
One script that does everything:

  long video(s) -> run the model once per frame -> find the bottle-flip attempts
  (cut the hand-occluded start and the static/lost end) -> write a CSV where
  every row is one attempt.

Row layout:
  video_name, then for every frame of the attempt:
  xcap, ycap, zcap, xbottle, ybottle, zbottle, frameRate     (y = 0)

Detections are cached next to each video, so re-tuning the parameters
below doesn't re-run the model. Delete the *.flip_cache.npz file to force a re-run.
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(r"C:\Users\TLP\.vscode\Magdad")
from bottle_detection_api.bottle_detection import find_bottle

# ---------------- paths ----------------
LONG_VIDEOS_DIR = Path(r"C:\Users\TLP\.vscode\Magdad\predictionMod\long_vids_120fps")  # TODO
OUTPUT_CSV = Path("Labled_Data.csv")

FRAME_RATE = 120          # TODO: set manually

# Optional: also save every attempt as a video clip, to visually check the cuts
SAVE_CLIPS = True
CLIPS_DIR = Path(r"C:\Users\TLP\.vscode\Magdad\predictionMod\Lable\clipped_vids_120fps")

# ---------------- parameters to tune ----------------
USE_SURE = True          # also require result["sure"] to be truthy for a frame to count as "visible"
GAP_TOL = 12             # frames of missed detection tolerated inside one attempt
MOVE_WINDOW = 5          # compare the bottle position to N frames ago
MOVE_PX = 8              # displacement (pixels) over MOVE_WINDOW frames that counts as "moving"
MIN_MOVING_FRAMES = 10   # an attempt needs at least this many moving frames
PAD_START = 5            # extra frames kept before the first movement
PAD_END = 10             # extra frames kept after the last movement

FIELDS = ["xcap", "ycap", "zcap", "xbottle", "ybottle", "zbottle", "frameRate"]


# ---------------- pass 1: run the model on every frame ----------------
def to_xy(v):
    return (np.nan, np.nan) if v is None else (float(v[0]), float(v[1]))


def detect_all(video):
    cache = video.with_suffix(".flip_cache.npz")
    if cache.exists():
        d = np.load(cache)
        print(f"  loaded cached detections: {cache.name}")
        return d["found"], d["sure"], d["bottle"], d["cap"]

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    found, sure, bottle, capc = [], [], [], []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = find_bottle(frame)
        if r and r["found"]:
            found.append(True)
            sure.append(bool(r["sure"]))
            bottle.append(to_xy(r["bottle_center"]))
            capc.append(to_xy(r["cap_center"]))
        else:
            found.append(False)
            sure.append(False)
            bottle.append((np.nan, np.nan))
            capc.append((np.nan, np.nan))

        if len(found) % 1000 == 0:
            print(f"  {len(found)}/{total} frames")
    cap.release()

    found, sure = np.array(found), np.array(sure)
    bottle, capc = np.array(bottle, dtype=float), np.array(capc, dtype=float)
    np.savez(cache, found=found, sure=sure, bottle=bottle, cap=capc)
    return found, sure, bottle, capc


# ---------------- pass 2: find the attempts ----------------
def valid_runs(valid):
    """Groups of visible frames, tolerating short detection dropouts."""
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev - 1 > GAP_TOL:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return runs


def trim_to_motion(pos, valid, s, e):
    """Cut the static start/end of a run. Returns None if the bottle never really moved."""
    moving = []
    for i in range(s + MOVE_WINDOW, e + 1):
        if valid[i] and valid[i - MOVE_WINDOW]:
            if np.hypot(*(pos[i] - pos[i - MOVE_WINDOW])) > MOVE_PX:
                moving.append(i)
    if len(moving) < MIN_MOVING_FRAMES:
        return None
    return max(s, moving[0] - MOVE_WINDOW - PAD_START), min(e, moving[-1] + PAD_END)


def find_attempts(valid, bottle):
    segments = []
    for s, e in valid_runs(valid):
        seg = trim_to_motion(bottle, valid, s, e)
        if seg is None:
            continue
        if segments and seg[0] <= segments[-1][1]:   # padding must not overlap the previous clip
            seg = (segments[-1][1] + 1, seg[1])
        segments.append(seg)
    return segments


# ---------------- build a CSV row for one attempt ----------------
def fmt(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else v


def attempt_row(name, s, e, valid, bottle, capc):
    row = [name]
    for i in range(s, e + 1):
        if valid[i]:
            xcap, zcap = capc[i]
            xbottle, zbottle = bottle[i]
            vals = [xcap, 0, zcap, xbottle, 0, zbottle, FRAME_RATE]    # y = 0
        else:
            vals = [""] * (len(FIELDS) - 1) + [FRAME_RATE]             # not visible; frameRate stays
        row.extend(fmt(v) for v in vals)
    return row


# ---------------- optional: save the clips for a visual check ----------------
def write_clips(video, segments):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    writer, k, i = None, 0, 0
    while k < len(segments):
        ok, frame = cap.read()
        if not ok:
            break
        s, e = segments[k]
        if i == s:
            writer = cv2.VideoWriter(str(CLIPS_DIR / f"{video.stem}_{k + 1:03d}.mp4"), fourcc, fps, (w, h))
        if s <= i <= e:
            writer.write(frame)
        if i == e:
            writer.release()
            writer = None
            k += 1
        i += 1
    cap.release()
    if writer is not None:
        writer.release()


def main():
    videos = sorted(
        p for p in LONG_VIDEOS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}
    )
    rows, segment_log = [], []

    for video in videos:
        print(f"{video.name}")
        found, sure, bottle, capc = detect_all(video)
        valid = (found & sure) if USE_SURE else found
        segments = find_attempts(valid, bottle)
        print(f"  {len(segments)} attempts")

        for k, (s, e) in enumerate(segments, start=1):
            name = f"{video.stem}_{k:03d}"
            rows.append(attempt_row(name, s, e, valid, bottle, capc))
            segment_log.append([name, video.name, s, e, e - s + 1])

        if SAVE_CLIPS:
            write_clips(video, segments)

    if not rows:
        sys.exit("No attempts found - try lowering MOVE_PX or setting USE_SURE = False")

    max_frames = max((len(r) - 1) // len(FIELDS) for r in rows)
    header = ["video_name"] + [
        f"{name}_{i}" for i in range(1, max_frames + 1) for name in FIELDS
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    # which frames of which long video each row came from
    with open(OUTPUT_CSV.with_name(OUTPUT_CSV.stem + "_segments.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["row_name", "source_video", "start_frame", "end_frame", "num_frames"])
        writer.writerows(segment_log)

    print(f"\nSaved {len(rows)} attempts to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()