"""
One script that does everything:

  long video(s) -> run the model once per frame -> find the bottle-flip attempts
  (cut the hand-occluded start and the static/lost end) -> write a CSV where
  every row is one attempt.

Row layout (one file, the label comes right after the name):
  video_name, z_pred, frame_number, then for every frame of the attempt:
  xcap, ycap, zcap, xbottle, ybottle, zbottle, duration      (y = 0; duration = 1 / frame rate, in seconds)

Normalization: x and z are min-max scaled to 0-1 separately for every row, using the min/max
of that row's input frames (the same frames that are written to the row). z_pred is scaled
with the same z range.

Attempts where no label frame is found (or nothing usable is left before it) are dropped,
so there is no row for them.

Frames inside a good attempt where the bottle was not detected are written as -1 in x and z
(y stays 0, and the duration is kept).

Every data row ends with a single cell (END_MARKER) so the reader knows where the row stops.
The file has no header row (WRITE_HEADER = False), as the data convention asks.

Detections are cached next to each video, so re-tuning the parameters
below doesn't re-run the model. Delete the *.flip_cache.npz file to force a re-run.
"""

import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append("/content/drive/MyDrive/Magdad")
from bottle_detection_api.bottle_detection import find_bottle

# ---------------- paths (Google Colab) ----------------
LONG_VIDEOS_DIR = Path("/content/drive/MyDrive/Magdad/long_vids_GoPro")
OUTPUT_CSV = Path("/content/drive/MyDrive/Magdad/GoPro_Labled_Data.csv")

# Decode the video on the GPU (NVDEC) through ffmpeg instead of cv2.VideoCapture.
# HEVC 10-bit 1080x1920@120fps is very slow to decode on Colab's single CPU core.
USE_FFMPEG_GPU_DECODE = False

# Save progress to Drive every N frames, so a Colab disconnect doesn't lose everything.
# On the next run the script resumes from the last checkpoint.
CHECKPOINT_EVERY = 5000

# When resuming, jump straight to the checkpoint frame instead of decoding all the earlier frames
# (decoding is ~100 ms/frame here, so skipping 75,000 frames the slow way takes about two hours).
# After the jump, the last VERIFY_FRAMES frames before the checkpoint are re-detected and compared
# with the saved detections; if they don't match, it falls back to the slow, exact skip.
FAST_SEEK = True
VERIFY_FRAMES = 20

FRAME_RATE = 120          # TODO: set manually
FRAME_DURATION = 1 / FRAME_RATE   # seconds per frame; this is what gets written to the CSV

# ---------------- labels ----------------
# The label frame = the first frame where the cap and the bottle center have the same x while the cap
# is ABOVE the bottle center, after the bottle has been upside down. A flip passes "same x" twice:
# upside down (cap below the center, middle of the flip) and upright again (cap above the center, right
# before the landing). Only the second one is the label.
# MIN_OFFSET_PX: the cap must first be this many pixels away from the bottle center along x before a
# "crossing" counts, and this far below the center to count as upside down.
MIN_OFFSET_PX = 15

# Which sign of (z cap - z bottle) means "cap above the bottle center", i.e. the bottle stands upright:
#   None = infer it for each video (the bottle stands upright most of the time, so the median tells),
#   +1   = the cap has the larger z when upright (z grows upwards),
#   -1   = the cap has the smaller z when upright (z grows downwards, like image rows).
UPRIGHT_SIGN = None

# Normalization scale. True  = one min/max per axis, shared by cap and bottle (x: xcap+xbottle,
#                              z: zcap+zbottle), which keeps the cap-vs-bottle geometry intact.
#                    False = a separate min/max for each of the 4 columns.
SHARED_AXIS_SCALE = True

# One cell written after the last frame of every data row, so the reader knows where to cut the row.
END_MARKER = 2

# The data convention says the CSV files must not contain column names.
WRITE_HEADER = False

# How many attempts to save as short video clips, per long video (the first N), to see what was cut.
# 0 = don't save any clips.
NUM_CLIPS_TO_SAVE = 10
CLIPS_DIR = Path("/content/drive/MyDrive/Magdad/clipped_GoPro_vids_")

# ---------------- parameters to tune ----------------
USE_SURE = True          # also require result["sure"] to be truthy for a frame to count as "visible"
GAP_TOL = 12             # frames of missed detection tolerated inside one attempt
MOVE_WINDOW = 5          # compare the bottle position to N frames ago
MOVE_PX = 8              # displacement (pixels) over MOVE_WINDOW frames that counts as "moving"
MIN_MOVING_FRAMES = 10   # an attempt needs at least this many moving frames
PAD_START = 5            # extra frames kept before the first movement
PAD_END = 10             # extra frames kept after the last movement

FIELDS = ["xcap", "ycap", "zcap", "xbottle", "ybottle", "zbottle", "duration"]


# ---------------- frame reading ----------------
def frames_from_cap(cap):
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def read_frames_cv2(video, skip=0):
    """Slow but exact: decodes every frame up to `skip` and throws it away."""
    cap = cv2.VideoCapture(str(video))
    for i in range(skip):
        if not cap.grab():
            break
        if i % 5000 == 0 and i:
            print(f"  skipping... {i}/{skip}")
    return frames_from_cap(cap)


def open_at(video, skip, found, bottle):
    """
    Jump to frame `skip` and check the jump was frame-accurate: re-run the model on the
    VERIFY_FRAMES frames before it and compare with the detections saved in the checkpoint.
    Returns an open VideoCapture positioned at `skip`, or None if the check fails.
    """
    cap = cv2.VideoCapture(str(video))
    start = max(0, skip - VERIFY_FRAMES)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != start:
        cap.release()
        return None

    dists, flag_mismatch = [], 0
    for i in range(start, skip):
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return None
        r = find_bottle(frame)
        det = bool(r and r["found"])
        if det != bool(found[i]):
            flag_mismatch += 1
        elif det:
            dists.append(float(np.hypot(*(np.array(to_xy(r["bottle_center"])) - np.array(bottle[i])))))

    if flag_mismatch > (skip - start) // 5 or (dists and float(np.median(dists)) > 3):
        cap.release()
        return None
    if not dists:
        print("  note: no detections in the verification window, so the jump could not be verified")
    return cap


def read_frames_ffmpeg(video):
    """Yield BGR frames decoded by ffmpeg with GPU (NVDEC) decoding, piped as raw video."""
    # Frame size and the reference first frame, exactly as cv2 would deliver them
    cap = cv2.VideoCapture(str(video))
    ok, ref = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read {video}")
    h, w = ref.shape[:2]
    size = w * h * 3

    cmd = ["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-i", str(video),
           "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=size * 4)
    try:
        first = True
        while True:
            frame = np.empty((h, w, 3), np.uint8)
            if proc.stdout.readinto(frame) < size:
                break
            if first:
                # sanity check: ffmpeg's first frame must match cv2's (same size/rotation/colors)
                diff = np.abs(frame.astype(np.int16) - ref.astype(np.int16)).mean()
                if diff > 15:
                    raise RuntimeError(
                        f"ffmpeg frames differ from cv2 frames (mean diff {diff:.1f}) - "
                        "rotation or color mismatch. Set USE_FFMPEG_GPU_DECODE = False."
                    )
                first = False
            yield frame
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def read_frames(video, skip=0):
    if USE_FFMPEG_GPU_DECODE:
        frames = read_frames_ffmpeg(video)
        for _ in range(skip):
            next(frames, None)
        return frames
    return read_frames_cv2(video, skip)


# ---------------- pass 1: run the model on every frame ----------------
def to_xy(v):
    return (np.nan, np.nan) if v is None else (float(v[0]), float(v[1]))


def detect_all(video):
    cache = video.with_suffix(".flip_cache.npz")
    if cache.exists():
        d = np.load(cache)
        print(f"  loaded cached detections: {cache.name}")
        return d["found"], d["sure"], d["bottle"], d["cap"]

    partial = video.with_suffix(".flip_partial.npz")
    partial_tmp = video.with_suffix(".flip_partial_tmp.npz")

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    found, sure, bottle, capc = [], [], [], []

    # resume from the last checkpoint, if there is one
    skip = 0
    if partial.exists():
        d = np.load(partial)
        found, sure = d["found"].tolist(), d["sure"].tolist()
        bottle = [tuple(x) for x in d["bottle"].tolist()]
        capc = [tuple(x) for x in d["cap"].tolist()]
        skip = len(found)
        print(f"  resuming from checkpoint: {skip}/{total} frames already done")

    frames = None
    if skip and FAST_SEEK and not USE_FFMPEG_GPU_DECODE:
        cap = open_at(video, skip, found, bottle)
        if cap is not None:
            print(f"  jumped to frame {skip} (verified against the saved detections)")
            frames = frames_from_cap(cap)
        else:
            print("  fast jump failed verification -> falling back to the slow exact skip")
    if frames is None:
        frames = read_frames(video, skip)

    for frame in frames:
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

        if len(found) % CHECKPOINT_EVERY == 0:
            np.savez(partial_tmp, found=np.array(found), sure=np.array(sure),
                     bottle=np.array(bottle, dtype=float), cap=np.array(capc, dtype=float))
            partial_tmp.replace(partial)   # replace only after the write finished

    found, sure = np.array(found), np.array(sure)
    bottle, capc = np.array(bottle, dtype=float), np.array(capc, dtype=float)
    np.savez(cache, found=found, sure=sure, bottle=bottle, cap=capc)
    partial.unlink(missing_ok=True)
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
def input_ranges(valid, bottle, capc, s, k):
    """
    (min, max) for x and z of the cap and the bottle, over the visible input frames s .. k-1 only
    (the frames written to the row, so nothing from the label frame or later leaks into the scale).
    Returns None if there is no usable frame.
    """
    idx = [i for i in range(s, k) if valid[i]]
    if not idx:
        return None
    cap_v, bot_v = capc[idx], bottle[idx]            # columns: 0 = x, 1 = z
    ranges = {}
    for ax, axis in ((0, "x"), (1, "z")):
        if SHARED_AXIS_SCALE:
            both = np.concatenate([cap_v[:, ax], bot_v[:, ax]])
            ranges[("cap", axis)] = ranges[("bottle", axis)] = (np.nanmin(both), np.nanmax(both))
        else:
            ranges[("cap", axis)] = (np.nanmin(cap_v[:, ax]), np.nanmax(cap_v[:, ax]))
            ranges[("bottle", axis)] = (np.nanmin(bot_v[:, ax]), np.nanmax(bot_v[:, ax]))
    if any(np.isnan(lo) or np.isnan(hi) for lo, hi in ranges.values()):
        return None
    return ranges


def norm(v, rng):
    """Min-max scale to 0-1. Undetected values become -1; a constant axis (max == min) gives 0."""
    lo, hi = rng
    if np.isnan(v):
        return -1
    if hi == lo:
        return 0.0
    return round(float((v - lo) / (hi - lo)), 6)


def infer_upright_sign(valid, bottle, capc):
    """+1 if the cap has the larger z when the bottle stands upright, -1 otherwise."""
    if UPRIGHT_SIGN is not None:
        return UPRIGHT_SIGN
    dz = capc[valid, 1] - bottle[valid, 1]
    dz = dz[~np.isnan(dz)]
    # the bottle stands upright most of the time (an attempt is only ~1-2 s out of every ~5 s)
    return 1 if dz.size == 0 or np.median(dz) >= 0 else -1


def find_label_frame(valid, bottle, capc, s, e, upright):
    """
    The label frame: the first frame in [s, e] where the cap and the bottle center have the same x
    (the bottle is vertical) while the cap is above the bottle center, after the bottle has been
    upside down. The same-x moment while upside down (cap below the center) is skipped.
    `upright` is +1 if the cap has the larger z when upright, else -1 (see infer_upright_sign).
    Returns None if it never happens (e.g. the bottle fell over).
    """
    armed, inverted, prev = False, False, None
    for i in range(s, e + 1):
        if not valid[i] or np.isnan(capc[i]).any() or np.isnan(bottle[i]).any():
            continue
        d = capc[i][0] - bottle[i][0]                      # x: 0 when the bottle is vertical
        up = (capc[i][1] - bottle[i][1]) * upright         # > 0: the cap is above the bottle center
        if up < -MIN_OFFSET_PX:
            inverted = True
        if not armed:
            if abs(d) >= MIN_OFFSET_PX:
                armed, prev = True, (i, d, up)
            continue
        pi, pd, pu = prev
        if d == 0 or (pd < 0) != (d < 0):
            j, ju = (i, up) if abs(d) <= abs(pd) else (pi, pu)   # the frame closest to alignment
            if inverted and ju > 0:
                return j
        prev = (i, d, up)
    return None


def attempt_row(name, z_pred, frame_number, s, k, valid, bottle, capc, ranges):
    """
    One row: the name, the label (z_pred, frame_number), then the input frames s .. k-1 only
    (never the label frame k itself or later), normalized, followed by one END_MARKER cell.
    """
    row = [name, z_pred, frame_number]
    for i in range(s, k):
        if valid[i]:
            xcap, zcap = capc[i]
            xbottle, zbottle = bottle[i]
            row.extend([
                norm(xcap, ranges[("cap", "x")]), 0, norm(zcap, ranges[("cap", "z")]),
                norm(xbottle, ranges[("bottle", "x")]), 0, norm(zbottle, ranges[("bottle", "z")]),
                FRAME_DURATION,                                    # y = 0
            ])
        else:
            row.extend([-1, 0, -1, -1, 0, -1, FRAME_DURATION])        # not visible: -1 in x and z, y = 0; duration stays
    row.append(END_MARKER)
    return row


# ---------------- optional: save a plot to check where the label frame was put ----------------
def save_signal_plot(video, n, s, e, k, valid, bottle, capc, upright):
    """
    PNG of (x cap - x bottle) and (z cap - z bottle) over one attempt, with the label frame marked.
    Saved for the same first NUM_CLIPS_TO_SAVE attempts as the clips. Skipped if matplotlib is missing.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    idx = np.arange(s, e + 1)
    ok = valid[idx]
    dx = np.where(ok, capc[idx, 0] - bottle[idx, 0], np.nan)
    dz = np.where(ok, (capc[idx, 1] - bottle[idx, 1]) * upright, np.nan)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(idx, dx, label="x cap - x bottle  (0 = bottle vertical)")
    ax.plot(idx, dz, label="z cap - z bottle  (positive = cap above the center)")
    ax.axhline(0, color="gray", linewidth=0.8)
    if k is not None:
        ax.axvline(k, color="red", linestyle="--", label=f"label frame ({k})")
    ax.set_xlabel("frame")
    ax.set_title(f"{video.stem}_{n:03d}" + ("" if k is not None else "  - no label frame found"))
    ax.legend(fontsize=8)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CLIPS_DIR / f"{video.stem}_{n:03d}_signals.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------- optional: save the clips for a visual check ----------------
def write_clips(video, segments, max_clips=NUM_CLIPS_TO_SAVE):
    """Save the first `max_clips` attempts of `video` as short clips (named like the CSV rows)."""
    segments = segments[:max_clips]
    if not segments:
        return
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
    n_dropped = n_out_of_range = 0

    for video in videos:
        print(f"{video.name}")
        found, sure, bottle, capc = detect_all(video)
        valid = (found & sure) if USE_SURE else found
        segments = find_attempts(valid, bottle)
        upright = infer_upright_sign(valid, bottle, capc)
        print(f"  {len(segments)} attempts (upright bottle = cap has the "
              f"{'larger' if upright > 0 else 'smaller'} z)")

        dropped_here = 0
        for n, (s, e) in enumerate(segments, start=1):
            name = f"{video.stem}_{n:03d}"
            k = find_label_frame(valid, bottle, capc, s, e, upright)
            if n <= NUM_CLIPS_TO_SAVE:
                save_signal_plot(video, n, s, e, k, valid, bottle, capc, upright)
            ranges = input_ranges(valid, bottle, capc, s, k) if (k is not None and k > s) else None

            if ranges is None:   # no label frame / no usable input frames -> drop the attempt
                dropped_here += 1
                continue

            z_pred = norm(bottle[k][1], ranges[("bottle", "z")])   # z of the bottle at the label frame
            if not 0 <= z_pred <= 1:
                n_out_of_range += 1
            frame_number = k - s + 1                      # 1-based, counted from the start of the attempt

            rows.append(attempt_row(name, z_pred, frame_number, s, k, valid, bottle, capc, ranges))
            segment_log.append([name, video.name, s, e, k])
        if dropped_here:
            print(f"  {dropped_here} attempts dropped (no label frame found)")
        n_dropped += dropped_here

        if NUM_CLIPS_TO_SAVE > 0:
            write_clips(video, segments, NUM_CLIPS_TO_SAVE)

    if not rows:
        sys.exit("No labeled attempts - try lowering MOVE_PX / MIN_OFFSET_PX, setting USE_SURE = False, "
                 "or setting UPRIGHT_SIGN by hand (+1 / -1)")

    # plain utf-8 (no BOM): with no header row, a BOM would stick to the first video name
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if WRITE_HEADER:
            max_frames = max((len(r) - 4) // len(FIELDS) for r in rows)   # minus name, z_pred, frame_number, END_MARKER
            writer.writerow(["video_name", "z_pred", "frame_number"] + [
                f"{name}_{i}" for i in range(1, max_frames + 1) for name in FIELDS
            ])
        writer.writerows(rows)

    # which frames of which long video each row came from
    with open(OUTPUT_CSV.with_name(OUTPUT_CSV.stem + "_segments.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["row_name", "source_video", "start_frame", "end_frame", "label_frame"])
        writer.writerows(segment_log)

    print(f"\nSaved {len(rows)} rows ({n_dropped} attempts dropped) to {OUTPUT_CSV}")
    if n_out_of_range:
        print(f"Note: {n_out_of_range} normalized z_pred values fall outside 0-1 "
              "(the bottle at the label frame is beyond the range of the input frames).")


if __name__ == "__main__":
    main()