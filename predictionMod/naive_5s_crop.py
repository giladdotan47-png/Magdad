"""
Naive split: cut every video in a folder into fixed N-second clips with ffmpeg.

"""

import shutil
import subprocess
import sys
from pathlib import Path

INPUT_DIR = Path(r"C:\Users\TLP\Desktop\magdad\predictionMod\preprocessed")   # TODO
OUTPUT_DIR = Path(r"C:\Users\TLP\Desktop\magdad\predictionMod\seperated")  # TODO
SEGMENT_SECONDS = 5

# True  = exact 5s cuts, re-encodes the video (slower, tiny quality loss)
# False = very fast, no re-encoding, but cuts snap to the nearest keyframe (lengths are not exact)
REENCODE = True

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def split_video(video: Path):
    out_pattern = OUTPUT_DIR / f"{video.stem}_%03d.mp4"

    if REENCODE:
        codec_args = [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-an",
            "-force_key_frames", f"expr:gte(t,n_forced*{SEGMENT_SECONDS})",
        ]
    else:
        codec_args = ["-c", "copy", "-map", "0"]

    cmd = [
        "ffmpeg", "-y", "-i", str(video),
        *codec_args,
        "-f", "segment", "-segment_time", str(SEGMENT_SECONDS),
        "-reset_timestamps", "1",
        str(out_pattern),
    ]
    subprocess.run(cmd, check=True)


def main():
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found in PATH")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    for i, video in enumerate(videos, start=1):
        print(f"[{i}/{len(videos)}] {video.name}")
        split_video(video)

    print(f"Done. Clips saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()