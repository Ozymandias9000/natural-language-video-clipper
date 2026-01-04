"""Video file operations using ffmpeg/ffprobe.

This module hides all subprocess and ffmpeg complexity behind a simple interface.
All operations are synchronous and raise exceptions on failure.
"""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from .models import Shot


def get_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def extract_audio(video_path: Path, output_path: Path) -> Path:
    """Extract audio track to 16kHz mono WAV (optimal for Whisper)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(output_path),
        ],
        capture_output=True,
        check=True,
    )
    return output_path


def extract_frame(video_path: Path, timestamp: float, output_path: Path) -> Path:
    """Extract a single frame at the given timestamp."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(timestamp),
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            str(output_path),
        ],
        capture_output=True,
        check=True,
    )
    return output_path


def extract_keyframes(
    video_path: Path,
    shots: list[Shot],
    output_dir: Path,
    max_workers: int = 8,
) -> list[Shot]:
    """
    Extract keyframe (midpoint frame) for each shot in parallel.

    Modifies shots in-place, setting keyframe_path for each.
    Returns the same list for chaining.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    def _extract_one(shot: Shot) -> Shot:
        output_path = output_dir / f"shot_{shot.index:04d}.jpg"
        extract_frame(video_path, shot.midpoint, output_path)
        shot.keyframe_path = output_path
        return shot

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_extract_one, shot): shot for shot in shots}
        for future in as_completed(futures):
            future.result()  # Propagate exceptions

    return shots


def extract_clip(
    video_path: Path,
    output_path: Path,
    start_time: float,
    end_time: float,
    padding: float = 0.5,
    reencode: bool = False,
) -> Path:
    """
    Extract a clip from the video.

    Args:
        video_path: Source video file
        output_path: Destination file path
        start_time: Clip start in seconds
        end_time: Clip end in seconds
        padding: Seconds to add before/after the clip
        reencode: If True, re-encode for frame-accurate cuts (slower but precise)

    Returns:
        Path to the extracted clip
    """
    start = max(0, start_time - padding)
    end = end_time + padding

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if reencode:
        # Frame-accurate cuts with re-encoding
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-ss", str(start),
            "-to", str(end),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            str(output_path),
        ]
    else:
        # Fast stream copy (cuts on keyframes, may be slightly imprecise)
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(video_path),
            "-to", str(end - start),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(output_path),
        ]

    subprocess.run(cmd, capture_output=True, check=True)
    return output_path
