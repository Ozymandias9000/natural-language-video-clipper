"""Scene detection using PySceneDetect.

Detects shot boundaries in video based on visual content changes.
"""

from pathlib import Path

from .models import Shot
from . import video


def detect_scenes(
    video_path: Path,
    threshold: float = 27.0,
    min_scene_len: float = 0.5,
) -> list[Shot]:
    """
    Detect scene/shot boundaries in a video.

    Args:
        video_path: Path to the video file
        threshold: Content detection threshold (lower = more sensitive, default 27.0)
        min_scene_len: Minimum scene length in seconds

    Returns:
        List of Shot objects with start/end times.
        If no scenes detected, returns a single shot covering the entire video.
    """
    from scenedetect import detect, ContentDetector

    # min_scene_len is in frames, assume 30fps as default
    min_frames = int(min_scene_len * 30)

    scene_list = detect(
        str(video_path),
        ContentDetector(threshold=threshold, min_scene_len=min_frames),
    )

    shots = [
        Shot(
            index=i,
            start_time=scene[0].get_seconds(),
            end_time=scene[1].get_seconds(),
        )
        for i, scene in enumerate(scene_list)
    ]

    # Fallback: treat entire video as one shot if no boundaries detected
    if not shots:
        duration = video.get_duration(video_path)
        shots = [Shot(index=0, start_time=0.0, end_time=duration)]

    return shots


def find_containing_shot(timestamp: float, shots: list[Shot]) -> Shot | None:
    """Find the shot containing a given timestamp."""
    for shot in shots:
        if shot.start_time <= timestamp <= shot.end_time:
            return shot
    return None
