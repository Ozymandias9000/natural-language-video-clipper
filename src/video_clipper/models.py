"""Data models for video clip extraction."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Shot:
    """A detected shot/scene in the video."""

    index: int
    start_time: float
    end_time: float
    keyframe_path: Optional[Path] = None
    visual_embedding: Optional[np.ndarray] = None
    transcript_segments: list["TranscriptSegment"] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def midpoint(self) -> float:
        return (self.start_time + self.end_time) / 2


@dataclass
class TranscriptSegment:
    """A segment of transcribed audio."""

    start_time: float
    end_time: float
    text: str
    embedding: Optional[np.ndarray] = None


@dataclass
class ClipMatch:
    """A matched clip with relevance score."""

    start_time: float
    end_time: float
    score: float
    match_type: str  # 'visual', 'audio', 'combined', or '*_scene'
    matched_text: Optional[str] = None


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
