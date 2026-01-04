"""
Video Clip Extractor

Extract video clips using natural language descriptions.

Basic usage:
    from video_clipper import VideoIndex

    index = VideoIndex("video.mp4")
    index.build()

    matches = index.search("person laughing")
    for match in matches:
        index.extract_clip(f"clip_{i}.mp4", match.start_time, match.end_time)
"""

from .models import Shot, TranscriptSegment, ClipMatch, format_timestamp
from .index import VideoIndex
from .embeddings import CLIPEmbedder
from .transcription import Transcriber

__version__ = "0.1.0"

__all__ = [
    # Main API
    "VideoIndex",
    # Data models
    "Shot",
    "TranscriptSegment",
    "ClipMatch",
    # Utilities
    "format_timestamp",
    # Low-level (for advanced use)
    "CLIPEmbedder",
    "Transcriber",
]
