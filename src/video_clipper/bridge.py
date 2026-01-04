"""PyBridge API for the video clipper UI.

This module exposes video clipper functionality to the Bun/Elysia server
via PyBridge. Functions are called from TypeScript and return JSON-serializable data.
"""

import base64
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

# Global state - keeps models loaded between calls
_index: Optional["VideoIndex"] = None
_video_path: Optional[Path] = None


def load_video(path: str) -> dict:
    """Load a video file and return metadata."""
    from . import video
    from .index import VideoIndex

    global _index, _video_path

    video_path = Path(path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    _video_path = video_path
    duration = video.get_duration(video_path)

    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]

    fps_parts = stream["r_frame_rate"].split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])

    return {
        "duration": duration,
        "fps": fps,
        "width": stream["width"],
        "height": stream["height"],
        "path": str(video_path),
    }


def build_index(transcribe: bool = True) -> dict:
    """Build search index for the loaded video."""
    from .index import VideoIndex

    global _index, _video_path

    if _video_path is None:
        raise RuntimeError("No video loaded. Call load_video first.")

    index_path = _video_path.with_suffix(".vidx")

    if index_path.exists():
        _index = VideoIndex.load(index_path)
        return {
            "status": "loaded",
            "shots": len(_index.shots),
            "segments": len(_index.segments),
        }

    _index = VideoIndex(_video_path)
    _index.build(transcribe=transcribe)
    _index.save(index_path)

    return {
        "status": "built",
        "shots": len(_index.shots),
        "segments": len(_index.segments),
    }


def get_index_status() -> dict:
    """Check if index is ready."""
    global _index
    return {
        "ready": _index is not None,
        "shots": len(_index.shots) if _index else 0,
        "segments": len(_index.segments) if _index else 0,
    }


def search(query: str, top_k: int = 5) -> list[dict]:
    """Search for clips matching query."""
    global _index

    if _index is None:
        raise RuntimeError("No index loaded. Call build_index first.")

    matches = _index.search(query, top_k=top_k)

    return [
        {
            "start": m.start_time,
            "end": m.end_time,
            "score": m.score,
            "match_type": m.match_type,
            "matched_text": m.matched_text,
        }
        for m in matches
    ]


def get_thumbnail(time: float, width: int = 160) -> str:
    """Get a thumbnail at the given timestamp as base64."""
    global _video_path

    if _video_path is None:
        raise RuntimeError("No video loaded. Call load_video first.")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        temp_path = f.name

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(time),
            "-i", str(_video_path),
            "-vframes", "1",
            "-vf", f"scale={width}:-1",
            "-q:v", "3",
            temp_path,
        ],
        capture_output=True,
        check=True,
    )

    with open(temp_path, "rb") as f:
        data = f.read()

    Path(temp_path).unlink()

    return base64.b64encode(data).decode("utf-8")


def get_thumbnails_batch(times: list[float], width: int = 160) -> list[str]:
    """Get multiple thumbnails efficiently."""
    return [get_thumbnail(t, width) for t in times]


def export_clips(
    clips: list[dict],
    output_dir: str,
    stitch: bool = False,
) -> dict:
    """Export selected clips."""
    from .models import ClipMatch

    global _index, _video_path

    if _index is None or _video_path is None:
        raise RuntimeError("No index loaded.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if stitch:
        out_file = output_path / f"{_video_path.stem}_stitched.mp4"
        matches = [
            ClipMatch(start_time=c["start"], end_time=c["end"], score=0, match_type="manual")
            for c in clips
        ]
        _index.stitch_clips(out_file, matches)
        return {"outputs": [str(out_file)]}
    else:
        outputs = []
        for i, clip in enumerate(clips):
            out_file = output_path / f"{_video_path.stem}_clip_{i:02d}.mp4"
            _index.extract_clip(out_file, clip["start"], clip["end"])
            outputs.append(str(out_file))
        return {"outputs": outputs}
