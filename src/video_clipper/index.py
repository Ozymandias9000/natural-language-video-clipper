"""Video index for semantic search.

The VideoIndex class orchestrates the full indexing pipeline and provides
a simple interface for searching and extracting clips.
"""

import pickle
from pathlib import Path
from typing import Callable, Optional

from .models import ClipMatch, Shot, TranscriptSegment
from .embeddings import CLIPEmbedder, compute_shot_embeddings, compute_segment_embeddings
from .transcription import Transcriber, assign_segments_to_shots
from . import scene
from . import search as search_module
from . import video


class VideoIndex:
    """
    Pre-indexed video for fast semantic search.

    Encapsulates the full pipeline: scene detection, keyframe extraction,
    visual/text embeddings, and search. Indexes can be saved/loaded for
    fast repeated queries.
    """

    def __init__(
        self,
        video_path: Path,
        cache_dir: Optional[Path] = None,
        clip_model: str = "ViT-B/32",
        whisper_model: str = "base",
        whisper_backend: str = "faster-whisper",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_keyframe_workers: int = 8,
    ):
        """
        Initialize a video index.

        Args:
            video_path: Path to the video file
            cache_dir: Directory for caching keyframes and audio
            clip_model: CLIP model variant
            whisper_model: Whisper model size
            whisper_backend: 'whisper' or 'faster-whisper'
            device: Compute device (auto-detected if None)
            batch_size: Batch size for embedding computation
            max_keyframe_workers: Parallel workers for keyframe extraction
        """
        self.video_path = Path(video_path).resolve()
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_keyframe_workers = max_keyframe_workers

        self._embedder = CLIPEmbedder(
            model_name=clip_model,
            device=device,
            batch_size=batch_size,
        )
        self._transcriber = Transcriber(
            model_name=whisper_model,
            backend=whisper_backend,
            device=device,
        )

        self.shots: list[Shot] = []
        self.segments: list[TranscriptSegment] = []
        self._indexed = False

    def build(
        self,
        scene_threshold: float = 27.0,
        transcribe: bool = True,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> "VideoIndex":
        """
        Build the full index for this video.

        Args:
            scene_threshold: Scene detection threshold (lower = more sensitive)
            transcribe: Whether to transcribe audio
            on_progress: Optional callback for progress updates

        Returns:
            self, for chaining
        """

        def report(msg: str):
            if on_progress:
                on_progress(msg)

        report("Detecting scenes...")
        self.shots = scene.detect_scenes(self.video_path, threshold=scene_threshold)
        report(f"Found {len(self.shots)} scenes")

        report("Extracting keyframes...")
        keyframe_dir = self.cache_dir / "keyframes"
        video.extract_keyframes(
            self.video_path,
            self.shots,
            keyframe_dir,
            max_workers=self.max_keyframe_workers,
        )

        report("Computing visual embeddings...")
        compute_shot_embeddings(self.shots, self._embedder)

        if transcribe:
            report("Extracting audio...")
            audio_path = self.cache_dir / f"{self.video_path.stem}_audio.wav"
            if not audio_path.exists():
                video.extract_audio(self.video_path, audio_path)

            report("Transcribing audio...")
            self.segments = self._transcriber.transcribe(audio_path)
            report(f"Found {len(self.segments)} transcript segments")

            report("Computing text embeddings...")
            compute_segment_embeddings(self.segments, self._embedder)
            assign_segments_to_shots(self.shots, self.segments)

        self._indexed = True
        report("Done!")
        return self

    def search(
        self,
        query: str,
        top_k: int = 5,
        visual_weight: float = 0.6,
        audio_weight: float = 0.4,
        full_scene: bool = False,
        expand: bool = False,
    ) -> list[ClipMatch]:
        """
        Search for clips matching a natural language query.

        Args:
            query: Description of desired content
            top_k: Number of results
            visual_weight: Weight for visual similarity (0-1)
            audio_weight: Weight for transcript similarity (0-1)
            full_scene: Expand matches to full scene boundaries
            expand: Use LLM to expand query into variations (requires ANTHROPIC_API_KEY)

        Returns:
            List of ClipMatch objects sorted by relevance
        """
        if not self._indexed:
            raise RuntimeError("Index not built. Call build() first.")

        return search_module.search(
            query=query,
            shots=self.shots,
            segments=self.segments,
            embedder=self._embedder,
            top_k=top_k,
            visual_weight=visual_weight,
            audio_weight=audio_weight,
            full_scene=full_scene,
            expand=expand,
        )

    def extract_clip(
        self,
        output_path: Path,
        start_time: float,
        end_time: float,
        padding: float = 0.5,
        reencode: bool = False,
    ) -> Path:
        """Extract a clip from the video."""
        return video.extract_clip(
            self.video_path,
            output_path,
            start_time,
            end_time,
            padding=padding,
            reencode=reencode,
        )

    def stitch_clips(
        self,
        output_path: Path,
        matches: list[ClipMatch],
        padding: float = 0.5,
        reencode: bool = False,
    ) -> Path:
        """
        Stitch multiple clips into a single output file.

        Args:
            output_path: Destination file path
            matches: List of ClipMatch objects to stitch (will be sorted chronologically)
            padding: Seconds to add before/after each clip
            reencode: If True, re-encode for frame-accurate cuts

        Returns:
            Path to the stitched output file
        """
        sorted_matches = sorted(matches, key=lambda m: m.start_time)
        time_ranges = [(m.start_time, m.end_time) for m in sorted_matches]
        return video.stitch_clips(
            self.video_path,
            output_path,
            time_ranges,
            padding=padding,
            reencode=reencode,
        )

    def extract_clips(
        self,
        queries: list[str],
        output_dir: Path,
        top_k: int = 1,
        **search_kwargs,
    ) -> dict[str, list[Path]]:
        """
        Search and extract clips for multiple queries.

        Returns:
            Dict mapping query -> list of extracted clip paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        for query in queries:
            matches = self.search(query, top_k=top_k, **search_kwargs)
            results[query] = []

            for i, match in enumerate(matches):
                safe_name = "".join(c if c.isalnum() else "_" for c in query)[:50]
                output_path = output_dir / f"{safe_name}_{i:02d}.mp4"

                self.extract_clip(
                    output_path,
                    match.start_time,
                    match.end_time,
                )
                results[query].append(output_path)

        return results

    def save(self, path: Path) -> None:
        """Save index to disk for later reuse."""
        data = {
            "video_path": str(self.video_path),
            "shots": [
                {
                    "index": s.index,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "keyframe_path": str(s.keyframe_path) if s.keyframe_path else None,
                    "visual_embedding": s.visual_embedding,
                    "transcript_segments": [
                        {
                            "start_time": seg.start_time,
                            "end_time": seg.end_time,
                            "text": seg.text,
                            "embedding": seg.embedding,
                        }
                        for seg in s.transcript_segments
                    ],
                }
                for s in self.shots
            ],
            "segments": [
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "text": seg.text,
                    "embedding": seg.embedding,
                }
                for seg in self.segments
            ],
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path, **kwargs) -> "VideoIndex":
        """
        Load a previously saved index.

        Args:
            path: Path to saved index file
            **kwargs: Override constructor arguments (e.g., device)

        Returns:
            Loaded VideoIndex
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        index = cls(Path(data["video_path"]), **kwargs)

        # Reconstruct segments
        segment_map = {}
        for seg_data in data["segments"]:
            seg = TranscriptSegment(
                start_time=seg_data["start_time"],
                end_time=seg_data["end_time"],
                text=seg_data["text"],
                embedding=seg_data["embedding"],
            )
            index.segments.append(seg)
            segment_map[(seg.start_time, seg.end_time)] = seg

        # Reconstruct shots
        for shot_data in data["shots"]:
            shot = Shot(
                index=shot_data["index"],
                start_time=shot_data["start_time"],
                end_time=shot_data["end_time"],
                keyframe_path=Path(shot_data["keyframe_path"]) if shot_data["keyframe_path"] else None,
                visual_embedding=shot_data["visual_embedding"],
            )
            for seg_data in shot_data["transcript_segments"]:
                key = (seg_data["start_time"], seg_data["end_time"])
                if key in segment_map:
                    shot.transcript_segments.append(segment_map[key])
            index.shots.append(shot)

        index._indexed = True
        return index
