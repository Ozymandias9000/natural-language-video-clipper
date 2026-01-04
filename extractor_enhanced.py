"""
Enhanced Video Clip Extractor
Extracts clips from video based on natural language descriptions.

Pipeline:
1. Scene detection (PySceneDetect) → shot boundaries
2. Keyframe extraction → representative frame per shot
3. CLIP embeddings → visual semantic index
4. Whisper transcription → audio semantic index
5. Vector search → match descriptions to timestamps
6. ffmpeg → cut clips

Performance optimizations:
- GPU auto-detection (CUDA when available)
- faster-whisper backend (4-8x faster transcription)
- Batched CLIP embeddings (image and text)
- Parallel keyframe extraction
- Pre-extracted audio caching
"""

import subprocess
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch


@dataclass
class Shot:
    """A detected shot/scene in the video."""
    index: int
    start_time: float
    end_time: float
    keyframe_path: Optional[Path] = None
    visual_embedding: Optional[np.ndarray] = None
    transcript_segments: list = None

    def __post_init__(self):
        if self.transcript_segments is None:
            self.transcript_segments = []

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
    match_type: str  # 'visual', 'audio', 'combined'
    matched_text: Optional[str] = None


class VideoClipExtractor:
    def __init__(
        self,
        clip_model: str = "ViT-B/32",
        whisper_model: str = "base",
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        whisper_backend: Literal["whisper", "faster-whisper"] = "faster-whisper",
        batch_size: int = 32,
        max_keyframe_workers: int = 8
    ):
        # Auto-detect GPU if device not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.clip_model_name = clip_model
        self.whisper_model_name = whisper_model
        self.device = device
        self.whisper_backend = whisper_backend
        self.batch_size = batch_size
        self.max_keyframe_workers = max_keyframe_workers
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazy load models
        self._clip_model = None
        self._clip_preprocess = None
        self._whisper_model = None

    @property
    def clip_model(self):
        if self._clip_model is None:
            import clip
            self._clip_model, self._clip_preprocess = clip.load(
                self.clip_model_name, device=self.device
            )
        return self._clip_model

    @property
    def clip_preprocess(self):
        if self._clip_preprocess is None:
            self.clip_model  # trigger load
        return self._clip_preprocess

    @property
    def whisper_model(self):
        if self._whisper_model is None:
            if self.whisper_backend == "faster-whisper":
                from faster_whisper import WhisperModel
                compute_type = "float16" if self.device == "cuda" else "int8"
                self._whisper_model = WhisperModel(
                    self.whisper_model_name,
                    device=self.device,
                    compute_type=compute_type
                )
            else:
                import whisper
                self._whisper_model = whisper.load_model(
                    self.whisper_model_name, device=self.device
                )
        return self._whisper_model

    def detect_scenes(
        self,
        video_path: Path,
        threshold: float = 27.0,
        min_scene_len: float = 0.5
    ) -> list[Shot]:
        """
        Detect scene boundaries using PySceneDetect.

        Args:
            video_path: Path to video file
            threshold: Content detection threshold (lower = more sensitive)
            min_scene_len: Minimum scene length in seconds

        Returns:
            List of Shot objects with start/end times
        """
        from scenedetect import detect, ContentDetector, AdaptiveDetector

        # ContentDetector works well for most videos
        # AdaptiveDetector is better for videos with lots of motion
        scene_list = detect(
            str(video_path),
            ContentDetector(threshold=threshold, min_scene_len=int(min_scene_len * 30))
        )

        shots = []
        for i, scene in enumerate(scene_list):
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            shots.append(Shot(index=i, start_time=start_time, end_time=end_time))

        # Handle edge case: no scenes detected (treat whole video as one shot)
        if not shots:
            duration = self._get_video_duration(video_path)
            shots.append(Shot(index=0, start_time=0.0, end_time=duration))

        return shots

    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration using ffprobe."""
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "json", str(video_path)
            ],
            capture_output=True, text=True
        )
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])

    def _extract_single_keyframe(
        self,
        video_path: Path,
        shot: Shot,
        output_dir: Path
    ) -> Shot:
        """Extract a single keyframe (for parallel execution)."""
        output_path = output_dir / f"shot_{shot.index:04d}.jpg"
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", str(shot.midpoint),
                "-i", str(video_path),
                "-vframes", "1", "-q:v", "2",
                str(output_path)
            ],
            capture_output=True
        )
        shot.keyframe_path = output_path
        return shot

    def extract_keyframes(
        self,
        video_path: Path,
        shots: list[Shot],
        output_dir: Optional[Path] = None
    ) -> list[Shot]:
        """
        Extract a keyframe from each shot (frame at midpoint).
        Uses parallel extraction for speed.

        Returns shots with keyframe_path populated.
        """
        output_dir = output_dir or self.cache_dir / "keyframes"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Parallel extraction using thread pool
        with ThreadPoolExecutor(max_workers=self.max_keyframe_workers) as executor:
            futures = {
                executor.submit(
                    self._extract_single_keyframe, video_path, shot, output_dir
                ): shot
                for shot in shots
            }
            for future in as_completed(futures):
                future.result()  # Propagate any exceptions

        return shots

    def compute_visual_embeddings(self, shots: list[Shot]) -> list[Shot]:
        """Compute CLIP embeddings for each shot's keyframe using batching."""
        from PIL import Image

        # Collect valid shots and preprocess images
        valid_shots = []
        images = []
        for shot in shots:
            if shot.keyframe_path and shot.keyframe_path.exists():
                image = Image.open(shot.keyframe_path).convert("RGB")
                images.append(self.clip_preprocess(image))
                valid_shots.append(shot)

        if not images:
            return shots

        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i : i + self.batch_size]
            batch_shots = valid_shots[i : i + self.batch_size]

            image_input = torch.stack(batch_images).to(self.device)

            with torch.no_grad():
                embeddings = self.clip_model.encode_image(image_input)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                embeddings = embeddings.cpu().numpy()

            for shot, emb in zip(batch_shots, embeddings):
                shot.visual_embedding = emb

        return shots

    def _extract_audio(self, video_path: Path) -> Path:
        """Pre-extract audio to WAV for faster Whisper processing."""
        audio_path = self.cache_dir / f"{video_path.stem}_audio.wav"
        if not audio_path.exists():
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(video_path),
                    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    str(audio_path)
                ],
                capture_output=True
            )
        return audio_path

    def transcribe_audio(
        self,
        video_path: Path,
        language: Optional[str] = None
    ) -> list[TranscriptSegment]:
        """
        Transcribe audio track using Whisper.
        Pre-extracts audio to WAV for faster processing.

        Returns list of timestamped transcript segments.
        """
        # Pre-extract audio for faster processing
        audio_path = self._extract_audio(video_path)

        segments = []

        if self.whisper_backend == "faster-whisper":
            transcribe_segments, _ = self.whisper_model.transcribe(
                str(audio_path),
                language=language,
                word_timestamps=True
            )
            for seg in transcribe_segments:
                segments.append(TranscriptSegment(
                    start_time=seg.start,
                    end_time=seg.end,
                    text=seg.text.strip()
                ))
        else:
            result = self.whisper_model.transcribe(
                str(audio_path),
                language=language,
                word_timestamps=True
            )
            for seg in result["segments"]:
                segments.append(TranscriptSegment(
                    start_time=seg["start"],
                    end_time=seg["end"],
                    text=seg["text"].strip()
                ))

        return segments

    def compute_text_embeddings(
        self,
        segments: list[TranscriptSegment]
    ) -> list[TranscriptSegment]:
        """Compute CLIP text embeddings for transcript segments using batching."""
        import clip

        # Collect valid segments
        valid_segments = [seg for seg in segments if seg.text]

        if not valid_segments:
            return segments

        # Process in batches
        for i in range(0, len(valid_segments), self.batch_size):
            batch_segments = valid_segments[i : i + self.batch_size]
            texts = [seg.text for seg in batch_segments]

            text_tokens = clip.tokenize(texts, truncate=True).to(self.device)

            with torch.no_grad():
                embeddings = self.clip_model.encode_text(text_tokens)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                embeddings = embeddings.cpu().numpy()

            for seg, emb in zip(batch_segments, embeddings):
                seg.embedding = emb

        return segments

    def assign_transcript_to_shots(
        self,
        shots: list[Shot],
        segments: list[TranscriptSegment]
    ) -> list[Shot]:
        """Assign transcript segments to overlapping shots."""
        for segment in segments:
            for shot in shots:
                # Check for temporal overlap
                if segment.start_time < shot.end_time and segment.end_time > shot.start_time:
                    shot.transcript_segments.append(segment)

        return shots

    def search(
        self,
        query: str,
        shots: list[Shot],
        segments: list[TranscriptSegment],
        top_k: int = 5,
        visual_weight: float = 0.6,
        audio_weight: float = 0.4
    ) -> list[ClipMatch]:
        """
        Search for clips matching the query description.

        Args:
            query: Natural language description of desired clip
            shots: Indexed shots with visual embeddings
            segments: Indexed transcript segments
            top_k: Number of results to return
            visual_weight: Weight for visual similarity
            audio_weight: Weight for audio/transcript similarity

        Returns:
            List of ClipMatch objects sorted by relevance
        """
        import clip

        # Embed query
        text_tokens = clip.tokenize([query], truncate=True).to(self.device)
        with torch.no_grad():
            query_embedding = self.clip_model.encode_text(text_tokens)
            query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
            query_embedding = query_embedding.cpu().numpy().squeeze()

        matches = []

        # Visual search (shot-level)
        for shot in shots:
            if shot.visual_embedding is not None:
                similarity = np.dot(query_embedding, shot.visual_embedding)
                matches.append(ClipMatch(
                    start_time=shot.start_time,
                    end_time=shot.end_time,
                    score=float(similarity) * visual_weight,
                    match_type='visual'
                ))

        # Audio search (segment-level)
        for segment in segments:
            if segment.embedding is not None:
                similarity = np.dot(query_embedding, segment.embedding)
                matches.append(ClipMatch(
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    score=float(similarity) * audio_weight,
                    match_type='audio',
                    matched_text=segment.text
                ))

        # Sort by score and merge overlapping matches
        matches.sort(key=lambda m: m.score, reverse=True)
        merged = self._merge_overlapping_matches(matches[:top_k * 2])

        return merged[:top_k]

    def _merge_overlapping_matches(
        self,
        matches: list[ClipMatch],
        gap_threshold: float = 1.0
    ) -> list[ClipMatch]:
        """Merge matches that are close together or overlapping."""
        if not matches:
            return []

        # Sort by start time
        sorted_matches = sorted(matches, key=lambda m: m.start_time)
        merged = [sorted_matches[0]]

        for match in sorted_matches[1:]:
            last = merged[-1]

            # If overlapping or within gap threshold, merge
            if match.start_time <= last.end_time + gap_threshold:
                merged[-1] = ClipMatch(
                    start_time=last.start_time,
                    end_time=max(last.end_time, match.end_time),
                    score=max(last.score, match.score),  # Take higher score
                    match_type='combined' if last.match_type != match.match_type else last.match_type,
                    matched_text=last.matched_text or match.matched_text
                )
            else:
                merged.append(match)

        # Re-sort by score
        merged.sort(key=lambda m: m.score, reverse=True)
        return merged

    def extract_clip(
        self,
        video_path: Path,
        output_path: Path,
        start_time: float,
        end_time: float,
        padding: float = 0.5,
        reencode: bool = False
    ) -> Path:
        """
        Extract a clip from the video using ffmpeg.

        Args:
            video_path: Source video
            output_path: Destination path
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            padding: Seconds to add before/after
            reencode: If True, re-encode for precise cuts (slower)

        Returns:
            Path to extracted clip
        """
        start = max(0, start_time - padding)
        end = end_time + padding

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if reencode:
            # Precise cuts but slower
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-ss", str(start),
                "-to", str(end),
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                str(output_path)
            ]
        else:
            # Fast copy (cuts on keyframes, might be slightly imprecise)
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", str(video_path),
                "-to", str(end - start),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                str(output_path)
            ]

        subprocess.run(cmd, capture_output=True)
        return output_path


class VideoIndex:
    """
    Pre-indexed video for fast repeated queries.
    """
    def __init__(self, video_path: Path, extractor: VideoClipExtractor):
        self.video_path = Path(video_path)
        self.extractor = extractor
        self.shots: list[Shot] = []
        self.segments: list[TranscriptSegment] = []
        self._indexed = False

    def build_index(
        self,
        scene_threshold: float = 27.0,
        transcribe: bool = True,
        verbose: bool = True
    ):
        """Build the full index for this video."""
        if verbose:
            print(f"Indexing: {self.video_path.name}")

        # 1. Detect scenes
        if verbose:
            print("  Detecting scenes...")
        self.shots = self.extractor.detect_scenes(
            self.video_path,
            threshold=scene_threshold
        )
        if verbose:
            print(f"  Found {len(self.shots)} shots")

        # 2. Extract keyframes
        if verbose:
            print("  Extracting keyframes...")
        self.extractor.extract_keyframes(self.video_path, self.shots)

        # 3. Compute visual embeddings
        if verbose:
            print("  Computing visual embeddings...")
        self.extractor.compute_visual_embeddings(self.shots)

        # 4. Transcribe audio
        if transcribe:
            if verbose:
                print("  Transcribing audio...")
            self.segments = self.extractor.transcribe_audio(self.video_path)
            if verbose:
                print(f"  Found {len(self.segments)} transcript segments")

            # 5. Compute text embeddings
            if verbose:
                print("  Computing text embeddings...")
            self.extractor.compute_text_embeddings(self.segments)

            # 6. Assign transcript to shots
            self.extractor.assign_transcript_to_shots(self.shots, self.segments)

        self._indexed = True
        if verbose:
            print("  Done!")

    def search(self, query: str, top_k: int = 5, **kwargs) -> list[ClipMatch]:
        """Search the index for matching clips."""
        if not self._indexed:
            raise RuntimeError("Index not built. Call build_index() first.")

        return self.extractor.search(
            query=query,
            shots=self.shots,
            segments=self.segments,
            top_k=top_k,
            **kwargs
        )

    def extract_clips(
        self,
        queries: list[str],
        output_dir: Path,
        top_k: int = 1,
        **kwargs
    ) -> dict[str, list[Path]]:
        """
        Search and extract clips for multiple queries.

        Returns dict mapping query -> list of extracted clip paths.
        """
        output_dir = Path(output_dir)
        results = {}

        for query in queries:
            matches = self.search(query, top_k=top_k, **kwargs)
            results[query] = []

            for i, match in enumerate(matches):
                # Sanitize query for filename
                safe_query = "".join(c if c.isalnum() else "_" for c in query)[:50]
                output_path = output_dir / f"{safe_query}_{i:02d}.mp4"

                self.extractor.extract_clip(
                    self.video_path,
                    output_path,
                    match.start_time,
                    match.end_time
                )
                results[query].append(output_path)

        return results

    def save_index(self, path: Path):
        """Save index to disk for later reuse."""
        import pickle

        # Store numpy arrays separately for efficiency
        data = {
            'video_path': str(self.video_path),
            'shots': [
                {
                    'index': s.index,
                    'start_time': s.start_time,
                    'end_time': s.end_time,
                    'keyframe_path': str(s.keyframe_path) if s.keyframe_path else None,
                    'visual_embedding': s.visual_embedding,
                    'transcript_segments': [
                        {
                            'start_time': seg.start_time,
                            'end_time': seg.end_time,
                            'text': seg.text,
                            'embedding': seg.embedding
                        }
                        for seg in s.transcript_segments
                    ]
                }
                for s in self.shots
            ],
            'segments': [
                {
                    'start_time': seg.start_time,
                    'end_time': seg.end_time,
                    'text': seg.text,
                    'embedding': seg.embedding
                }
                for seg in self.segments
            ]
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load_index(cls, path: Path, extractor: VideoClipExtractor) -> 'VideoIndex':
        """Load a previously saved index."""
        import pickle

        with open(path, 'rb') as f:
            data = pickle.load(f)

        index = cls(Path(data['video_path']), extractor)

        # Reconstruct segments first (for reference)
        segment_map = {}
        for seg_data in data['segments']:
            seg = TranscriptSegment(
                start_time=seg_data['start_time'],
                end_time=seg_data['end_time'],
                text=seg_data['text'],
                embedding=seg_data['embedding']
            )
            index.segments.append(seg)
            segment_map[(seg.start_time, seg.end_time)] = seg

        # Reconstruct shots
        for shot_data in data['shots']:
            shot = Shot(
                index=shot_data['index'],
                start_time=shot_data['start_time'],
                end_time=shot_data['end_time'],
                keyframe_path=Path(shot_data['keyframe_path']) if shot_data['keyframe_path'] else None,
                visual_embedding=shot_data['visual_embedding']
            )

            # Reconnect transcript segments
            for seg_data in shot_data['transcript_segments']:
                key = (seg_data['start_time'], seg_data['end_time'])
                if key in segment_map:
                    shot.transcript_segments.append(segment_map[key])

            index.shots.append(shot)

        index._indexed = True
        return index


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


# Enhanced classes with full scene extraction

class EnhancedVideoClipExtractor(VideoClipExtractor):
    """Extended extractor with full scene extraction capabilities."""

    def find_containing_shot(self, timestamp: float, shots: list[Shot]) -> Optional[Shot]:
        """Find the shot that contains the given timestamp."""
        for shot in shots:
            if shot.start_time <= timestamp <= shot.end_time:
                return shot
        return None

    def search_with_scene_expansion(
        self,
        query: str,
        shots: list[Shot],
        segments: list[TranscriptSegment],
        top_k: int = 5,
        visual_weight: float = 0.6,
        audio_weight: float = 0.4,
        full_scene: bool = False
    ) -> list[ClipMatch]:
        """
        Search for clips with optional full scene expansion.

        Args:
            query: Natural language description
            shots: Indexed shots
            segments: Indexed transcript segments
            top_k: Number of results
            visual_weight: Weight for visual similarity
            audio_weight: Weight for audio similarity
            full_scene: If True, expand matches to full scene boundaries

        Returns:
            List of ClipMatch objects
        """
        # Use base search
        matches = self.search(query, shots, segments, top_k, visual_weight, audio_weight)

        # Expand to full scene if requested
        if full_scene:
            expanded = []
            seen_shots = set()

            for match in matches:
                # Find containing shot based on match midpoint
                midpoint = (match.start_time + match.end_time) / 2
                containing_shot = self.find_containing_shot(midpoint, shots)

                if containing_shot and containing_shot.index not in seen_shots:
                    seen_shots.add(containing_shot.index)
                    expanded.append(ClipMatch(
                        start_time=containing_shot.start_time,
                        end_time=containing_shot.end_time,
                        score=match.score,
                        match_type=f"{match.match_type}_scene",
                        matched_text=match.matched_text
                    ))
                elif not containing_shot:
                    expanded.append(match)

            return expanded

        return matches


class EnhancedVideoIndex(VideoIndex):
    """Enhanced index with full scene extraction."""

    def __init__(self, video_path: Path, extractor: VideoClipExtractor = None):
        if extractor is None:
            extractor = EnhancedVideoClipExtractor()
        super().__init__(video_path, extractor)

    def search(self, query: str, top_k: int = 5, full_scene: bool = False, **kwargs) -> list[ClipMatch]:
        """Search with optional full scene extraction."""
        if not self._indexed:
            raise RuntimeError("Index not built. Call build_index() first.")

        if isinstance(self.extractor, EnhancedVideoClipExtractor):
            return self.extractor.search_with_scene_expansion(
                query=query,
                shots=self.shots,
                segments=self.segments,
                top_k=top_k,
                full_scene=full_scene,
                **kwargs
            )
        else:
            return self.extractor.search(
                query=query,
                shots=self.shots,
                segments=self.segments,
                top_k=top_k,
                **kwargs
            )
