"""
Video Clip Extractor
Extracts clips from video based on natural language descriptions.

Pipeline:
1. Scene detection (PySceneDetect) → shot boundaries
2. Keyframe extraction → representative frame per shot
3. CLIP embeddings → visual semantic index
4. Whisper transcription → audio semantic index
5. Vector search → match descriptions to timestamps
6. ffmpeg → cut clips
"""

import subprocess
import json
from pathlib import Path
from dataclasses import dataclass
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
        device: str = "cpu",
        cache_dir: Optional[Path] = None
    ):
        self.clip_model_name = clip_model
        self.whisper_model_name = whisper_model
        self.device = device
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
    
    def extract_keyframes(
        self, 
        video_path: Path, 
        shots: list[Shot],
        output_dir: Optional[Path] = None
    ) -> list[Shot]:
        """
        Extract a keyframe from each shot (frame at midpoint).
        
        Returns shots with keyframe_path populated.
        """
        output_dir = output_dir or self.cache_dir / "keyframes"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for shot in shots:
            output_path = output_dir / f"shot_{shot.index:04d}.jpg"
            
            # Extract frame at shot midpoint
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
        
        return shots
    
    def compute_visual_embeddings(self, shots: list[Shot]) -> list[Shot]:
        """Compute CLIP embeddings for each shot's keyframe."""
        import clip
        from PIL import Image
        import torch
        
        for shot in shots:
            if shot.keyframe_path and shot.keyframe_path.exists():
                image = Image.open(shot.keyframe_path).convert("RGB")
                image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    embedding = self.clip_model.encode_image(image_input)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                    shot.visual_embedding = embedding.cpu().numpy().squeeze()
        
        return shots
    
    def transcribe_audio(
        self, 
        video_path: Path,
        language: Optional[str] = None
    ) -> list[TranscriptSegment]:
        """
        Transcribe audio track using Whisper.
        
        Returns list of timestamped transcript segments.
        """
        result = self.whisper_model.transcribe(
            str(video_path),
            language=language,
            word_timestamps=True
        )
        
        segments = []
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
        """Compute CLIP text embeddings for transcript segments."""
        import clip
        import torch
        
        for segment in segments:
            if segment.text:
                text_tokens = clip.tokenize([segment.text], truncate=True).to(self.device)
                
                with torch.no_grad():
                    embedding = self.clip_model.encode_text(text_tokens)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                    segment.embedding = embedding.cpu().numpy().squeeze()
        
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
        import torch
        
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
