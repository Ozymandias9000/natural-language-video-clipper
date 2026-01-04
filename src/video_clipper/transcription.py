"""Audio transcription using Whisper.

Supports both OpenAI Whisper and faster-whisper backends.
"""

from pathlib import Path
from typing import Literal, Optional

from .models import TranscriptSegment


class Transcriber:
    """
    Whisper-based audio transcription.

    Lazily loads the model on first use. Supports both standard Whisper
    and faster-whisper (4-8x faster) backends.
    """

    def __init__(
        self,
        model_name: str = "base",
        backend: Literal["whisper", "faster-whisper"] = "faster-whisper",
        device: Optional[str] = None,
    ):
        """
        Initialize the transcriber.

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            backend: Which Whisper implementation to use
            device: Compute device (auto-detected if None)
        """
        self.model_name = model_name
        self.backend = backend
        self.device = device or self._detect_device()

        self._model = None

    @staticmethod
    def _detect_device() -> str:
        """Auto-detect the best available compute device."""
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self):
        """Lazily load the Whisper model."""
        if self._model is not None:
            return

        if self.backend == "faster-whisper":
            from faster_whisper import WhisperModel

            # faster-whisper doesn't support MPS
            device = self.device if self.device != "mps" else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"

            self._model = WhisperModel(
                self.model_name,
                device=device,
                compute_type=compute_type,
            )
        else:
            import whisper

            self._model = whisper.load_model(self.model_name, device=self.device)

    @property
    def model(self):
        self._load_model()
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
    ) -> list[TranscriptSegment]:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to audio file (WAV recommended for speed)
            language: Language code (auto-detected if None)

        Returns:
            List of timestamped transcript segments
        """
        segments = []

        if self.backend == "faster-whisper":
            result_segments, _ = self.model.transcribe(
                str(audio_path),
                language=language,
                word_timestamps=True,
            )
            for seg in result_segments:
                segments.append(
                    TranscriptSegment(
                        start_time=seg.start,
                        end_time=seg.end,
                        text=seg.text.strip(),
                    )
                )
        else:
            result = self.model.transcribe(
                str(audio_path),
                language=language,
                word_timestamps=True,
            )
            for seg in result["segments"]:
                segments.append(
                    TranscriptSegment(
                        start_time=seg["start"],
                        end_time=seg["end"],
                        text=seg["text"].strip(),
                    )
                )

        return segments


def assign_segments_to_shots(
    shots: list,
    segments: list[TranscriptSegment],
) -> None:
    """
    Assign transcript segments to overlapping shots.

    Modifies shots in-place, appending segments to each shot's
    transcript_segments list where there is temporal overlap.
    """
    for segment in segments:
        for shot in shots:
            # Check for temporal overlap
            if segment.start_time < shot.end_time and segment.end_time > shot.start_time:
                shot.transcript_segments.append(segment)
