"""CLIP embeddings for images and text.

Handles model loading, GPU detection, and batched embedding computation.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .models import Shot, TranscriptSegment


class CLIPEmbedder:
    """
    CLIP model wrapper for computing image and text embeddings.

    Lazily loads the model on first use. Supports batched processing
    for efficiency with large numbers of images/texts.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """
        Initialize the embedder.

        Args:
            model_name: CLIP model variant (ViT-B/32, ViT-L/14, etc.)
            device: Compute device (auto-detected if None)
            batch_size: Batch size for embedding computation
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or self._detect_device()

        self._model = None
        self._preprocess = None

    @staticmethod
    def _detect_device() -> str:
        """Auto-detect the best available compute device."""
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self):
        """Lazily load the CLIP model."""
        if self._model is None:
            import clip

            self._model, self._preprocess = clip.load(self.model_name, device=self.device)

    @property
    def model(self):
        self._load_model()
        return self._model

    @property
    def preprocess(self):
        self._load_model()
        return self._preprocess

    def embed_images(self, image_paths: list[Path]) -> list[np.ndarray]:
        """
        Compute normalized embeddings for a list of images.

        Args:
            image_paths: Paths to image files

        Returns:
            List of embedding arrays (same order as input)
        """
        from PIL import Image

        if not image_paths:
            return []

        # Preprocess all images
        images = []
        for path in image_paths:
            if path.exists():
                img = Image.open(path).convert("RGB")
                images.append(self.preprocess(img))

        if not images:
            return []

        embeddings = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            batch_tensor = torch.stack(batch).to(self.device)

            with torch.no_grad():
                batch_emb = self.model.encode_image(batch_tensor)
                batch_emb = batch_emb / batch_emb.norm(dim=-1, keepdim=True)
                embeddings.extend(batch_emb.cpu().numpy())

        return embeddings

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        """
        Compute normalized embeddings for a list of texts.

        Args:
            texts: List of text strings

        Returns:
            List of embedding arrays (same order as input)
        """
        import clip

        if not texts:
            return []

        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            tokens = clip.tokenize(batch, truncate=True).to(self.device)

            with torch.no_grad():
                batch_emb = self.model.encode_text(tokens)
                batch_emb = batch_emb / batch_emb.norm(dim=-1, keepdim=True)
                embeddings.extend(batch_emb.cpu().numpy())

        return embeddings

    def embed_query(self, query: str, use_templates: bool = True) -> np.ndarray:
        """
        Compute embedding for a single query string.

        Args:
            query: Natural language query
            use_templates: If True, use CLIP-friendly prompt templates and
                          average the embeddings for better retrieval

        Returns:
            Normalized embedding array
        """
        if not use_templates:
            return self.embed_texts([query])[0]

        # CLIP responds better to caption-like descriptions
        templates = [
            "{}",  # Original query
            "a photo of {}",
            "a video frame showing {}",
            "a scene with {}",
            "an image of {}",
        ]

        prompts = [t.format(query) for t in templates]
        embeddings = self.embed_texts(prompts)

        # Average and re-normalize
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        return avg_embedding


def compute_shot_embeddings(shots: list[Shot], embedder: CLIPEmbedder) -> list[Shot]:
    """
    Compute visual embeddings for shots with keyframes.

    Modifies shots in-place, setting visual_embedding for each.
    Returns the same list for chaining.
    """
    # Collect shots with valid keyframes
    valid_shots = [s for s in shots if s.keyframe_path and s.keyframe_path.exists()]
    paths = [s.keyframe_path for s in valid_shots]

    embeddings = embedder.embed_images(paths)

    for shot, emb in zip(valid_shots, embeddings):
        shot.visual_embedding = emb

    return shots


def compute_segment_embeddings(
    segments: list[TranscriptSegment],
    embedder: CLIPEmbedder,
) -> list[TranscriptSegment]:
    """
    Compute text embeddings for transcript segments.

    Modifies segments in-place, setting embedding for each.
    Returns the same list for chaining.
    """
    valid_segments = [s for s in segments if s.text]
    texts = [s.text for s in valid_segments]

    embeddings = embedder.embed_texts(texts)

    for seg, emb in zip(valid_segments, embeddings):
        seg.embedding = emb

    return segments
