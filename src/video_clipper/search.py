"""Semantic search over video content.

Matches query embeddings against visual and audio embeddings,
with support for merging overlapping results and scene expansion.
"""

import numpy as np

from .models import ClipMatch, Shot, TranscriptSegment
from .embeddings import CLIPEmbedder
from . import scene as scene_module


def search(
    query: str,
    shots: list[Shot],
    segments: list[TranscriptSegment],
    embedder: CLIPEmbedder,
    top_k: int = 5,
    visual_weight: float = 0.6,
    audio_weight: float = 0.4,
    full_scene: bool = False,
) -> list[ClipMatch]:
    """
    Search for clips matching a natural language query.

    Searches both visual (shot keyframes) and audio (transcript) content,
    combining results with configurable weights.

    Args:
        query: Natural language description of desired content
        shots: Indexed shots with visual embeddings
        segments: Indexed transcript segments with text embeddings
        embedder: CLIP embedder for query encoding
        top_k: Number of results to return
        visual_weight: Weight for visual similarity (0-1)
        audio_weight: Weight for audio/transcript similarity (0-1)
        full_scene: If True, expand matches to full scene boundaries

    Returns:
        List of ClipMatch objects sorted by relevance score
    """
    query_embedding = embedder.embed_query(query)

    matches = []

    # Visual search (shot-level)
    for shot in shots:
        if shot.visual_embedding is not None:
            similarity = float(np.dot(query_embedding, shot.visual_embedding))
            matches.append(
                ClipMatch(
                    start_time=shot.start_time,
                    end_time=shot.end_time,
                    score=similarity * visual_weight,
                    match_type="visual",
                )
            )

    # Audio search (segment-level)
    for segment in segments:
        if segment.embedding is not None:
            similarity = float(np.dot(query_embedding, segment.embedding))
            matches.append(
                ClipMatch(
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    score=similarity * audio_weight,
                    match_type="audio",
                    matched_text=segment.text,
                )
            )

    # Sort by score and merge overlapping
    matches.sort(key=lambda m: m.score, reverse=True)
    merged = _merge_overlapping(matches[: top_k * 2])

    # Expand to full scenes if requested
    if full_scene:
        merged = _expand_to_scenes(merged, shots)

    return merged[:top_k]


def _merge_overlapping(
    matches: list[ClipMatch],
    gap_threshold: float = 1.0,
) -> list[ClipMatch]:
    """
    Merge matches that overlap or are within gap_threshold seconds.

    When merging, takes the higher score and combines match types.
    """
    if not matches:
        return []

    sorted_matches = sorted(matches, key=lambda m: m.start_time)
    merged = [sorted_matches[0]]

    for match in sorted_matches[1:]:
        last = merged[-1]

        if match.start_time <= last.end_time + gap_threshold:
            # Merge with previous
            merged[-1] = ClipMatch(
                start_time=last.start_time,
                end_time=max(last.end_time, match.end_time),
                score=max(last.score, match.score),
                match_type="combined" if last.match_type != match.match_type else last.match_type,
                matched_text=last.matched_text or match.matched_text,
            )
        else:
            merged.append(match)

    # Re-sort by score
    merged.sort(key=lambda m: m.score, reverse=True)
    return merged


def _expand_to_scenes(
    matches: list[ClipMatch],
    shots: list[Shot],
) -> list[ClipMatch]:
    """Expand each match to its containing scene boundaries."""
    expanded = []
    seen_shots = set()

    for match in matches:
        midpoint = (match.start_time + match.end_time) / 2
        containing_shot = scene_module.find_containing_shot(midpoint, shots)

        if containing_shot and containing_shot.index not in seen_shots:
            seen_shots.add(containing_shot.index)
            expanded.append(
                ClipMatch(
                    start_time=containing_shot.start_time,
                    end_time=containing_shot.end_time,
                    score=match.score,
                    match_type=f"{match.match_type}_scene",
                    matched_text=match.matched_text,
                )
            )
        elif not containing_shot:
            expanded.append(match)

    return expanded
