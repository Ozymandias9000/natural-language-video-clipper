"""
Enhanced Video Clip Extractor with full scene extraction
"""

from extractor import *

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