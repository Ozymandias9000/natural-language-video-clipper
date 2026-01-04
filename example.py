#!/usr/bin/env python3
"""
Minimal example: extract clips from a video using natural language.
"""

from pathlib import Path
from extractor_enhanced import VideoClipExtractor, VideoIndex

def main():
    # Config
    VIDEO_PATH = Path("your_video.mp4")  # Change this
    QUERIES = [
        "two people having a conversation",
        "outdoor scene with trees",
        "someone mentions money",
    ]
    OUTPUT_DIR = Path("./extracted_clips")

    # Initialize (GPU auto-detected, faster-whisper enabled by default)
    extractor = VideoClipExtractor()
    
    # Build index (or load if exists)
    index_path = VIDEO_PATH.with_suffix('.vidx')
    
    if index_path.exists():
        print(f"Loading existing index: {index_path}")
        index = VideoIndex.load_index(index_path, extractor)
    else:
        print(f"Building index for: {VIDEO_PATH}")
        index = VideoIndex(VIDEO_PATH, extractor)
        index.build_index(verbose=True)
        index.save_index(index_path)
    
    # Extract clips
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    for query in QUERIES:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        matches = index.search(query, top_k=2)
        
        if not matches:
            print("  No matches found")
            continue
        
        for i, match in enumerate(matches):
            print(f"\n  Match {i+1}:")
            print(f"    Time: {match.start_time:.1f}s - {match.end_time:.1f}s")
            print(f"    Score: {match.score:.3f} ({match.match_type})")
            
            if match.matched_text:
                print(f"    Transcript: \"{match.matched_text[:50]}...\"")
            
            # Extract
            safe_name = "".join(c if c.isalnum() else "_" for c in query)[:30]
            output_path = OUTPUT_DIR / f"{safe_name}_{i}.mp4"
            
            extractor.extract_clip(
                VIDEO_PATH,
                output_path,
                match.start_time,
                match.end_time,
                padding=0.5
            )
            print(f"    Saved: {output_path}")


if __name__ == "__main__":
    main()
