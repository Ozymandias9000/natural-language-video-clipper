#!/usr/bin/env python3
"""
CLI for Video Clip Extractor

Usage:
    # Index a video and extract clips interactively
    python cli.py index video.mp4
    
    # Search an indexed video
    python cli.py search video.mp4 "person walking through door"
    
    # Extract clips from a list of descriptions
    python cli.py extract video.mp4 -q "sunset scene" -q "dialogue about money" -o clips/
    
    # One-shot: index + extract in one command
    python cli.py clip video.mp4 "car chase scene" -o clips/
"""

import argparse
from pathlib import Path
import sys

from extractor import VideoClipExtractor, VideoIndex, format_timestamp


def cmd_index(args):
    """Build index for a video."""
    extractor = VideoClipExtractor(
        clip_model=args.clip_model,
        whisper_model=args.whisper_model,
        device=args.device
    )
    
    index = VideoIndex(args.video, extractor)
    index.build_index(
        scene_threshold=args.threshold,
        transcribe=not args.no_audio,
        verbose=True
    )
    
    # Save index
    index_path = Path(args.video).with_suffix('.vidx')
    if args.index_path:
        index_path = Path(args.index_path)
    
    index.save_index(index_path)
    print(f"\nIndex saved to: {index_path}")
    
    # Print summary
    print(f"\nVideo indexed:")
    print(f"  Shots: {len(index.shots)}")
    print(f"  Transcript segments: {len(index.segments)}")
    
    return index


def cmd_search(args):
    """Search an indexed video."""
    extractor = VideoClipExtractor(
        clip_model=args.clip_model,
        whisper_model=args.whisper_model,
        device=args.device
    )
    
    # Load or build index
    index_path = Path(args.video).with_suffix('.vidx')
    if index_path.exists():
        print(f"Loading index from {index_path}")
        index = VideoIndex.load_index(index_path, extractor)
    else:
        print("No index found, building...")
        index = VideoIndex(args.video, extractor)
        index.build_index(verbose=True)
    
    # Search
    query = " ".join(args.query)
    print(f"\nSearching for: '{query}'")
    
    matches = index.search(
        query,
        top_k=args.top_k,
        visual_weight=args.visual_weight,
        audio_weight=1.0 - args.visual_weight
    )
    
    if not matches:
        print("No matches found.")
        return
    
    print(f"\nFound {len(matches)} matches:\n")
    for i, match in enumerate(matches, 1):
        print(f"  {i}. [{format_timestamp(match.start_time)} - {format_timestamp(match.end_time)}]")
        print(f"     Score: {match.score:.3f} ({match.match_type})")
        if match.matched_text:
            print(f"     Text: \"{match.matched_text[:80]}{'...' if len(match.matched_text) > 80 else ''}\"")
        print()
    
    return matches


def cmd_extract(args):
    """Extract clips based on queries."""
    extractor = VideoClipExtractor(
        clip_model=args.clip_model,
        whisper_model=args.whisper_model,
        device=args.device
    )
    
    # Load or build index
    index_path = Path(args.video).with_suffix('.vidx')
    if index_path.exists():
        print(f"Loading index from {index_path}")
        index = VideoIndex.load_index(index_path, extractor)
    else:
        print("No index found, building...")
        index = VideoIndex(args.video, extractor)
        index.build_index(verbose=True)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for query in args.queries:
        print(f"\nSearching for: '{query}'")
        matches = index.search(query, top_k=args.top_k)
        
        if not matches:
            print("  No matches found.")
            continue
        
        for i, match in enumerate(matches):
            safe_query = "".join(c if c.isalnum() else "_" for c in query)[:40]
            output_path = output_dir / f"{safe_query}_{i:02d}.mp4"
            
            print(f"  Extracting: {format_timestamp(match.start_time)} - {format_timestamp(match.end_time)}")
            print(f"    → {output_path}")
            
            extractor.extract_clip(
                Path(args.video),
                output_path,
                match.start_time,
                match.end_time,
                padding=args.padding,
                reencode=args.reencode
            )
    
    print(f"\nClips saved to: {output_dir}")


def cmd_clip(args):
    """One-shot: index and extract a single clip."""
    extractor = VideoClipExtractor(
        clip_model=args.clip_model,
        whisper_model=args.whisper_model,
        device=args.device
    )
    
    # Always build fresh index for one-shot
    index = VideoIndex(args.video, extractor)
    index.build_index(verbose=True)
    
    query = " ".join(args.query)
    print(f"\nSearching for: '{query}'")
    
    matches = index.search(query, top_k=args.top_k)
    
    if not matches:
        print("No matches found.")
        return
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, match in enumerate(matches):
        safe_query = "".join(c if c.isalnum() else "_" for c in query)[:40]
        output_path = output_dir / f"{safe_query}_{i:02d}.mp4"
        
        print(f"  Extracting: {format_timestamp(match.start_time)} - {format_timestamp(match.end_time)}")
        print(f"    Score: {match.score:.3f}")
        print(f"    → {output_path}")
        
        extractor.extract_clip(
            Path(args.video),
            output_path,
            match.start_time,
            match.end_time,
            padding=args.padding,
            reencode=args.reencode
        )


def cmd_interactive(args):
    """Interactive search session."""
    extractor = VideoClipExtractor(
        clip_model=args.clip_model,
        whisper_model=args.whisper_model,
        device=args.device
    )
    
    # Load or build index
    index_path = Path(args.video).with_suffix('.vidx')
    if index_path.exists():
        print(f"Loading index from {index_path}")
        index = VideoIndex.load_index(index_path, extractor)
    else:
        print("Building index...")
        index = VideoIndex(args.video, extractor)
        index.build_index(verbose=True)
        index.save_index(index_path)
    
    print(f"\nInteractive mode. Commands:")
    print("  <query>           - Search for clips")
    print("  :extract <n>      - Extract match number n")
    print("  :extractall       - Extract all last matches")
    print("  :quit             - Exit")
    print()
    
    last_matches = []
    output_dir = Path(args.output)
    
    while True:
        try:
            user_input = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.startswith(":quit"):
            print("Goodbye!")
            break
        
        if user_input.startswith(":extract"):
            if not last_matches:
                print("No matches to extract. Search first.")
                continue
            
            parts = user_input.split()
            if len(parts) > 1 and parts[1].isdigit():
                idx = int(parts[1]) - 1
                if 0 <= idx < len(last_matches):
                    matches_to_extract = [last_matches[idx]]
                else:
                    print(f"Invalid index. Choose 1-{len(last_matches)}")
                    continue
            else:
                matches_to_extract = last_matches
            
            output_dir.mkdir(parents=True, exist_ok=True)
            for i, match in enumerate(matches_to_extract):
                output_path = output_dir / f"clip_{i:02d}.mp4"
                extractor.extract_clip(
                    Path(args.video),
                    output_path,
                    match.start_time,
                    match.end_time
                )
                print(f"  Saved: {output_path}")
            continue
        
        # Regular search
        matches = index.search(user_input, top_k=5)
        last_matches = matches
        
        if not matches:
            print("No matches found.\n")
            continue
        
        for i, match in enumerate(matches, 1):
            print(f"  {i}. [{format_timestamp(match.start_time)} - {format_timestamp(match.end_time)}] "
                  f"score={match.score:.3f}")
            if match.matched_text:
                print(f"     \"{match.matched_text[:60]}{'...' if len(match.matched_text) > 60 else ''}\"")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Extract video clips using natural language descriptions"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for ML models (cpu, cuda, mps)"
    )
    parser.add_argument(
        "--clip-model", default="ViT-B/32",
        help="CLIP model variant"
    )
    parser.add_argument(
        "--whisper-model", default="base",
        help="Whisper model size (tiny, base, small, medium, large)"
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Index command
    p_index = subparsers.add_parser("index", help="Build search index for a video")
    p_index.add_argument("video", type=Path, help="Video file path")
    p_index.add_argument("--index-path", help="Custom path to save index")
    p_index.add_argument("--threshold", type=float, default=27.0,
                         help="Scene detection threshold (lower=more sensitive)")
    p_index.add_argument("--no-audio", action="store_true",
                         help="Skip audio transcription")
    p_index.set_defaults(func=cmd_index)
    
    # Search command
    p_search = subparsers.add_parser("search", help="Search indexed video")
    p_search.add_argument("video", type=Path, help="Video file path")
    p_search.add_argument("query", nargs="+", help="Search query")
    p_search.add_argument("-k", "--top-k", type=int, default=5,
                          help="Number of results")
    p_search.add_argument("--visual-weight", type=float, default=0.6,
                          help="Weight for visual vs audio (0-1)")
    p_search.set_defaults(func=cmd_search)
    
    # Extract command
    p_extract = subparsers.add_parser("extract", help="Extract clips from queries")
    p_extract.add_argument("video", type=Path, help="Video file path")
    p_extract.add_argument("-q", "--queries", action="append", required=True,
                           help="Query descriptions (can specify multiple)")
    p_extract.add_argument("-o", "--output", default="./clips",
                           help="Output directory")
    p_extract.add_argument("-k", "--top-k", type=int, default=1,
                           help="Clips per query")
    p_extract.add_argument("--padding", type=float, default=0.5,
                           help="Seconds to pad before/after clip")
    p_extract.add_argument("--reencode", action="store_true",
                           help="Re-encode for precise cuts (slower)")
    p_extract.set_defaults(func=cmd_extract)
    
    # One-shot clip command
    p_clip = subparsers.add_parser("clip", help="Index + extract in one shot")
    p_clip.add_argument("video", type=Path, help="Video file path")
    p_clip.add_argument("query", nargs="+", help="Clip description")
    p_clip.add_argument("-o", "--output", default="./clips",
                        help="Output directory")
    p_clip.add_argument("-k", "--top-k", type=int, default=1,
                        help="Number of clips to extract")
    p_clip.add_argument("--padding", type=float, default=0.5,
                        help="Seconds to pad before/after")
    p_clip.add_argument("--reencode", action="store_true",
                        help="Re-encode for precise cuts")
    p_clip.set_defaults(func=cmd_clip)
    
    # Interactive command
    p_interactive = subparsers.add_parser("interactive", help="Interactive search session")
    p_interactive.add_argument("video", type=Path, help="Video file path")
    p_interactive.add_argument("-o", "--output", default="./clips",
                               help="Output directory for extracts")
    p_interactive.set_defaults(func=cmd_interactive)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
