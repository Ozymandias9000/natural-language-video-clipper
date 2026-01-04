#!/usr/bin/env python3
"""
Enhanced CLI for Video Clip Extractor
Now with full scene extraction and better UX
"""

import argparse
from pathlib import Path
import sys
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich import print as rprint

from extractor_enhanced import EnhancedVideoClipExtractor, EnhancedVideoIndex, format_timestamp

console = Console()


def cmd_index(args):
    """Build index for a video with progress display."""
    extractor = EnhancedVideoClipExtractor(
        clip_model=args.clip_model,
        whisper_model=args.whisper_model,
        device=args.device
    )
    
    index = EnhancedVideoIndex(args.video, extractor)
    
    with console.status(f"[bold green]Indexing {args.video.name}...") as status:
        status.update("Detecting scenes...")
        index.shots = extractor.detect_scenes(args.video, threshold=args.threshold)
        console.print(f"✓ Found [cyan]{len(index.shots)}[/cyan] scenes")
        
        status.update("Extracting keyframes...")
        extractor.extract_keyframes(args.video, index.shots)
        console.print("✓ Keyframes extracted")
        
        status.update("Computing visual embeddings...")
        extractor.compute_visual_embeddings(index.shots)
        console.print("✓ Visual embeddings computed")
        
        if not args.no_audio:
            status.update("Transcribing audio (this may take a while)...")
            index.segments = extractor.transcribe_audio(args.video)
            console.print(f"✓ Found [cyan]{len(index.segments)}[/cyan] transcript segments")
            
            status.update("Computing text embeddings...")
            extractor.compute_text_embeddings(index.segments)
            extractor.assign_transcript_to_shots(index.shots, index.segments)
            console.print("✓ Text embeddings computed")
    
    index._indexed = True
    
    # Save index
    index_path = Path(args.index_path) if args.index_path else Path(args.video).with_suffix('.vidx')
    index.save_index(index_path)
    
    console.print(f"\n[bold green]✓ Index saved to:[/bold green] {index_path}")
    console.print(f"[dim]Scenes: {len(index.shots)} | Transcript segments: {len(index.segments)}[/dim]")
    
    return index


def cmd_search(args):
    """Search an indexed video with nice output formatting."""
    extractor = EnhancedVideoClipExtractor(
        clip_model=args.clip_model,
        whisper_model=args.whisper_model,
        device=args.device
    )
    
    # Load or build index
    index_path = Path(args.video).with_suffix('.vidx')
    if index_path.exists():
        console.print(f"[dim]Loading index from {index_path}[/dim]")
        index = EnhancedVideoIndex.load_index(index_path, extractor)
    else:
        console.print("[yellow]No index found, building...[/yellow]")
        index = EnhancedVideoIndex(args.video, extractor)
        with console.status("[bold green]Building index..."):
            index.build_index(verbose=False)
            index.save_index(index_path)
    
    # Search
    query = " ".join(args.query)
    console.print(f"\n[bold]Searching for:[/bold] '{query}'")
    
    matches = index.search(
        query,
        top_k=args.top_k,
        visual_weight=args.visual_weight,
        audio_weight=1.0 - args.visual_weight,
        full_scene=args.full_scene
    )
    
    if not matches:
        console.print("[red]No matches found.[/red]")
        return
    
    # Display results in a nice table
    table = Table(title=f"Found {len(matches)} matches")
    table.add_column("#", style="cyan", width=3)
    table.add_column("Time Range", style="green")
    table.add_column("Duration", style="yellow")
    table.add_column("Score", style="magenta")
    table.add_column("Type", style="blue")
    
    for i, match in enumerate(matches, 1):
        duration = match.end_time - match.start_time
        table.add_row(
            str(i),
            f"{format_timestamp(match.start_time)} - {format_timestamp(match.end_time)}",
            f"{duration:.1f}s",
            f"{match.score:.3f}",
            match.match_type
        )
        if match.matched_text and args.show_text:
            console.print(f"    [dim]Text: \"{match.matched_text[:80]}{'...' if len(match.matched_text) > 80 else ''}\"[/dim]")
    
    console.print(table)
    return matches


def cmd_extract(args):
    """Extract clips with progress bar."""
    extractor = EnhancedVideoClipExtractor(
        clip_model=args.clip_model,
        whisper_model=args.whisper_model,
        device=args.device
    )
    
    # Load or build index
    index_path = Path(args.video).with_suffix('.vidx')
    if index_path.exists():
        console.print(f"[dim]Loading index from {index_path}[/dim]")
        index = EnhancedVideoIndex.load_index(index_path, extractor)
    else:
        console.print("[yellow]No index found, building...[/yellow]")
        index = EnhancedVideoIndex(args.video, extractor)
        with console.status("[bold green]Building index..."):
            index.build_index(verbose=False)
            index.save_index(index_path)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_clips = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        
        task = progress.add_task(f"Extracting clips...", total=len(args.queries) * args.top_k)
        
        for query in args.queries:
            console.print(f"\n[bold]Query:[/bold] '{query}'")
            matches = index.search(query, top_k=args.top_k, full_scene=args.full_scene)
            
            if not matches:
                console.print("  [yellow]No matches found[/yellow]")
                progress.update(task, advance=args.top_k)
                continue
            
            for i, match in enumerate(matches):
                safe_query = "".join(c if c.isalnum() else "_" for c in query)[:40]
                suffix = "_scene" if args.full_scene else ""
                output_path = output_dir / f"{safe_query}{suffix}_{i:02d}.mp4"
                
                duration = match.end_time - match.start_time
                console.print(f"  [green]→[/green] {format_timestamp(match.start_time)} - {format_timestamp(match.end_time)} ({duration:.1f}s)")
                
                extractor.extract_clip(
                    Path(args.video),
                    output_path,
                    match.start_time,
                    match.end_time,
                    padding=args.padding,
                    reencode=args.reencode
                )
                total_clips += 1
                progress.update(task, advance=1)
    
    console.print(f"\n[bold green]✓ {total_clips} clips saved to:[/bold green] {output_dir}")


def cmd_clip(args):
    """One-shot: index and extract a single clip."""
    extractor = EnhancedVideoClipExtractor(
        clip_model=args.clip_model,
        whisper_model=args.whisper_model,
        device=args.device
    )
    
    # Build index
    console.print(f"[bold]Processing:[/bold] {args.video.name}")
    index = EnhancedVideoIndex(args.video, extractor)
    
    with console.status("[bold green]Building index...") as status:
        status.update("Detecting scenes...")
        index.shots = extractor.detect_scenes(args.video)
        
        status.update("Processing visuals...")
        extractor.extract_keyframes(args.video, index.shots)
        extractor.compute_visual_embeddings(index.shots)
        
        if not args.no_audio:
            status.update("Transcribing audio...")
            index.segments = extractor.transcribe_audio(args.video)
            extractor.compute_text_embeddings(index.segments)
            extractor.assign_transcript_to_shots(index.shots, index.segments)
        
        index._indexed = True
    
    query = " ".join(args.query)
    console.print(f"\n[bold]Searching for:[/bold] '{query}'")
    
    matches = index.search(query, top_k=args.top_k, full_scene=args.full_scene)
    
    if not matches:
        console.print("[red]No matches found.[/red]")
        return
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"\n[bold]Extracting {len(matches)} clips:[/bold]")
    for i, match in enumerate(matches):
        safe_query = "".join(c if c.isalnum() else "_" for c in query)[:40]
        suffix = "_scene" if args.full_scene else ""
        output_path = output_dir / f"{safe_query}{suffix}_{i:02d}.mp4"
        
        duration = match.end_time - match.start_time
        console.print(f"  [{i+1}/{len(matches)}] {format_timestamp(match.start_time)} - {format_timestamp(match.end_time)} ({duration:.1f}s) → {output_path.name}")
        
        extractor.extract_clip(
            Path(args.video),
            output_path,
            match.start_time,
            match.end_time,
            padding=args.padding,
            reencode=args.reencode
        )
    
    console.print(f"\n[bold green]✓ Clips saved to:[/bold green] {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract video clips using natural language descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s clip video.mp4 "person laughing" --full-scene
  %(prog)s index video.mp4 --threshold 20
  %(prog)s search video.mp4 "sunset" --show-text
  %(prog)s extract video.mp4 -q "car chase" -q "explosion" --full-scene
        """
    )
    
    # Global options
    parser.add_argument(
        "--device", default="cpu",
        help="Device for ML models (cpu, cuda, mps)"
    )
    parser.add_argument(
        "--clip-model", default="ViT-B/32",
        help="CLIP model variant (ViT-B/32, ViT-L/14)"
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
                         help="Scene detection threshold (lower=more scenes)")
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
    p_search.add_argument("--full-scene", action="store_true",
                          help="Return full scenes containing matches")
    p_search.add_argument("--show-text", action="store_true",
                          help="Show matched transcript text")
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
    p_extract.add_argument("--full-scene", action="store_true",
                           help="Extract full scenes containing matches")
    p_extract.add_argument("--padding", type=float, default=0.5,
                           help="Seconds to pad clips (ignored with --full-scene)")
    p_extract.add_argument("--reencode", action="store_true",
                           help="Re-encode for precise cuts (slower)")
    p_extract.set_defaults(func=cmd_extract)
    
    # One-shot clip command
    p_clip = subparsers.add_parser("clip", help="Quick extract (index + extract)")
    p_clip.add_argument("video", type=Path, help="Video file path")
    p_clip.add_argument("query", nargs="+", help="Clip description")
    p_clip.add_argument("-o", "--output", default="./clips",
                        help="Output directory")
    p_clip.add_argument("-k", "--top-k", type=int, default=1,
                        help="Number of clips to extract")
    p_clip.add_argument("--full-scene", action="store_true",
                        help="Extract full scenes containing matches")
    p_clip.add_argument("--padding", type=float, default=0.5,
                        help="Seconds to pad clips (ignored with --full-scene)")
    p_clip.add_argument("--reencode", action="store_true",
                        help="Re-encode for precise cuts")
    p_clip.add_argument("--no-audio", action="store_true",
                        help="Skip audio transcription")
    p_clip.set_defaults(func=cmd_clip)
    
    args = parser.parse_args()
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()