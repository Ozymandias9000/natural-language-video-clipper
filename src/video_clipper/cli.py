#!/usr/bin/env python3
"""
Video Clip Extractor CLI

Usage:
    ve movie.mp4 "explosion scene"           # Quick clip extraction (default)
    ve index movie.mp4                       # Build reusable index
    ve search movie.mp4 "car chase"          # Search indexed video
    ve extract movie.mp4 -q "chase" -q "explosion"  # Extract multiple clips
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from .index import VideoIndex
from .models import format_timestamp

console = Console()


def _create_index(args, video_path: Path) -> VideoIndex:
    """Create a VideoIndex with CLI args."""
    return VideoIndex(
        video_path,
        clip_model=args.clip_model,
        whisper_model=args.whisper_model,
        whisper_backend=args.whisper_backend,
        device=args.device,
        batch_size=args.batch_size,
    )


def _load_or_build_index(args) -> VideoIndex:
    """Load existing index or build new one."""
    index_path = args.video.with_suffix(".vidx")

    if index_path.exists():
        console.print(f"[dim]Loading index: {index_path}[/dim]")
        return VideoIndex.load(
            index_path,
            clip_model=args.clip_model,
            whisper_model=args.whisper_model,
            whisper_backend=args.whisper_backend,
            device=args.device,
            batch_size=args.batch_size,
        )

    console.print("[yellow]No index found, building...[/yellow]")
    index = _create_index(args, args.video)
    with console.status("[bold green]Building index..."):
        index.build(on_progress=lambda _: None)
        index.save(index_path)
    return index


def cmd_index(args):
    """Build index for a video with progress display."""
    index = _create_index(args, args.video)

    with console.status(f"[bold green]Indexing {args.video.name}...") as status:

        def on_progress(msg):
            status.update(msg)
            if msg.startswith("Found") or msg == "Done!":
                console.print(f"  [green]>[/green] {msg}")

        index.build(
            scene_threshold=args.threshold,
            transcribe=not args.no_audio,
            on_progress=on_progress,
        )

    index_path = Path(args.index_path) if args.index_path else args.video.with_suffix(".vidx")
    index.save(index_path)

    console.print(f"\n[bold green]Done![/bold green] Index saved to: {index_path}")
    console.print(f"[dim]Scenes: {len(index.shots)} | Transcript segments: {len(index.segments)}[/dim]")


def cmd_search(args):
    """Search an indexed video with nice output formatting."""
    index = _load_or_build_index(args)

    query = " ".join(args.query)
    console.print(f"\n[bold]Searching for:[/bold] '{query}'")

    matches = index.search(
        query,
        top_k=args.top_k,
        visual_weight=args.visual_weight,
        audio_weight=1.0 - args.visual_weight,
        full_scene=args.full_scene,
        expand=args.expand,
    )

    if not matches:
        console.print("[red]No matches found.[/red]")
        return

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
            match.match_type,
        )

    console.print(table)

    if args.show_text:
        for i, match in enumerate(matches, 1):
            if match.matched_text:
                text = match.matched_text[:80] + ("..." if len(match.matched_text) > 80 else "")
                console.print(f"  [dim]#{i}: \"{text}\"[/dim]")


def cmd_extract(args):
    """Extract clips with progress bar."""
    index = _load_or_build_index(args)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_clips = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting clips...", total=len(args.queries) * args.top_k)

        for query in args.queries:
            console.print(f"\n[bold]Query:[/bold] '{query}'")
            matches = index.search(query, top_k=args.top_k, full_scene=args.full_scene, expand=args.expand)

            if not matches:
                console.print("  [yellow]No matches found[/yellow]")
                progress.update(task, advance=args.top_k)
                continue

            for i, match in enumerate(matches):
                safe_name = "".join(c if c.isalnum() else "_" for c in query)[:40]
                suffix = "_scene" if args.full_scene else ""
                output_path = output_dir / f"{safe_name}{suffix}_{i:02d}.mp4"

                duration = match.end_time - match.start_time
                console.print(
                    f"  [green]>[/green] {format_timestamp(match.start_time)} - "
                    f"{format_timestamp(match.end_time)} ({duration:.1f}s)"
                )

                index.extract_clip(
                    output_path,
                    match.start_time,
                    match.end_time,
                    padding=args.padding,
                    reencode=args.reencode,
                )
                total_clips += 1
                progress.update(task, advance=1)

    console.print(f"\n[bold green]Done![/bold green] {total_clips} clips saved to: {output_dir}")


def cmd_clip(args):
    """One-shot: index and extract a single clip."""
    console.print(f"[bold]Processing:[/bold] {args.video.name}")

    index_path = args.video.with_suffix(".vidx")
    if index_path.exists():
        console.print(f"[dim]Loading existing index: {index_path}[/dim]")
        index = VideoIndex.load(
            index_path,
            clip_model=args.clip_model,
            whisper_model=args.whisper_model,
            whisper_backend=args.whisper_backend,
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        index = _create_index(args, args.video)
        with console.status("[bold green]Building index...") as status:

            def on_progress(msg):
                status.update(msg)

            index.build(transcribe=not args.no_audio, on_progress=on_progress)
        index.save(index_path)
        console.print(f"[dim]Index saved: {index_path}[/dim]")

    query = " ".join(args.query)
    console.print(f"\n[bold]Searching for:[/bold] '{query}'")

    matches = index.search(query, top_k=args.top_k, full_scene=args.full_scene, expand=args.expand)

    if not matches:
        console.print("[red]No matches found.[/red]")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = "".join(c if c.isalnum() else "_" for c in query)[:40]

    if args.stitch:
        # Stitch all matches into one file
        suffix = "_scene" if args.full_scene else ""
        output_path = output_dir / f"{safe_name}{suffix}_stitched.mp4"

        console.print(f"\n[bold]Stitching {len(matches)} clips:[/bold]")
        sorted_matches = sorted(matches, key=lambda m: m.start_time)
        for i, match in enumerate(sorted_matches):
            duration = match.end_time - match.start_time
            console.print(
                f"  [{i + 1}/{len(matches)}] {format_timestamp(match.start_time)} - "
                f"{format_timestamp(match.end_time)} ({duration:.1f}s, score: {match.score:.2f})"
            )

        with console.status("[bold green]Stitching clips..."):
            index.stitch_clips(
                output_path,
                matches,
                padding=args.padding,
                reencode=args.reencode,
            )

        console.print(f"\n[bold green]Done![/bold green] Stitched clip saved to: {output_path}")
    else:
        # Extract individual clips
        console.print(f"\n[bold]Extracting {len(matches)} clips:[/bold]")

        for i, match in enumerate(matches):
            suffix = "_scene" if args.full_scene else ""
            output_path = output_dir / f"{safe_name}{suffix}_{i:02d}.mp4"

            duration = match.end_time - match.start_time
            console.print(
                f"  [{i + 1}/{len(matches)}] {format_timestamp(match.start_time)} - "
                f"{format_timestamp(match.end_time)} ({duration:.1f}s) > {output_path.name}"
            )

            index.extract_clip(
                output_path,
                match.start_time,
                match.end_time,
                padding=args.padding,
                reencode=args.reencode,
            )

        console.print(f"\n[bold green]Done![/bold green] Clips saved to: {output_dir}")


def _add_global_options(parser):
    """Add global options shared by all commands."""
    parser.add_argument("--device", default=None, help="Compute device (cuda, mps, cpu; auto-detected)")
    parser.add_argument("--clip-model", default="ViT-B/32", help="CLIP model (ViT-B/32, ViT-L/14)")
    parser.add_argument("--whisper-model", default="base", help="Whisper model (tiny, base, small, medium, large)")
    parser.add_argument(
        "--whisper-backend",
        default="faster-whisper",
        choices=["faster-whisper", "whisper"],
        help="Whisper backend",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")


def main():
    parser = argparse.ArgumentParser(
        prog="ve",
        description="Extract video clips using natural language descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ve movie.mp4 "explosion scene"              # Quick extract
  ve movie.mp4 "car chase" --full-scene       # Extract full scene
  ve movie.mp4 "funny moment" -k 3            # Extract top 3 matches

  ve index movie.mp4                          # Build reusable index
  ve search movie.mp4 "sunset"                # Search indexed video
  ve extract movie.mp4 -q "chase" -q "fight"  # Multiple queries
        """,
    )

    _add_global_options(parser)

    subparsers = parser.add_subparsers(dest="command")

    # index
    p_index = subparsers.add_parser("index", help="Build search index")
    p_index.add_argument("video", type=Path, help="Video file")
    p_index.add_argument("--index-path", help="Custom index output path")
    p_index.add_argument("--threshold", type=float, default=27.0, help="Scene detection threshold")
    p_index.add_argument("--no-audio", action="store_true", help="Skip transcription")
    _add_global_options(p_index)
    p_index.set_defaults(func=cmd_index)

    # search
    p_search = subparsers.add_parser("search", help="Search indexed video")
    p_search.add_argument("video", type=Path, help="Video file")
    p_search.add_argument("query", nargs="+", help="Search query")
    p_search.add_argument("-k", "--top-k", type=int, default=5, help="Number of results")
    p_search.add_argument("--visual-weight", type=float, default=0.6, help="Visual vs audio weight (0-1)")
    p_search.add_argument("--full-scene", action="store_true", help="Return full scenes")
    p_search.add_argument("--show-text", action="store_true", help="Show matched transcript")
    p_search.add_argument("--expand", action="store_true", help="Use LLM to expand query (requires ANTHROPIC_API_KEY)")
    _add_global_options(p_search)
    p_search.set_defaults(func=cmd_search)

    # extract
    p_extract = subparsers.add_parser("extract", help="Extract clips from queries")
    p_extract.add_argument("video", type=Path, help="Video file")
    p_extract.add_argument("-q", "--queries", action="append", required=True, help="Query (repeatable)")
    p_extract.add_argument("-o", "--output", default="./clips", help="Output directory")
    p_extract.add_argument("-k", "--top-k", type=int, default=1, help="Clips per query")
    p_extract.add_argument("--full-scene", action="store_true", help="Extract full scenes")
    p_extract.add_argument("--padding", type=float, default=0.5, help="Padding seconds")
    p_extract.add_argument("--reencode", action="store_true", help="Re-encode for precise cuts")
    p_extract.add_argument("--expand", action="store_true", help="Use LLM to expand query (requires ANTHROPIC_API_KEY)")
    _add_global_options(p_extract)
    p_extract.set_defaults(func=cmd_extract)

    # clip (one-shot)
    p_clip = subparsers.add_parser("clip", help="Quick extract (index + extract)")
    p_clip.add_argument("video", type=Path, help="Video file")
    p_clip.add_argument("query", nargs="+", help="Clip description")
    p_clip.add_argument("-o", "--output", default="./clips", help="Output directory")
    p_clip.add_argument("-k", "--top-k", type=int, default=5, help="Number of clips")
    p_clip.add_argument("--full-scene", action="store_true", help="Extract full scenes")
    p_clip.add_argument("--padding", type=float, default=0.5, help="Padding seconds")
    p_clip.add_argument("--reencode", action="store_true", help="Re-encode for precise cuts")
    p_clip.add_argument("--no-audio", action="store_true", help="Skip transcription")
    p_clip.add_argument("--stitch", action="store_true", help="Stitch top-k matches into one file")
    p_clip.add_argument("--expand", action="store_true", help="Use LLM to expand query (requires ANTHROPIC_API_KEY)")
    _add_global_options(p_clip)
    p_clip.set_defaults(func=cmd_clip)

    # Handle default command: if first arg looks like a video file, use clip
    args = sys.argv[1:]
    if args and not args[0].startswith("-"):
        first_arg = args[0]
        video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
        if first_arg not in {"index", "search", "extract", "clip"} and Path(first_arg).suffix.lower() in video_extensions:
            args = ["clip"] + args

    parsed = parser.parse_args(args)

    if not hasattr(parsed, "func"):
        parser.print_help()
        sys.exit(1)

    try:
        parsed.func(parsed)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
