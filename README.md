# Video Clip Extractor

Extract video clips using natural language descriptions. Fully offline, no API costs.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT VIDEO                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │ PySceneDetect│ │   Whisper    │ │   ffprobe    │
        │ (shots)      │ │ (transcript) │ │ (metadata)   │
        └──────────────┘ └──────────────┘ └──────────────┘
                │               │
                ▼               ▼
        ┌──────────────┐ ┌──────────────┐
        │  Keyframes   │ │  Timestamped │
        │  (1 per shot)│ │  Segments    │
        └──────────────┘ └──────────────┘
                │               │
                ▼               ▼
        ┌──────────────┐ ┌──────────────┐
        │ CLIP Visual  │ │ CLIP Text    │
        │ Embeddings   │ │ Embeddings   │
        └──────────────┘ └──────────────┘
                │               │
                └───────┬───────┘
                        ▼
                ┌──────────────┐
                │ Temporal     │
                │ Index        │  ← Saved to disk (.vidx)
                └──────────────┘
                        │
                        ▼  (query)
                ┌──────────────┐
                │ Vector       │
                │ Search       │
                └──────────────┘
                        │
                        ▼
                ┌──────────────┐
                │   ffmpeg     │
                │   (cut)      │
                └──────────────┘
                        │
                        ▼
                ┌──────────────┐
                │ OUTPUT CLIPS │
                └──────────────┘
```

## Installation

```bash
# System dependencies (Ubuntu/Debian)
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Python dependencies
pip install -r requirements.txt
```

For GPU acceleration:
```bash
# CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Apple Silicon
pip install torch torchvision  # MPS support built-in
```

## Usage

### CLI

```bash
# One-shot: find and extract clips matching description
python cli.py clip movie.mp4 "explosion scene" -o clips/

# Build reusable index (for repeated queries)
python cli.py index movie.mp4

# Search indexed video
python cli.py search movie.mp4 "two people talking in kitchen"

# Extract multiple clips
python cli.py extract movie.mp4 \
    -q "car chase" \
    -q "romantic dialogue" \
    -q "sunset establishing shot" \
    -o clips/

# Interactive mode
python cli.py interactive movie.mp4
```

### Python API

```python
from pathlib import Path
from extractor import VideoClipExtractor, VideoIndex

# Initialize
extractor = VideoClipExtractor(
    clip_model="ViT-B/32",     # or ViT-L/14 for better quality
    whisper_model="base",      # tiny/base/small/medium/large
    device="cuda"              # or "cpu" or "mps"
)

# Build index
video = Path("movie.mp4")
index = VideoIndex(video, extractor)
index.build_index()

# Save for later
index.save_index(video.with_suffix('.vidx'))

# Search
matches = index.search("person walking through a door", top_k=5)

for match in matches:
    print(f"{match.start_time:.1f}s - {match.end_time:.1f}s (score: {match.score:.2f})")
    
    # Extract clip
    extractor.extract_clip(
        video,
        Path(f"clip_{match.start_time:.0f}.mp4"),
        match.start_time,
        match.end_time
    )
```

### Tuning Search

```python
# Favor visual matches (good for action, scenery)
matches = index.search("mountain landscape", visual_weight=0.9, audio_weight=0.1)

# Favor audio matches (good for dialogue, specific words)
matches = index.search("mentions the money", visual_weight=0.2, audio_weight=0.8)

# Scene detection sensitivity
index.build_index(scene_threshold=20.0)  # Lower = more scenes detected
```

## How It Works

### 1. Scene Detection
PySceneDetect analyzes frame-to-frame differences to find shot boundaries. This gives us natural semantic units (shots) rather than arbitrary time windows.

### 2. Keyframe Extraction  
One representative frame per shot (at the midpoint). This is what gets embedded.

### 3. CLIP Embeddings
CLIP maps both images and text into the same embedding space. A photo of a dog and the text "a dog" will have similar embeddings. We embed:
- Each keyframe → visual understanding of each shot
- Each transcript segment → what's being said

### 4. Whisper Transcription
Full audio transcription with timestamps. Each segment gets a text embedding too.

### 5. Vector Search
Your query gets embedded with CLIP's text encoder. We find shots/segments with the highest cosine similarity to your query, combining visual and audio signals.

### 6. ffmpeg Cutting
Matched timestamps get cut out. Use `-c copy` for fast keyframe-aligned cuts, or re-encode for frame-accurate cuts.

## Limitations

- **Motion understanding**: CLIP sees individual frames, not motion. "Person running" works via pose, but "ball bouncing" might miss.
- **Audio events**: Non-speech audio (music, sound effects) isn't captured. Only transcribed speech.
- **Very short clips**: Shots under ~0.5s might get merged or missed.
- **Ambiguous queries**: "The good part" won't work. Be specific.

## Performance

| Video Length | Index Time (CPU) | Index Time (GPU) | Index Size |
|--------------|------------------|------------------|------------|
| 10 min       | ~3 min           | ~45 sec          | ~5 MB      |
| 1 hour       | ~15 min          | ~4 min           | ~30 MB     |
| 2 hours      | ~30 min          | ~8 min           | ~60 MB     |

Search is instant (<100ms) once indexed.

## Alternatives Considered

| Approach | Pros | Cons |
|----------|------|------|
| This (CLIP + Whisper) | Offline, free, good quality | No motion, requires index |
| Gemini 1.5 Pro | Direct video understanding, motion-aware | API cost, latency, 1hr limit |
| Twelve Labs | Best quality, built for this | Expensive, cloud-only |
| Frame-by-frame VLM | Maximum detail | Extremely slow/expensive |

This approach hits a good tradeoff for local/offline use.
