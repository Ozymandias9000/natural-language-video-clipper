# Video Clipper UI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a web-based UI for viewing, searching, and exporting video clips with a scrubabble thumbnail timeline.

**Architecture:** Bun/Elysia server communicates with Python via PyBridge. React 19 frontend with Vite. Dark cinematic theme with amber accents.

**Tech Stack:** Bun, Elysia, PyBridge, React 19, Vite, Tailwind CSS, Motion (Framer Motion)

---

## Task 1: Initialize UI Project Structure

**Files:**
- Create: `ui/package.json`
- Create: `ui/tsconfig.json`
- Create: `ui/vite.config.ts`
- Create: `ui/tailwind.config.js`
- Create: `ui/postcss.config.js`
- Create: `ui/index.html`

**Step 1: Create package.json**

```json
{
  "name": "video-clipper-ui",
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "bun run --watch src/server/index.ts",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "@elysiajs/cors": "^1.0.0",
    "@elysiajs/static": "^1.0.0",
    "elysia": "^1.0.0",
    "motion": "^11.0.0",
    "pybridge": "^1.0.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0"
  },
  "devDependencies": {
    "@types/bun": "latest",
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "@vitejs/plugin-react": "^4.0.0",
    "autoprefixer": "^10.0.0",
    "postcss": "^8.0.0",
    "tailwindcss": "^3.4.0",
    "typescript": "^5.0.0",
    "vite": "^5.0.0"
  }
}
```

**Step 2: Create tsconfig.json**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "types": ["bun-types"],
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

**Step 3: Create vite.config.ts**

```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  root: ".",
  build: {
    outDir: "dist/client",
    emptyDirOnStart: true,
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      "/api": "http://localhost:3000",
    },
  },
});
```

**Step 4: Create tailwind.config.js**

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        charcoal: {
          900: "#0a0a0a",
          800: "#141414",
          700: "#1f1f1f",
          600: "#2a2a2a",
        },
        amber: {
          400: "#fbbf24",
          500: "#f59e0b",
          600: "#d97706",
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "monospace"],
        sans: ["Instrument Sans", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
```

**Step 5: Create postcss.config.js**

```javascript
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
```

**Step 6: Create index.html**

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Video Clipper</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link
      href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
      rel="stylesheet"
    />
  </head>
  <body class="bg-charcoal-900 text-white">
    <div id="root"></div>
    <script type="module" src="/src/client/main.tsx"></script>
  </body>
</html>
```

**Step 7: Install dependencies**

Run: `cd ui && bun install`
Expected: Dependencies installed successfully

**Step 8: Commit**

```bash
git add ui/
git commit -m "feat(ui): initialize project with Bun, Elysia, React 19, Vite, Tailwind"
```

---

## Task 2: Create Python Bridge Module

**Files:**
- Create: `src/video_clipper/bridge.py`

**Step 1: Create bridge.py with PyBridge-compatible functions**

```python
"""PyBridge API for the video clipper UI.

This module exposes video clipper functionality to the Bun/Elysia server
via PyBridge. Functions are called from TypeScript and return JSON-serializable data.
"""

import base64
import sys
from pathlib import Path
from typing import Optional

# Global state - keeps models loaded between calls
_index: Optional["VideoIndex"] = None
_video_path: Optional[Path] = None


def load_video(path: str) -> dict:
    """
    Load a video file and return metadata.

    Args:
        path: Absolute path to video file

    Returns:
        Video metadata: duration, fps, width, height
    """
    from . import video
    from .index import VideoIndex

    global _index, _video_path

    video_path = Path(path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    _video_path = video_path
    duration = video.get_duration(video_path)

    # Get dimensions via ffprobe
    import subprocess
    import json

    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]

    # Parse frame rate (e.g., "30000/1001" -> 29.97)
    fps_parts = stream["r_frame_rate"].split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])

    return {
        "duration": duration,
        "fps": fps,
        "width": stream["width"],
        "height": stream["height"],
        "path": str(video_path),
    }


def build_index(transcribe: bool = True) -> dict:
    """
    Build search index for the loaded video.

    Args:
        transcribe: Whether to transcribe audio

    Returns:
        Index status and stats
    """
    from .index import VideoIndex

    global _index, _video_path

    if _video_path is None:
        raise RuntimeError("No video loaded. Call load_video first.")

    index_path = _video_path.with_suffix(".vidx")

    # Try to load existing index
    if index_path.exists():
        _index = VideoIndex.load(index_path)
        return {
            "status": "loaded",
            "shots": len(_index.shots),
            "segments": len(_index.segments),
        }

    # Build new index
    _index = VideoIndex(_video_path)
    _index.build(transcribe=transcribe)
    _index.save(index_path)

    return {
        "status": "built",
        "shots": len(_index.shots),
        "segments": len(_index.segments),
    }


def get_index_status() -> dict:
    """Check if index is ready."""
    global _index
    return {
        "ready": _index is not None,
        "shots": len(_index.shots) if _index else 0,
        "segments": len(_index.segments) if _index else 0,
    }


def search(query: str, top_k: int = 5) -> list[dict]:
    """
    Search for clips matching query.

    Args:
        query: Natural language description
        top_k: Number of results

    Returns:
        List of matches with start, end, score, match_type
    """
    global _index

    if _index is None:
        raise RuntimeError("No index loaded. Call build_index first.")

    matches = _index.search(query, top_k=top_k)

    return [
        {
            "start": m.start_time,
            "end": m.end_time,
            "score": m.score,
            "match_type": m.match_type,
            "matched_text": m.matched_text,
        }
        for m in matches
    ]


def get_thumbnail(time: float, width: int = 160) -> str:
    """
    Get a thumbnail at the given timestamp as base64.

    Args:
        time: Timestamp in seconds
        width: Thumbnail width (height auto-scaled)

    Returns:
        Base64-encoded JPEG image
    """
    import subprocess
    import tempfile

    global _video_path

    if _video_path is None:
        raise RuntimeError("No video loaded. Call load_video first.")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        temp_path = f.name

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(time),
            "-i", str(_video_path),
            "-vframes", "1",
            "-vf", f"scale={width}:-1",
            "-q:v", "3",
            temp_path,
        ],
        capture_output=True,
        check=True,
    )

    with open(temp_path, "rb") as f:
        data = f.read()

    Path(temp_path).unlink()

    return base64.b64encode(data).decode("utf-8")


def get_thumbnails_batch(times: list[float], width: int = 160) -> list[str]:
    """
    Get multiple thumbnails efficiently.

    Args:
        times: List of timestamps in seconds
        width: Thumbnail width

    Returns:
        List of base64-encoded JPEG images
    """
    return [get_thumbnail(t, width) for t in times]


def export_clips(
    clips: list[dict],
    output_dir: str,
    stitch: bool = False,
) -> dict:
    """
    Export selected clips.

    Args:
        clips: List of {start, end} dicts
        output_dir: Output directory path
        stitch: If True, stitch all clips into one file

    Returns:
        Export result with output paths
    """
    from .models import ClipMatch

    global _index, _video_path

    if _index is None or _video_path is None:
        raise RuntimeError("No index loaded.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if stitch:
        # Stitch into one file
        out_file = output_path / f"{_video_path.stem}_stitched.mp4"
        matches = [
            ClipMatch(start_time=c["start"], end_time=c["end"], score=0, match_type="manual")
            for c in clips
        ]
        _index.stitch_clips(out_file, matches)
        return {"outputs": [str(out_file)]}
    else:
        # Export individual clips
        outputs = []
        for i, clip in enumerate(clips):
            out_file = output_path / f"{_video_path.stem}_clip_{i:02d}.mp4"
            _index.extract_clip(out_file, clip["start"], clip["end"])
            outputs.append(str(out_file))
        return {"outputs": outputs}


# Entry point for PyBridge
if __name__ == "__main__":
    # PyBridge handles stdin/stdout communication
    pass
```

**Step 2: Verify module imports work**

Run: `cd /Users/nicholasmurphy/Documents/Code/video-clipper && python -c "from video_clipper.bridge import load_video; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/video_clipper/bridge.py
git commit -m "feat(bridge): add PyBridge API module for UI communication"
```

---

## Task 3: Create Elysia Server

**Files:**
- Create: `ui/src/server/index.ts`
- Create: `ui/src/server/python.ts`
- Create: `ui/src/server/routes/video.ts`
- Create: `ui/src/server/routes/search.ts`
- Create: `ui/src/server/routes/export.ts`

**Step 1: Create PyBridge wrapper (python.ts)**

```typescript
import { PyBridge } from "pybridge";
import { join } from "path";

// Project root (one level up from ui/)
const projectRoot = join(import.meta.dir, "../../../");

const bridge = new PyBridge({
  python: "python3",
  cwd: projectRoot,
});

interface VideoInfo {
  duration: number;
  fps: number;
  width: number;
  height: number;
  path: string;
}

interface IndexStatus {
  ready: boolean;
  shots: number;
  segments: number;
}

interface BuildResult {
  status: string;
  shots: number;
  segments: number;
}

interface ClipMatch {
  start: number;
  end: number;
  score: number;
  match_type: string;
  matched_text: string | null;
}

interface ExportResult {
  outputs: string[];
}

interface VideoClipperAPI {
  load_video(path: string): VideoInfo;
  build_index(transcribe?: boolean): BuildResult;
  get_index_status(): IndexStatus;
  search(query: string, top_k?: number): ClipMatch[];
  get_thumbnail(time: number, width?: number): string;
  get_thumbnails_batch(times: number[], width?: number): string[];
  export_clips(
    clips: { start: number; end: number }[],
    output_dir: string,
    stitch?: boolean
  ): ExportResult;
}

export const clipper = bridge.controller<VideoClipperAPI>(
  "src/video_clipper/bridge.py"
);

export type { VideoInfo, IndexStatus, BuildResult, ClipMatch, ExportResult };
```

**Step 2: Create video routes (routes/video.ts)**

```typescript
import { Elysia, t } from "elysia";
import { clipper } from "../python";

export const videoRoutes = new Elysia({ prefix: "/api/video" })
  .post(
    "/load",
    async ({ body }) => {
      const info = await clipper.load_video(body.path);
      return info;
    },
    {
      body: t.Object({
        path: t.String(),
      }),
    }
  )
  .get("/thumbnail", async ({ query }) => {
    const time = parseFloat(query.t || "0");
    const width = parseInt(query.w || "160", 10);
    const base64 = await clipper.get_thumbnail(time, width);
    return { data: base64 };
  })
  .post(
    "/thumbnails",
    async ({ body }) => {
      const thumbnails = await clipper.get_thumbnails_batch(
        body.times,
        body.width || 160
      );
      return { thumbnails };
    },
    {
      body: t.Object({
        times: t.Array(t.Number()),
        width: t.Optional(t.Number()),
      }),
    }
  );
```

**Step 3: Create search routes (routes/search.ts)**

```typescript
import { Elysia, t } from "elysia";
import { clipper } from "../python";

export const searchRoutes = new Elysia({ prefix: "/api" })
  .post(
    "/index/build",
    async ({ body }) => {
      const result = await clipper.build_index(body.transcribe ?? true);
      return result;
    },
    {
      body: t.Object({
        transcribe: t.Optional(t.Boolean()),
      }),
    }
  )
  .get("/index/status", async () => {
    return await clipper.get_index_status();
  })
  .post(
    "/search",
    async ({ body }) => {
      const matches = await clipper.search(body.query, body.top_k || 5);
      return { matches };
    },
    {
      body: t.Object({
        query: t.String(),
        top_k: t.Optional(t.Number()),
      }),
    }
  );
```

**Step 4: Create export routes (routes/export.ts)**

```typescript
import { Elysia, t } from "elysia";
import { clipper } from "../python";

export const exportRoutes = new Elysia({ prefix: "/api/export" }).post(
  "/",
  async ({ body }) => {
    const result = await clipper.export_clips(
      body.clips,
      body.output_dir || "./clips",
      body.stitch || false
    );
    return result;
  },
  {
    body: t.Object({
      clips: t.Array(
        t.Object({
          start: t.Number(),
          end: t.Number(),
        })
      ),
      output_dir: t.Optional(t.String()),
      stitch: t.Optional(t.Boolean()),
    }),
  }
);
```

**Step 5: Create main server entry (index.ts)**

```typescript
import { Elysia } from "elysia";
import { cors } from "@elysiajs/cors";
import { staticPlugin } from "@elysiajs/static";
import { videoRoutes } from "./routes/video";
import { searchRoutes } from "./routes/search";
import { exportRoutes } from "./routes/export";

const app = new Elysia()
  .use(cors())
  .use(videoRoutes)
  .use(searchRoutes)
  .use(exportRoutes)
  .use(
    staticPlugin({
      assets: "dist/client",
      prefix: "/",
    })
  )
  .get("/health", () => ({ status: "ok" }))
  .listen(3000);

console.log(`Server running at http://localhost:${app.server?.port}`);

export type App = typeof app;
```

**Step 6: Verify server compiles**

Run: `cd ui && bun run --dry-run src/server/index.ts`
Expected: No TypeScript errors

**Step 7: Commit**

```bash
git add ui/src/server/
git commit -m "feat(server): add Elysia server with PyBridge routes"
```

---

## Task 4: Create React App Shell

**Files:**
- Create: `ui/src/client/main.tsx`
- Create: `ui/src/client/index.css`
- Create: `ui/src/client/App.tsx`
- Create: `ui/src/client/hooks/useApi.ts`
- Create: `ui/src/client/types.ts`

**Step 1: Create main.tsx entry point**

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
```

**Step 2: Create index.css with Tailwind**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  font-family: "Instrument Sans", system-ui, sans-serif;
}

.font-mono {
  font-family: "JetBrains Mono", monospace;
}
```

**Step 3: Create types.ts**

```typescript
export interface VideoInfo {
  duration: number;
  fps: number;
  width: number;
  height: number;
  path: string;
}

export interface ClipMatch {
  start: number;
  end: number;
  score: number;
  match_type: string;
  matched_text: string | null;
}

export interface Clip {
  id: string;
  start: number;
  end: number;
  score?: number;
  match_type?: string;
  selected: boolean;
}

export interface IndexStatus {
  ready: boolean;
  shots: number;
  segments: number;
}
```

**Step 4: Create useApi.ts hook**

```typescript
import { useState, useCallback } from "react";
import type { VideoInfo, ClipMatch, IndexStatus } from "../types";

const API_BASE = "/api";

export function useApi() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadVideo = useCallback(async (path: string): Promise<VideoInfo> => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/video/load`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path }),
      });
      if (!res.ok) throw new Error(await res.text());
      return await res.json();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
      throw e;
    } finally {
      setLoading(false);
    }
  }, []);

  const buildIndex = useCallback(
    async (transcribe = true): Promise<IndexStatus> => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API_BASE}/index/build`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ transcribe }),
        });
        if (!res.ok) throw new Error(await res.text());
        return await res.json();
      } catch (e) {
        setError(e instanceof Error ? e.message : "Unknown error");
        throw e;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const getIndexStatus = useCallback(async (): Promise<IndexStatus> => {
    const res = await fetch(`${API_BASE}/index/status`);
    return await res.json();
  }, []);

  const search = useCallback(
    async (query: string, topK = 5): Promise<ClipMatch[]> => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API_BASE}/search`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, top_k: topK }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        return data.matches;
      } catch (e) {
        setError(e instanceof Error ? e.message : "Unknown error");
        throw e;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const getThumbnail = useCallback(
    async (time: number, width = 160): Promise<string> => {
      const res = await fetch(
        `${API_BASE}/video/thumbnail?t=${time}&w=${width}`
      );
      const data = await res.json();
      return `data:image/jpeg;base64,${data.data}`;
    },
    []
  );

  const getThumbnails = useCallback(
    async (times: number[], width = 160): Promise<string[]> => {
      const res = await fetch(`${API_BASE}/video/thumbnails`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ times, width }),
      });
      const data = await res.json();
      return data.thumbnails.map(
        (t: string) => `data:image/jpeg;base64,${t}`
      );
    },
    []
  );

  const exportClips = useCallback(
    async (
      clips: { start: number; end: number }[],
      stitch = false,
      outputDir = "./clips"
    ): Promise<string[]> => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API_BASE}/export`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            clips,
            stitch,
            output_dir: outputDir,
          }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        return data.outputs;
      } catch (e) {
        setError(e instanceof Error ? e.message : "Unknown error");
        throw e;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  return {
    loading,
    error,
    loadVideo,
    buildIndex,
    getIndexStatus,
    search,
    getThumbnail,
    getThumbnails,
    exportClips,
  };
}
```

**Step 5: Create App.tsx shell**

```tsx
import { useState } from "react";
import type { VideoInfo, Clip } from "./types";
import { useApi } from "./hooks/useApi";

export default function App() {
  const [videoInfo, setVideoInfo] = useState<VideoInfo | null>(null);
  const [clips, setClips] = useState<Clip[]>([]);
  const [currentTime, setCurrentTime] = useState(0);
  const [indexReady, setIndexReady] = useState(false);
  const api = useApi();

  // Placeholder - will be replaced with actual components
  return (
    <div className="min-h-screen bg-charcoal-900 text-white p-4">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold">Video Clipper</h1>
      </header>

      {!videoInfo ? (
        <div className="text-charcoal-600">
          <p>No video loaded. Run: ve ui /path/to/video.mp4</p>
        </div>
      ) : (
        <div className="space-y-4">
          <p className="font-mono text-sm text-charcoal-600">
            {videoInfo.path} ({videoInfo.duration.toFixed(1)}s)
          </p>
        </div>
      )}

      {api.error && (
        <div className="mt-4 p-3 bg-red-900/50 border border-red-700 rounded">
          {api.error}
        </div>
      )}
    </div>
  );
}
```

**Step 6: Verify frontend builds**

Run: `cd ui && bun run build`
Expected: Build completes with output in dist/client

**Step 7: Commit**

```bash
git add ui/src/client/
git commit -m "feat(client): add React app shell with API hooks"
```

---

## Task 5: Build SearchBar Component

**Files:**
- Create: `ui/src/client/components/SearchBar.tsx`

**Step 1: Create SearchBar.tsx**

```tsx
import { useState, useCallback, type FormEvent } from "react";
import { motion } from "motion/react";

interface SearchBarProps {
  onSearch: (query: string) => Promise<void>;
  loading?: boolean;
  disabled?: boolean;
}

export function SearchBar({ onSearch, loading, disabled }: SearchBarProps) {
  const [query, setQuery] = useState("");

  const handleSubmit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      if (query.trim() && !loading && !disabled) {
        await onSearch(query.trim());
      }
    },
    [query, loading, disabled, onSearch]
  );

  return (
    <form onSubmit={handleSubmit} className="relative">
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search for clips..."
        disabled={disabled}
        className="w-full bg-charcoal-800 border border-charcoal-600 rounded-lg
                   px-4 py-3 pl-11 text-white placeholder-charcoal-500
                   focus:outline-none focus:border-amber-500 focus:ring-1 focus:ring-amber-500
                   disabled:opacity-50 disabled:cursor-not-allowed
                   transition-colors"
      />

      {/* Search icon */}
      <svg
        className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-charcoal-500"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
        />
      </svg>

      {/* Loading spinner */}
      {loading && (
        <motion.div
          className="absolute right-4 top-1/2 -translate-y-1/2"
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
        >
          <svg
            className="w-5 h-5 text-amber-500"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
            />
          </svg>
        </motion.div>
      )}
    </form>
  );
}
```

**Step 2: Commit**

```bash
git add ui/src/client/components/SearchBar.tsx
git commit -m "feat(ui): add SearchBar component with loading state"
```

---

## Task 6: Build VideoPlayer Component

**Files:**
- Create: `ui/src/client/components/VideoPlayer.tsx`

**Step 1: Create VideoPlayer.tsx**

```tsx
import { useRef, useEffect, useCallback, useState } from "react";
import type { VideoInfo } from "../types";

interface VideoPlayerProps {
  videoPath: string;
  videoInfo: VideoInfo;
  currentTime: number;
  onTimeUpdate: (time: number) => void;
}

export function VideoPlayer({
  videoPath,
  videoInfo,
  currentTime,
  onTimeUpdate,
}: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(1);

  // Sync video to external currentTime changes
  useEffect(() => {
    if (videoRef.current && Math.abs(videoRef.current.currentTime - currentTime) > 0.5) {
      videoRef.current.currentTime = currentTime;
    }
  }, [currentTime]);

  const handleTimeUpdate = useCallback(() => {
    if (videoRef.current) {
      onTimeUpdate(videoRef.current.currentTime);
    }
  }, [onTimeUpdate]);

  const togglePlay = useCallback(() => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  }, [isPlaying]);

  const seek = useCallback((delta: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = Math.max(
        0,
        Math.min(videoInfo.duration, videoRef.current.currentTime + delta)
      );
    }
  }, [videoInfo.duration]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;

      switch (e.code) {
        case "Space":
          e.preventDefault();
          togglePlay();
          break;
        case "ArrowLeft":
          e.preventDefault();
          seek(-5);
          break;
        case "ArrowRight":
          e.preventDefault();
          seek(5);
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [togglePlay, seek]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="relative bg-black rounded-lg overflow-hidden">
      <video
        ref={videoRef}
        src={`/api/video/stream?path=${encodeURIComponent(videoPath)}`}
        className="w-full aspect-video"
        onTimeUpdate={handleTimeUpdate}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
      />

      {/* Custom controls */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
        <div className="flex items-center gap-4">
          {/* Play/Pause */}
          <button
            onClick={togglePlay}
            className="text-white hover:text-amber-400 transition-colors"
          >
            {isPlaying ? (
              <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
                <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
              </svg>
            ) : (
              <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z" />
              </svg>
            )}
          </button>

          {/* Time display */}
          <span className="font-mono text-sm text-white">
            {formatTime(currentTime)} / {formatTime(videoInfo.duration)}
          </span>

          {/* Volume */}
          <div className="flex items-center gap-2 ml-auto">
            <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
              <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z" />
            </svg>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={volume}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                setVolume(v);
                if (videoRef.current) videoRef.current.volume = v;
              }}
              className="w-20 accent-amber-500"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
```

**Step 2: Add video streaming route**

Update `ui/src/server/routes/video.ts` to add:

```typescript
  .get("/stream", async ({ query, set }) => {
    const videoPath = query.path;
    if (!videoPath) {
      set.status = 400;
      return { error: "Missing path parameter" };
    }

    const file = Bun.file(videoPath);
    if (!(await file.exists())) {
      set.status = 404;
      return { error: "Video not found" };
    }

    set.headers["Content-Type"] = file.type;
    set.headers["Accept-Ranges"] = "bytes";
    return file;
  })
```

**Step 3: Commit**

```bash
git add ui/src/client/components/VideoPlayer.tsx ui/src/server/routes/video.ts
git commit -m "feat(ui): add VideoPlayer component with custom controls"
```

---

## Task 7: Build Timeline Component

**Files:**
- Create: `ui/src/client/components/Timeline.tsx`

**Step 1: Create Timeline.tsx**

```tsx
import { useRef, useState, useCallback, useEffect, useMemo } from "react";
import { motion } from "motion/react";
import type { Clip, VideoInfo } from "../types";

interface TimelineProps {
  videoInfo: VideoInfo;
  currentTime: number;
  clips: Clip[];
  thumbnails: Map<number, string>;
  onSeek: (time: number) => void;
  onClipClick: (clip: Clip) => void;
}

export function Timeline({
  videoInfo,
  currentTime,
  clips,
  thumbnails,
  onSeek,
  onClipClick,
}: TimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [zoom, setZoom] = useState(1);

  // Calculate thumbnail positions (1 per second at zoom 1)
  const thumbnailInterval = useMemo(() => Math.max(1, Math.floor(5 / zoom)), [zoom]);
  const thumbnailTimes = useMemo(() => {
    const times: number[] = [];
    for (let t = 0; t < videoInfo.duration; t += thumbnailInterval) {
      times.push(t);
    }
    return times;
  }, [videoInfo.duration, thumbnailInterval]);

  // Calculate playhead position
  const playheadPercent = (currentTime / videoInfo.duration) * 100;

  // Handle click/drag to seek
  const handleSeek = useCallback(
    (clientX: number) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const percent = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      onSeek(percent * videoInfo.duration);
    },
    [videoInfo.duration, onSeek]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      setIsDragging(true);
      handleSeek(e.clientX);
    },
    [handleSeek]
  );

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => handleSeek(e.clientX);
    const handleMouseUp = () => setIsDragging(false);

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isDragging, handleSeek]);

  // Handle zoom with wheel
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    setZoom((z) => Math.max(0.5, Math.min(5, z - e.deltaY * 0.001)));
  }, []);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="space-y-2">
      {/* Thumbnail strip */}
      <div
        ref={containerRef}
        className="relative h-24 bg-charcoal-800 rounded-lg overflow-hidden cursor-pointer select-none"
        onMouseDown={handleMouseDown}
        onWheel={handleWheel}
      >
        {/* Thumbnails */}
        <div
          className="absolute inset-0 flex"
          style={{ width: `${100 * zoom}%` }}
        >
          {thumbnailTimes.map((time) => (
            <div
              key={time}
              className="flex-shrink-0 h-full border-r border-charcoal-700"
              style={{ width: `${(thumbnailInterval / videoInfo.duration) * 100}%` }}
            >
              {thumbnails.has(time) ? (
                <img
                  src={thumbnails.get(time)}
                  alt={`Frame at ${formatTime(time)}`}
                  className="h-full w-full object-cover"
                />
              ) : (
                <div className="h-full w-full bg-charcoal-700 animate-pulse" />
              )}
            </div>
          ))}
        </div>

        {/* Clip highlights */}
        {clips.map((clip) => {
          const left = (clip.start / videoInfo.duration) * 100;
          const width = ((clip.end - clip.start) / videoInfo.duration) * 100;
          return (
            <motion.div
              key={clip.id}
              className={`absolute top-0 bottom-0 cursor-pointer ${
                clip.selected
                  ? "bg-amber-500/40 border-2 border-amber-500"
                  : "bg-amber-500/20 border border-amber-500/50"
              }`}
              style={{ left: `${left}%`, width: `${width}%` }}
              onClick={(e) => {
                e.stopPropagation();
                onClipClick(clip);
              }}
              whileHover={{ backgroundColor: "rgba(245, 158, 11, 0.5)" }}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            />
          );
        })}

        {/* Playhead */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-amber-400 pointer-events-none z-10"
          style={{ left: `${playheadPercent}%` }}
        >
          <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-3 h-3 bg-amber-400 rotate-45" />
        </div>
      </div>

      {/* Time markers */}
      <div className="flex justify-between text-xs font-mono text-charcoal-500">
        <span>0:00</span>
        <span>{formatTime(videoInfo.duration / 4)}</span>
        <span>{formatTime(videoInfo.duration / 2)}</span>
        <span>{formatTime((videoInfo.duration * 3) / 4)}</span>
        <span>{formatTime(videoInfo.duration)}</span>
      </div>
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add ui/src/client/components/Timeline.tsx
git commit -m "feat(ui): add Timeline component with thumbnails and playhead"
```

---

## Task 8: Build ClipList Component

**Files:**
- Create: `ui/src/client/components/ClipList.tsx`

**Step 1: Create ClipList.tsx**

```tsx
import { motion } from "motion/react";
import type { Clip } from "../types";

interface ClipListProps {
  clips: Clip[];
  onToggleClip: (clipId: string) => void;
  onClipClick: (clip: Clip) => void;
  onSelectAll: () => void;
  onDeselectAll: () => void;
}

export function ClipList({
  clips,
  onToggleClip,
  onClipClick,
  onSelectAll,
  onDeselectAll,
}: ClipListProps) {
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const selectedCount = clips.filter((c) => c.selected).length;

  if (clips.length === 0) {
    return (
      <div className="text-charcoal-500 text-center py-8">
        No clips found. Search for something above.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Header with select/deselect buttons */}
      <div className="flex items-center justify-between">
        <span className="text-sm text-charcoal-500">
          {selectedCount} of {clips.length} selected
        </span>
        <div className="flex gap-2">
          <button
            onClick={onSelectAll}
            className="px-3 py-1 text-xs bg-charcoal-700 hover:bg-charcoal-600
                       rounded transition-colors"
          >
            Select All
          </button>
          <button
            onClick={onDeselectAll}
            className="px-3 py-1 text-xs bg-charcoal-700 hover:bg-charcoal-600
                       rounded transition-colors"
          >
            Deselect All
          </button>
        </div>
      </div>

      {/* Clip list */}
      <div className="space-y-2">
        {clips.map((clip, index) => (
          <motion.div
            key={clip.id}
            className={`flex items-center gap-3 p-3 rounded-lg cursor-pointer
                        transition-colors ${
                          clip.selected
                            ? "bg-charcoal-700 border border-amber-500/50"
                            : "bg-charcoal-800 border border-transparent hover:bg-charcoal-700"
                        }`}
            onClick={() => onClipClick(clip)}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05 }}
          >
            {/* Checkbox */}
            <button
              onClick={(e) => {
                e.stopPropagation();
                onToggleClip(clip.id);
              }}
              className={`w-5 h-5 rounded border-2 flex items-center justify-center
                          transition-colors ${
                            clip.selected
                              ? "bg-amber-500 border-amber-500"
                              : "border-charcoal-500 hover:border-amber-500"
                          }`}
            >
              {clip.selected && (
                <svg
                  className="w-3 h-3 text-charcoal-900"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                    clipRule="evenodd"
                  />
                </svg>
              )}
            </button>

            {/* Clip info */}
            <div className="flex-1">
              <div className="font-mono text-sm">
                {formatTime(clip.start)} - {formatTime(clip.end)}
                <span className="text-charcoal-500 ml-2">
                  ({(clip.end - clip.start).toFixed(1)}s)
                </span>
              </div>
              {clip.score !== undefined && (
                <div className="text-xs text-charcoal-500">
                  Score: {(clip.score * 100).toFixed(0)}%
                  {clip.match_type && ` | ${clip.match_type}`}
                </div>
              )}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add ui/src/client/components/ClipList.tsx
git commit -m "feat(ui): add ClipList component with selection state"
```

---

## Task 9: Build ExportButton Component

**Files:**
- Create: `ui/src/client/components/ExportButton.tsx`

**Step 1: Create ExportButton.tsx**

```tsx
import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "motion/react";

interface ExportButtonProps {
  selectedCount: number;
  onExport: (stitch: boolean) => Promise<void>;
  loading?: boolean;
}

export function ExportButton({
  selectedCount,
  onExport,
  loading,
}: ExportButtonProps) {
  const [isOpen, setIsOpen] = useState(false);

  const handleExport = useCallback(
    async (stitch: boolean) => {
      setIsOpen(false);
      await onExport(stitch);
    },
    [onExport]
  );

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        disabled={selectedCount === 0 || loading}
        className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium
                    transition-colors ${
                      selectedCount === 0 || loading
                        ? "bg-charcoal-700 text-charcoal-500 cursor-not-allowed"
                        : "bg-amber-500 text-charcoal-900 hover:bg-amber-400"
                    }`}
      >
        {loading ? (
          <motion.span
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24">
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
              />
            </svg>
          </motion.span>
        ) : (
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
            />
          </svg>
        )}
        Export ({selectedCount})
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="absolute right-0 top-full mt-2 w-56 bg-charcoal-800
                       border border-charcoal-600 rounded-lg shadow-xl z-20 overflow-hidden"
          >
            <button
              onClick={() => handleExport(false)}
              className="w-full px-4 py-3 text-left hover:bg-charcoal-700
                         transition-colors border-b border-charcoal-700"
            >
              <div className="font-medium">Individual clips</div>
              <div className="text-xs text-charcoal-500">
                Export each clip as separate file
              </div>
            </button>
            <button
              onClick={() => handleExport(true)}
              className="w-full px-4 py-3 text-left hover:bg-charcoal-700 transition-colors"
            >
              <div className="font-medium">Stitch into one</div>
              <div className="text-xs text-charcoal-500">
                Combine all clips into single video
              </div>
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add ui/src/client/components/ExportButton.tsx
git commit -m "feat(ui): add ExportButton component with dropdown menu"
```

---

## Task 10: Wire Up App.tsx

**Files:**
- Modify: `ui/src/client/App.tsx`

**Step 1: Update App.tsx with all components**

```tsx
import { useState, useCallback, useEffect } from "react";
import type { VideoInfo, Clip, ClipMatch } from "./types";
import { useApi } from "./hooks/useApi";
import { SearchBar } from "./components/SearchBar";
import { VideoPlayer } from "./components/VideoPlayer";
import { Timeline } from "./components/Timeline";
import { ClipList } from "./components/ClipList";
import { ExportButton } from "./components/ExportButton";

export default function App() {
  const [videoInfo, setVideoInfo] = useState<VideoInfo | null>(null);
  const [clips, setClips] = useState<Clip[]>([]);
  const [currentTime, setCurrentTime] = useState(0);
  const [indexReady, setIndexReady] = useState(false);
  const [thumbnails, setThumbnails] = useState<Map<number, string>>(new Map());
  const [exporting, setExporting] = useState(false);
  const api = useApi();

  // Load video from URL params or prompt
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const videoPath = params.get("video");
    if (videoPath) {
      handleLoadVideo(videoPath);
    }
  }, []);

  const handleLoadVideo = useCallback(
    async (path: string) => {
      const info = await api.loadVideo(path);
      setVideoInfo(info);

      // Build index
      const indexStatus = await api.buildIndex(true);
      setIndexReady(indexStatus.ready || indexStatus.shots > 0);

      // Load initial thumbnails
      const times: number[] = [];
      for (let t = 0; t < info.duration; t += 5) {
        times.push(t);
      }
      const thumbs = await api.getThumbnails(times, 160);
      const thumbMap = new Map<number, string>();
      times.forEach((t, i) => thumbMap.set(t, thumbs[i]));
      setThumbnails(thumbMap);
    },
    [api]
  );

  const handleSearch = useCallback(
    async (query: string) => {
      const matches = await api.search(query, 10);
      const newClips: Clip[] = matches.map((m: ClipMatch, i: number) => ({
        id: `clip-${i}-${m.start}`,
        start: m.start,
        end: m.end,
        score: m.score,
        match_type: m.match_type,
        selected: true,
      }));
      setClips(newClips);
    },
    [api]
  );

  const handleToggleClip = useCallback((clipId: string) => {
    setClips((prev) =>
      prev.map((c) => (c.id === clipId ? { ...c, selected: !c.selected } : c))
    );
  }, []);

  const handleClipClick = useCallback((clip: Clip) => {
    setCurrentTime(clip.start);
  }, []);

  const handleSelectAll = useCallback(() => {
    setClips((prev) => prev.map((c) => ({ ...c, selected: true })));
  }, []);

  const handleDeselectAll = useCallback(() => {
    setClips((prev) => prev.map((c) => ({ ...c, selected: false })));
  }, []);

  const handleExport = useCallback(
    async (stitch: boolean) => {
      const selectedClips = clips.filter((c) => c.selected);
      if (selectedClips.length === 0) return;

      setExporting(true);
      try {
        const outputs = await api.exportClips(
          selectedClips.map((c) => ({ start: c.start, end: c.end })),
          stitch
        );
        // Show success (could add toast notification here)
        console.log("Exported to:", outputs);
        alert(`Exported ${outputs.length} file(s) to ./clips`);
      } finally {
        setExporting(false);
      }
    },
    [clips, api]
  );

  const selectedCount = clips.filter((c) => c.selected).length;

  if (!videoInfo) {
    return (
      <div className="min-h-screen bg-charcoal-900 text-white flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-semibold mb-4">Video Clipper</h1>
          <p className="text-charcoal-500">
            Run: <code className="font-mono bg-charcoal-800 px-2 py-1 rounded">ve ui /path/to/video.mp4</code>
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-charcoal-900 text-white">
      {/* Header */}
      <header className="border-b border-charcoal-700 p-4">
        <div className="max-w-6xl mx-auto flex items-center gap-4">
          <div className="flex-1">
            <SearchBar
              onSearch={handleSearch}
              loading={api.loading}
              disabled={!indexReady}
            />
          </div>
          <ExportButton
            selectedCount={selectedCount}
            onExport={handleExport}
            loading={exporting}
          />
        </div>
      </header>

      <main className="max-w-6xl mx-auto p-4 space-y-6">
        {/* Video player */}
        <VideoPlayer
          videoPath={videoInfo.path}
          videoInfo={videoInfo}
          currentTime={currentTime}
          onTimeUpdate={setCurrentTime}
        />

        {/* Timeline */}
        <Timeline
          videoInfo={videoInfo}
          currentTime={currentTime}
          clips={clips}
          thumbnails={thumbnails}
          onSeek={setCurrentTime}
          onClipClick={handleClipClick}
        />

        {/* Clip list */}
        <ClipList
          clips={clips}
          onToggleClip={handleToggleClip}
          onClipClick={handleClipClick}
          onSelectAll={handleSelectAll}
          onDeselectAll={handleDeselectAll}
        />

        {/* Status bar */}
        <div className="text-xs text-charcoal-500 font-mono">
          {videoInfo.path} | {videoInfo.duration.toFixed(1)}s | {videoInfo.width}x{videoInfo.height} @ {videoInfo.fps.toFixed(2)}fps
          {indexReady && " | Index ready"}
        </div>
      </main>

      {/* Error display */}
      {api.error && (
        <div className="fixed bottom-4 right-4 max-w-md p-4 bg-red-900/90 border border-red-700 rounded-lg">
          <div className="font-medium">Error</div>
          <div className="text-sm text-red-200">{api.error}</div>
        </div>
      )}
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add ui/src/client/App.tsx
git commit -m "feat(ui): wire up all components in App"
```

---

## Task 11: Add CLI Command

**Files:**
- Modify: `src/video_clipper/cli.py`

**Step 1: Add `ui` command to CLI**

Add after the `clip` subparser (around line 352):

```python
    # ui
    p_ui = subparsers.add_parser("ui", help="Launch web UI")
    p_ui.add_argument("video", type=Path, help="Video file")
    p_ui.add_argument("--port", type=int, default=3000, help="Server port")
    p_ui.add_argument("--no-browser", action="store_true", help="Don't open browser")
    _add_global_options(p_ui)
    p_ui.set_defaults(func=cmd_ui)
```

**Step 2: Add cmd_ui function**

Add before `main()`:

```python
def cmd_ui(args):
    """Launch the web UI."""
    import subprocess
    import webbrowser
    import os

    video_path = args.video.resolve()
    if not video_path.exists():
        console.print(f"[red]Error:[/red] Video not found: {video_path}")
        sys.exit(1)

    # Find UI directory
    ui_dir = Path(__file__).parent.parent.parent.parent / "ui"
    if not ui_dir.exists():
        console.print("[red]Error:[/red] UI not found. Run 'cd ui && bun install' first.")
        sys.exit(1)

    url = f"http://localhost:{args.port}?video={video_path}"
    console.print(f"[bold]Starting UI server...[/bold]")
    console.print(f"[dim]Video: {video_path}[/dim]")
    console.print(f"\n[green]Open:[/green] {url}\n")

    if not args.no_browser:
        webbrowser.open(url)

    # Start the server
    env = os.environ.copy()
    env["PORT"] = str(args.port)

    try:
        subprocess.run(
            ["bun", "run", "dev"],
            cwd=ui_dir,
            env=env,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped[/yellow]")
```

**Step 3: Verify CLI works**

Run: `cd /Users/nicholasmurphy/Documents/Code/video-clipper && python -c "from video_clipper.cli import main; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add src/video_clipper/cli.py
git commit -m "feat(cli): add 've ui' command to launch web interface"
```

---

## Task 12: Add Component Index

**Files:**
- Create: `ui/src/client/components/index.ts`

**Step 1: Create barrel export**

```typescript
export { SearchBar } from "./SearchBar";
export { VideoPlayer } from "./VideoPlayer";
export { Timeline } from "./Timeline";
export { ClipList } from "./ClipList";
export { ExportButton } from "./ExportButton";
```

**Step 2: Update App.tsx imports**

Replace individual imports with:

```typescript
import {
  SearchBar,
  VideoPlayer,
  Timeline,
  ClipList,
  ExportButton,
} from "./components";
```

**Step 3: Commit**

```bash
git add ui/src/client/components/index.ts ui/src/client/App.tsx
git commit -m "refactor(ui): add component barrel export"
```

---

## Task 13: Final Integration Test

**Step 1: Install UI dependencies**

Run: `cd ui && bun install`
Expected: Dependencies install successfully

**Step 2: Build frontend**

Run: `cd ui && bun run build`
Expected: Vite builds to dist/client

**Step 3: Test with sample video**

Run: `ve ui /path/to/any/video.mp4`
Expected: Browser opens, video loads, search works

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat(ui): complete MVP web interface

- Bun/Elysia server with PyBridge Python integration
- React 19 frontend with Vite and Tailwind
- Thumbnail timeline with clip highlights
- Video player with custom controls
- Search and export functionality
- Dark cinematic theme with amber accents"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Initialize UI project | package.json, configs |
| 2 | Python bridge module | bridge.py |
| 3 | Elysia server | server/, routes/ |
| 4 | React app shell | client/, hooks/ |
| 5 | SearchBar component | SearchBar.tsx |
| 6 | VideoPlayer component | VideoPlayer.tsx |
| 7 | Timeline component | Timeline.tsx |
| 8 | ClipList component | ClipList.tsx |
| 9 | ExportButton component | ExportButton.tsx |
| 10 | Wire up App | App.tsx |
| 11 | CLI command | cli.py |
| 12 | Component index | components/index.ts |
| 13 | Integration test | - |
