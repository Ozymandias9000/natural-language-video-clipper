import { useState, useEffect, useCallback } from "react";
import type { VideoInfo, Clip, ClipMatch } from "./types";
import { useApi } from "./hooks/useApi";
import { SearchBar } from "./components/SearchBar";
import { VideoPlayer } from "./components/VideoPlayer";
import { Timeline } from "./components/Timeline";
import ClipList from "./components/ClipList";
import { ExportButton } from "./components/ExportButton";

/**
 * Main application shell for the video clipper UI.
 * Loads video from URL params and provides the core application state.
 */
export default function App() {
  const [videoInfo, setVideoInfo] = useState<VideoInfo | null>(null);
  const [clips, setClips] = useState<Clip[]>([]);
  const [currentTime, setCurrentTime] = useState(0);
  const [indexReady, setIndexReady] = useState(false);
  const [thumbnails, setThumbnails] = useState<Map<number, string>>(new Map());
  const api = useApi();

  // Load video from URL params on mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const videoPath = params.get("video");
    if (videoPath) {
      handleLoadVideo(videoPath);
    }
  }, []);

  const handleLoadVideo = useCallback(
    async (path: string) => {
      try {
        const info = await api.loadVideo(path);
        setVideoInfo(info);

        // Build index after loading video
        const indexStatus = await api.buildIndex(true);
        setIndexReady(indexStatus.ready || indexStatus.shots > 0);

        // Load initial thumbnails (1 every 5 seconds)
        const times: number[] = [];
        for (let t = 0; t < info.duration; t += 5) {
          times.push(t);
        }
        if (times.length > 0) {
          const thumbs = await api.getThumbnails(times, 160);
          const thumbMap = new Map<number, string>();
          times.forEach((t, i) => thumbMap.set(t, thumbs[i]));
          setThumbnails(thumbMap);
        }
      } catch (err) {
        // Error is already captured in api.error
        console.error("Failed to load video:", err);
      }
    },
    [api]
  );

  // Placeholder search handler - will be wired to SearchBar component
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
      const selectedClips = clips
        .filter((c) => c.selected)
        .map((c) => ({ start: c.start, end: c.end }));
      if (selectedClips.length === 0) return;
      await api.exportClips(selectedClips, stitch);
    },
    [api, clips]
  );

  // No video loaded state
  if (!videoInfo) {
    return (
      <div className="min-h-screen bg-charcoal-900 text-white flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-semibold mb-4">Video Clipper</h1>
          {api.loading ? (
            <p className="text-charcoal-500">Loading video...</p>
          ) : (
            <p className="text-charcoal-500">
              No video loaded. Run:{" "}
              <code className="font-mono bg-charcoal-800 px-2 py-1 rounded">
                ve ui /path/to/video.mp4
              </code>
            </p>
          )}
          {api.error && (
            <div className="mt-4 p-3 bg-red-900/50 border border-red-700 rounded max-w-md mx-auto">
              <p className="text-red-200 text-sm">{api.error}</p>
            </div>
          )}
        </div>
      </div>
    );
  }

  const selectedCount = clips.filter((c) => c.selected).length;

  return (
    <div className="min-h-screen bg-charcoal-900 text-white p-4">
      {/* Header with SearchBar and ExportButton */}
      <header className="mb-6 max-w-6xl mx-auto">
        <div className="flex items-center gap-4">
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
            loading={api.loading}
          />
        </div>
      </header>

      <main className="max-w-6xl mx-auto space-y-6">
        {/* Video player */}
        <VideoPlayer
          videoPath={videoInfo.path}
          videoInfo={videoInfo}
          currentTime={currentTime}
          onTimeUpdate={setCurrentTime}
        />

        {/* Timeline with thumbnails and clip overlays */}
        <Timeline
          videoInfo={videoInfo}
          currentTime={currentTime}
          clips={clips}
          thumbnails={thumbnails}
          onSeek={setCurrentTime}
          onClipClick={handleClipClick}
        />

        {/* Clip list showing search results */}
        <ClipList
          clips={clips}
          onToggleClip={handleToggleClip}
          onClipClick={handleClipClick}
          onSelectAll={handleSelectAll}
          onDeselectAll={handleDeselectAll}
        />

        {/* Status bar */}
        <div className="text-xs text-charcoal-500 font-mono">
          {videoInfo.path} | {videoInfo.duration.toFixed(1)}s |{" "}
          {videoInfo.width}x{videoInfo.height} @ {videoInfo.fps.toFixed(2)}fps
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
