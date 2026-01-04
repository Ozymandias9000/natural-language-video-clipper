import { useEffect, useRef, useState, useCallback } from "react";
import type { VideoInfo } from "../types";

interface VideoPlayerProps {
  videoPath: string;
  videoInfo: VideoInfo;
  currentTime: number;
  onTimeUpdate: (time: number) => void;
}

/**
 * Formats seconds into MM:SS or HH:MM:SS display format.
 * Uses hours only when duration exceeds 1 hour.
 */
function formatTime(seconds: number): string {
  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hrs > 0) {
    return `${hrs}:${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  }
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

/**
 * Video player component with custom controls overlay.
 * Syncs playback position with external state via currentTime/onTimeUpdate props.
 */
export function VideoPlayer({
  videoPath,
  videoInfo,
  currentTime,
  onTimeUpdate,
}: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);

  // Track whether we're currently seeking to prevent feedback loops
  const isSeeking = useRef(false);

  /**
   * Sync external currentTime prop to video element.
   * Only seeks when the difference exceeds threshold to avoid constant micro-adjustments.
   */
  useEffect(() => {
    const video = videoRef.current;
    if (!video || isSeeking.current) return;

    const diff = Math.abs(video.currentTime - currentTime);
    // Threshold of 0.5s prevents jitter during normal playback
    if (diff > 0.5) {
      video.currentTime = currentTime;
    }
  }, [currentTime]);

  /**
   * Handle video timeupdate events and propagate to parent.
   */
  const handleTimeUpdate = useCallback(() => {
    const video = videoRef.current;
    if (!video || isSeeking.current) return;
    onTimeUpdate(video.currentTime);
  }, [onTimeUpdate]);

  /**
   * Toggle play/pause state.
   */
  const togglePlayPause = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;

    if (video.paused) {
      video.play();
      setIsPlaying(true);
    } else {
      video.pause();
      setIsPlaying(false);
    }
  }, []);

  /**
   * Seek relative to current position.
   */
  const seekRelative = useCallback((delta: number) => {
    const video = videoRef.current;
    if (!video) return;

    const newTime = Math.max(0, Math.min(video.duration, video.currentTime + delta));
    video.currentTime = newTime;
    onTimeUpdate(newTime);
  }, [onTimeUpdate]);

  /**
   * Handle volume changes from slider.
   */
  const handleVolumeChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newVolume = parseFloat(e.target.value);
    setVolume(newVolume);
    setIsMuted(newVolume === 0);

    const video = videoRef.current;
    if (video) {
      video.volume = newVolume;
      video.muted = newVolume === 0;
    }
  }, []);

  /**
   * Toggle mute state.
   */
  const toggleMute = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;

    const newMuted = !isMuted;
    setIsMuted(newMuted);
    video.muted = newMuted;
  }, [isMuted]);

  /**
   * Keyboard shortcuts for video control.
   * Space: play/pause, ArrowLeft: -5s, ArrowRight: +5s
   */
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle shortcuts when container or video has focus
      const container = containerRef.current;
      if (!container?.contains(document.activeElement) && document.activeElement !== document.body) {
        return;
      }

      switch (e.code) {
        case "Space":
          e.preventDefault();
          togglePlayPause();
          break;
        case "ArrowLeft":
          e.preventDefault();
          seekRelative(-5);
          break;
        case "ArrowRight":
          e.preventDefault();
          seekRelative(5);
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [togglePlayPause, seekRelative]);

  /**
   * Sync play/pause state when video events occur.
   */
  const handlePlay = useCallback(() => setIsPlaying(true), []);
  const handlePause = useCallback(() => setIsPlaying(false), []);

  /**
   * Progress bar click handler for seeking.
   */
  const handleProgressClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const video = videoRef.current;
    if (!video) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const percent = (e.clientX - rect.left) / rect.width;
    const newTime = percent * video.duration;

    isSeeking.current = true;
    video.currentTime = newTime;
    onTimeUpdate(newTime);
    // Reset seeking flag after a brief delay
    setTimeout(() => { isSeeking.current = false; }, 100);
  }, [onTimeUpdate]);

  const progress = videoInfo.duration > 0 ? (currentTime / videoInfo.duration) * 100 : 0;

  return (
    <div
      ref={containerRef}
      className="relative w-full bg-black rounded-lg overflow-hidden focus:outline-none"
      tabIndex={0}
    >
      <video
        ref={videoRef}
        src={`/api/video/stream?path=${encodeURIComponent(videoPath)}`}
        className="w-full h-full"
        onTimeUpdate={handleTimeUpdate}
        onPlay={handlePlay}
        onPause={handlePause}
      />

      {/* Custom controls overlay */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent pt-8 pb-3 px-4">
        {/* Progress bar */}
        <div
          className="w-full h-1 bg-white/30 rounded-full cursor-pointer mb-3 group"
          onClick={handleProgressClick}
        >
          <div
            className="h-full bg-white rounded-full relative"
            style={{ width: `${progress}%` }}
          >
            <div className="absolute right-0 top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity" />
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* Play/Pause button */}
          <button
            onClick={togglePlayPause}
            className="text-white hover:text-white/80 transition-colors"
            aria-label={isPlaying ? "Pause" : "Play"}
          >
            {isPlaying ? (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="currentColor"
                className="w-8 h-8"
              >
                <path
                  fillRule="evenodd"
                  d="M6.75 5.25a.75.75 0 01.75-.75H9a.75.75 0 01.75.75v13.5a.75.75 0 01-.75.75H7.5a.75.75 0 01-.75-.75V5.25zm7.5 0A.75.75 0 0115 4.5h1.5a.75.75 0 01.75.75v13.5a.75.75 0 01-.75.75H15a.75.75 0 01-.75-.75V5.25z"
                  clipRule="evenodd"
                />
              </svg>
            ) : (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="currentColor"
                className="w-8 h-8"
              >
                <path
                  fillRule="evenodd"
                  d="M4.5 5.653c0-1.426 1.529-2.33 2.779-1.643l11.54 6.348c1.295.712 1.295 2.573 0 3.285L7.28 19.991c-1.25.687-2.779-.217-2.779-1.643V5.653z"
                  clipRule="evenodd"
                />
              </svg>
            )}
          </button>

          {/* Time display */}
          <span className="text-white text-sm font-mono">
            {formatTime(currentTime)} / {formatTime(videoInfo.duration)}
          </span>

          {/* Spacer */}
          <div className="flex-1" />

          {/* Volume controls */}
          <div className="flex items-center gap-2">
            <button
              onClick={toggleMute}
              className="text-white hover:text-white/80 transition-colors"
              aria-label={isMuted ? "Unmute" : "Mute"}
            >
              {isMuted || volume === 0 ? (
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                  className="w-6 h-6"
                >
                  <path d="M13.5 4.06c0-1.336-1.616-2.005-2.56-1.06l-4.5 4.5H4.508c-1.141 0-2.318.664-2.66 1.905A9.76 9.76 0 001.5 12c0 .898.121 1.768.35 2.595.341 1.24 1.518 1.905 2.659 1.905h1.93l4.5 4.5c.945.945 2.561.276 2.561-1.06V4.06zM17.78 9.22a.75.75 0 10-1.06 1.06L18.44 12l-1.72 1.72a.75.75 0 101.06 1.06l1.72-1.72 1.72 1.72a.75.75 0 101.06-1.06L20.56 12l1.72-1.72a.75.75 0 10-1.06-1.06l-1.72 1.72-1.72-1.72z" />
                </svg>
              ) : (
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                  className="w-6 h-6"
                >
                  <path d="M13.5 4.06c0-1.336-1.616-2.005-2.56-1.06l-4.5 4.5H4.508c-1.141 0-2.318.664-2.66 1.905A9.76 9.76 0 001.5 12c0 .898.121 1.768.35 2.595.341 1.24 1.518 1.905 2.659 1.905h1.93l4.5 4.5c.945.945 2.561.276 2.561-1.06V4.06zM18.584 5.106a.75.75 0 011.06 0c3.808 3.807 3.808 9.98 0 13.788a.75.75 0 11-1.06-1.06 8.25 8.25 0 000-11.668.75.75 0 010-1.06z" />
                  <path d="M15.932 7.757a.75.75 0 011.061 0 6 6 0 010 8.486.75.75 0 01-1.06-1.061 4.5 4.5 0 000-6.364.75.75 0 010-1.06z" />
                </svg>
              )}
            </button>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={isMuted ? 0 : volume}
              onChange={handleVolumeChange}
              className="w-20 h-1 bg-white/30 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:rounded-full"
              aria-label="Volume"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
