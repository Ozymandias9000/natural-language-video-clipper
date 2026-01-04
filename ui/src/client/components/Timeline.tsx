import { useRef, useState, useCallback, useEffect } from "react";
import { motion } from "motion/react";
import type { VideoInfo, Clip } from "../types";

interface TimelineProps {
  /** Video metadata for duration calculations */
  videoInfo: VideoInfo;
  /** Current playback position in seconds */
  currentTime: number;
  /** Array of clips to highlight on timeline */
  clips: Clip[];
  /** Map of timestamp (seconds) to base64 thumbnail data URL */
  thumbnails: Map<number, string>;
  /** Callback when user seeks to a new time */
  onSeek: (time: number) => void;
  /** Callback when user clicks on a clip overlay */
  onClipClick: (clip: Clip) => void;
}

/**
 * Formats seconds into MM:SS display format.
 */
function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

/**
 * Timeline component providing visual navigation through video content.
 * Features thumbnail strip, playhead, clip overlays, and seek functionality.
 */
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
  const [zoomLevel, setZoomLevel] = useState(1);

  const duration = videoInfo.duration;

  /**
   * Calculates time position from mouse X coordinate relative to container.
   */
  const getTimeFromMouseX = useCallback(
    (clientX: number): number => {
      if (!containerRef.current) return 0;
      const rect = containerRef.current.getBoundingClientRect();
      const x = clientX - rect.left;
      const ratio = Math.max(0, Math.min(1, x / rect.width));
      return ratio * duration;
    },
    [duration]
  );

  /**
   * Handles mouse down to initiate seeking and dragging.
   */
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      setIsDragging(true);
      const time = getTimeFromMouseX(e.clientX);
      onSeek(time);
    },
    [getTimeFromMouseX, onSeek]
  );

  /**
   * Handles mouse move during drag to update seek position.
   */
  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging) return;
      const time = getTimeFromMouseX(e.clientX);
      onSeek(time);
    },
    [isDragging, getTimeFromMouseX, onSeek]
  );

  /**
   * Handles mouse up to end dragging state.
   */
  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Attach global mouse listeners for drag behavior
  useEffect(() => {
    if (isDragging) {
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);
      return () => {
        window.removeEventListener("mousemove", handleMouseMove);
        window.removeEventListener("mouseup", handleMouseUp);
      };
    }
  }, [isDragging, handleMouseMove, handleMouseUp]);

  /**
   * Handles mouse wheel to zoom timeline view.
   */
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    setZoomLevel((prev) => {
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      return Math.max(0.5, Math.min(4, prev * delta));
    });
  }, []);

  /**
   * Calculates percentage position on timeline for a given time.
   */
  const getPositionPercent = (time: number): number => {
    return (time / duration) * 100;
  };

  // Sort thumbnails by timestamp for ordered rendering
  const sortedThumbnails = Array.from(thumbnails.entries()).sort(
    ([a], [b]) => a - b
  );

  // Time markers at 0, 1/4, 1/2, 3/4, and full duration
  const timeMarkers = [
    0,
    duration / 4,
    duration / 2,
    (3 * duration) / 4,
    duration,
  ];

  return (
    <div className="w-full select-none">
      {/* Timeline container with thumbnails and overlays */}
      <div
        ref={containerRef}
        className="relative h-20 bg-zinc-900 rounded-lg overflow-hidden cursor-pointer"
        style={{ transform: `scaleX(${zoomLevel})`, transformOrigin: "left" }}
        onMouseDown={handleMouseDown}
        onWheel={handleWheel}
      >
        {/* Thumbnail strip */}
        <div className="absolute inset-0 flex">
          {sortedThumbnails.map(([timestamp, dataUrl]) => (
            <div
              key={timestamp}
              className="h-full flex-shrink-0"
              style={{
                width: `${100 / Math.max(sortedThumbnails.length, 1)}%`,
              }}
            >
              <img
                src={dataUrl}
                alt={`Frame at ${formatTime(timestamp)}`}
                className="w-full h-full object-cover"
                draggable={false}
              />
            </div>
          ))}
          {/* Fallback gradient when no thumbnails */}
          {sortedThumbnails.length === 0 && (
            <div className="absolute inset-0 bg-gradient-to-r from-zinc-800 to-zinc-700" />
          )}
        </div>

        {/* Clip highlight overlays */}
        {clips.map((clip) => (
          <motion.div
            key={clip.id}
            className="absolute top-0 bottom-0 bg-amber-500/30 border-l border-r border-amber-500/50 cursor-pointer"
            style={{
              left: `${getPositionPercent(clip.start)}%`,
              width: `${getPositionPercent(clip.end - clip.start)}%`,
            }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            whileHover={{
              backgroundColor: "rgba(245, 158, 11, 0.45)",
              transition: { duration: 0.15 },
            }}
            transition={{ duration: 0.2 }}
            onClick={(e) => {
              e.stopPropagation();
              onClipClick(clip);
            }}
          />
        ))}

        {/* Playhead indicator */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-amber-500 pointer-events-none"
          style={{ left: `${getPositionPercent(currentTime)}%` }}
        >
          {/* Triangle top marker */}
          <div
            className="absolute -top-1 left-1/2 -translate-x-1/2 w-0 h-0"
            style={{
              borderLeft: "6px solid transparent",
              borderRight: "6px solid transparent",
              borderTop: "8px solid rgb(245, 158, 11)",
            }}
          />
        </div>
      </div>

      {/* Time markers */}
      <div className="relative h-6 mt-1">
        {timeMarkers.map((time, index) => (
          <div
            key={index}
            className="absolute text-xs text-zinc-400 -translate-x-1/2"
            style={{
              left: `${getPositionPercent(time)}%`,
              // Adjust edge markers to prevent overflow
              ...(index === 0 && { transform: "translateX(0)" }),
              ...(index === timeMarkers.length - 1 && {
                transform: "translateX(-100%)",
              }),
            }}
          >
            {formatTime(time)}
          </div>
        ))}
      </div>
    </div>
  );
}
