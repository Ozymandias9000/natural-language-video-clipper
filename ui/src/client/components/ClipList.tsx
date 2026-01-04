import { motion } from "motion/react";
import type { Clip } from "../types";

interface ClipListProps {
  /** Array of clips to display */
  clips: Clip[];
  /** Callback when a clip's selection state is toggled */
  onToggleClip: (clipId: string) => void;
  /** Callback when a clip is clicked (for seeking to that time) */
  onClipClick: (clip: Clip) => void;
  /** Callback to select all clips */
  onSelectAll: () => void;
  /** Callback to deselect all clips */
  onDeselectAll: () => void;
}

/**
 * Formats seconds into a human-readable time string (M:SS or H:MM:SS).
 */
function formatTime(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  }
  return `${minutes}:${secs.toString().padStart(2, "0")}`;
}

/**
 * Displays a list of video clips with selection controls.
 * Supports bulk selection operations and individual clip interactions.
 */
export default function ClipList({
  clips,
  onToggleClip,
  onClipClick,
  onSelectAll,
  onDeselectAll,
}: ClipListProps) {
  const selectedCount = clips.filter((c) => c.selected).length;

  // Empty state when no clips are available
  if (clips.length === 0) {
    return (
      <div className="bg-charcoal-800 rounded-lg p-8 text-center">
        <p className="text-charcoal-500">
          No clips found. Search for something above.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-charcoal-800 rounded-lg overflow-hidden">
      {/* Header with selection controls */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-charcoal-700">
        <span className="text-sm text-charcoal-400">
          {selectedCount} of {clips.length} selected
        </span>
        <div className="flex gap-2">
          <button
            onClick={onSelectAll}
            className="px-3 py-1 text-sm text-charcoal-300 hover:text-white hover:bg-charcoal-700 rounded transition-colors"
          >
            Select All
          </button>
          <button
            onClick={onDeselectAll}
            className="px-3 py-1 text-sm text-charcoal-300 hover:text-white hover:bg-charcoal-700 rounded transition-colors"
          >
            Deselect All
          </button>
        </div>
      </div>

      {/* Clip items */}
      <ul className="divide-y divide-charcoal-700">
        {clips.map((clip, index) => (
          <motion.li
            key={clip.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.2, delay: index * 0.05 }}
            className="flex items-center gap-3 px-4 py-3 hover:bg-charcoal-700/50 cursor-pointer transition-colors"
            onClick={() => onClipClick(clip)}
          >
            {/* Selection checkbox */}
            <button
              onClick={(e) => {
                e.stopPropagation();
                onToggleClip(clip.id);
              }}
              className={`w-5 h-5 rounded border-2 flex items-center justify-center transition-colors ${
                clip.selected
                  ? "bg-amber-500 border-amber-500"
                  : "border-charcoal-500 hover:border-charcoal-400"
              }`}
              aria-label={clip.selected ? "Deselect clip" : "Select clip"}
            >
              {clip.selected && (
                <svg
                  className="w-3 h-3 text-charcoal-900"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={3}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M5 13l4 4L19 7"
                  />
                </svg>
              )}
            </button>

            {/* Time range display */}
            <div className="flex-1 min-w-0">
              <span className="font-mono text-sm text-white">
                {formatTime(clip.start)} - {formatTime(clip.end)} (
                {(clip.end - clip.start).toFixed(1)}s)
              </span>
            </div>

            {/* Score and match type (if available) */}
            {clip.score !== undefined && (
              <div className="flex items-center gap-2 text-sm">
                <span className="text-charcoal-400">
                  {Math.round(clip.score * 100)}%
                </span>
                {clip.match_type && (
                  <span className="px-2 py-0.5 text-xs rounded bg-charcoal-700 text-charcoal-300">
                    {clip.match_type}
                  </span>
                )}
              </div>
            )}
          </motion.li>
        ))}
      </ul>
    </div>
  );
}
