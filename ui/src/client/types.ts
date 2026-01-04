/**
 * Core type definitions for the video clipper UI.
 * These mirror the server-side types from the Python bridge.
 */

/** Video metadata returned from load_video API */
export interface VideoInfo {
  /** Total duration in seconds */
  duration: number;
  /** Frame rate (frames per second) */
  fps: number;
  /** Video width in pixels */
  width: number;
  /** Video height in pixels */
  height: number;
  /** Absolute path to the video file */
  path: string;
}

/** Search result from the semantic/text search API */
export interface ClipMatch {
  /** Start time in seconds */
  start: number;
  /** End time in seconds */
  end: number;
  /** Relevance score (0-1) */
  score: number;
  /** Type of match: "visual", "transcript", or "hybrid" */
  match_type: string;
  /** Matched transcript text if applicable */
  matched_text: string | null;
}

/** User-selected clip with UI state */
export interface Clip {
  /** Unique identifier for React key and state management */
  id: string;
  /** Start time in seconds */
  start: number;
  /** End time in seconds */
  end: number;
  /** Optional relevance score from search */
  score?: number;
  /** Optional match type from search */
  match_type?: string;
  /** Whether this clip is selected for export */
  selected: boolean;
}

/** Index build/status response */
export interface IndexStatus {
  /** Whether the index is ready for queries */
  ready: boolean;
  /** Number of detected shots in the index */
  shots: number;
  /** Number of transcript segments in the index */
  segments: number;
}
