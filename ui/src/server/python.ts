/**
 * PyBridge wrapper for video clipper Python API.
 *
 * Provides typed access to the Python video clipper functionality via PyBridge.
 * The bridge maintains a persistent Python process to keep models loaded between calls.
 */
import { PyBridge } from "pybridge";
import { join } from "path";

// Navigate from ui/src/server/ up to project root
const projectRoot = join(import.meta.dir, "../../../");

const bridge = new PyBridge({
  python: "python3",
  cwd: projectRoot,
});

export interface VideoInfo {
  duration: number;
  fps: number;
  width: number;
  height: number;
  path: string;
}

export interface IndexStatus {
  ready: boolean;
  shots: number;
  segments: number;
}

export interface BuildResult {
  status: string;
  shots: number;
  segments: number;
}

export interface ClipMatch {
  start: number;
  end: number;
  score: number;
  match_type: string;
  matched_text: string | null;
}

export interface ExportResult {
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

export const clipper =
  bridge.controller<VideoClipperAPI>("src/video_clipper/bridge.py");
