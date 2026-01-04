/**
 * Simple subprocess bridge for video clipper Python API.
 *
 * Spawns a Python process that reads JSON-RPC commands from stdin
 * and writes responses to stdout. Keeps models loaded between calls.
 */
import { spawn, type Subprocess } from "bun";
import { join } from "path";

// Navigate from ui/src/server/ up to project root
const projectRoot = join(import.meta.dir, "../../../");

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

class PythonBridge {
  private process: Subprocess<"pipe", "pipe", "pipe"> | null = null;
  private requestId = 0;
  private pendingRequests = new Map<
    number,
    { resolve: (value: unknown) => void; reject: (error: Error) => void }
  >();
  private buffer = "";
  private starting: Promise<void> | null = null;

  private async ensureStarted(): Promise<void> {
    if (this.process) return;

    if (this.starting) {
      await this.starting;
      return;
    }

    this.starting = this.start();
    await this.starting;
    this.starting = null;
  }

  private async start(): Promise<void> {
    this.process = spawn({
      cmd: ["python3", "-m", "src.video_clipper.bridge_server"],
      cwd: projectRoot,
      stdin: "pipe",
      stdout: "pipe",
      stderr: "pipe",
    });

    // Handle stdout - parse JSON responses
    this.readOutput();

    // Log stderr for debugging
    this.readStderr();
  }

  private async readOutput(): Promise<void> {
    if (!this.process?.stdout) return;

    const reader = this.process.stdout.getReader();
    const decoder = new TextDecoder();

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        this.buffer += decoder.decode(value, { stream: true });

        // Process complete lines
        let newlineIdx;
        while ((newlineIdx = this.buffer.indexOf("\n")) !== -1) {
          const line = this.buffer.slice(0, newlineIdx);
          this.buffer = this.buffer.slice(newlineIdx + 1);

          if (line.trim()) {
            this.handleResponse(line);
          }
        }
      }
    } catch (error) {
      console.error("Bridge stdout error:", error);
    }
  }

  private async readStderr(): Promise<void> {
    if (!this.process?.stderr) return;

    const reader = this.process.stderr.getReader();
    const decoder = new TextDecoder();

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const text = decoder.decode(value, { stream: true });
        if (text.trim()) {
          console.error("[Python]", text);
        }
      }
    } catch (error) {
      // Ignore stderr read errors on shutdown
    }
  }

  private handleResponse(line: string): void {
    try {
      const response = JSON.parse(line);
      const pending = this.pendingRequests.get(response.id);

      if (pending) {
        this.pendingRequests.delete(response.id);

        if (response.error) {
          const error = new Error(response.error.message);
          if (response.error.traceback) {
            console.error("Python traceback:", response.error.traceback);
          }
          pending.reject(error);
        } else {
          pending.resolve(response.result);
        }
      }
    } catch (error) {
      console.error("Failed to parse response:", line, error);
    }
  }

  async call<T>(method: string, ...params: unknown[]): Promise<T> {
    await this.ensureStarted();

    if (!this.process?.stdin) {
      throw new Error("Bridge process not running");
    }

    const id = ++this.requestId;
    const request = JSON.stringify({ id, method, params }) + "\n";

    return new Promise<T>((resolve, reject) => {
      this.pendingRequests.set(id, {
        resolve: resolve as (value: unknown) => void,
        reject,
      });

      this.process!.stdin.write(request);
    });
  }

  async close(): Promise<void> {
    if (this.process) {
      this.process.kill();
      this.process = null;
    }
  }
}

const bridge = new PythonBridge();

// Typed API wrapper
export const clipper = {
  load_video: (path: string) => bridge.call<VideoInfo>("load_video", path),

  build_index: (transcribe = true) =>
    bridge.call<BuildResult>("build_index", transcribe),

  get_index_status: () => bridge.call<IndexStatus>("get_index_status"),

  search: (query: string, top_k = 5) =>
    bridge.call<ClipMatch[]>("search", query, top_k),

  get_thumbnail: (time: number, width = 160) =>
    bridge.call<string>("get_thumbnail", time, width),

  get_thumbnails_batch: (times: number[], width = 160) =>
    bridge.call<string[]>("get_thumbnails_batch", times, width),

  export_clips: (
    clips: { start: number; end: number }[],
    output_dir: string,
    stitch = false
  ) => bridge.call<ExportResult>("export_clips", clips, output_dir, stitch),
};
