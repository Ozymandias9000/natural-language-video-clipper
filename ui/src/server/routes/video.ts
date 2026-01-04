/**
 * Video-related API routes.
 *
 * Handles video loading, metadata retrieval, thumbnail generation, and streaming.
 */
import { Elysia, t } from "elysia";
import { clipper } from "../python";
import { statSync, existsSync, readdirSync } from "fs";
import { join } from "path";
import { homedir } from "os";

// Track loaded video path for streaming
let loadedVideoPath: string | null = null;

// Video file extensions to search for
const VIDEO_EXTENSIONS = new Set([".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"]);

/**
 * Recursively search for a file by name and size in a directory.
 * Returns the full path if found, null otherwise.
 */
function findFileRecursive(
  dir: string,
  filename: string,
  targetSize: number,
  maxDepth: number = 3,
  currentDepth: number = 0
): string | null {
  if (currentDepth > maxDepth) return null;

  try {
    const entries = readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = join(dir, entry.name);

      if (entry.isFile() && entry.name === filename) {
        try {
          const stat = statSync(fullPath);
          if (stat.size === targetSize) {
            return fullPath;
          }
        } catch {
          // Skip files we can't stat
        }
      } else if (entry.isDirectory() && !entry.name.startsWith(".")) {
        const found = findFileRecursive(fullPath, filename, targetSize, maxDepth, currentDepth + 1);
        if (found) return found;
      }
    }
  } catch {
    // Skip directories we can't read
  }

  return null;
}

/**
 * Search common directories for a video file matching name and size.
 */
function findVideoFile(filename: string, size: number): string | null {
  const home = homedir();
  const cwd = process.cwd();

  // Search these directories in order of likelihood
  const searchDirs = [
    join(home, "Movies"),
    join(home, "Downloads"),
    join(home, "Desktop"),
    join(home, "Documents"),
    join(home, "Videos"),
    cwd,
    join(cwd, ".."),
    join(cwd, "videos"),
    join(cwd, "..", "videos"),
    home,
    "/tmp",
  ];

  console.log(`[findVideoFile] Looking for "${filename}" (${size} bytes)`);
  console.log(`[findVideoFile] Searching directories:`, searchDirs.filter(d => existsSync(d)));

  for (const dir of searchDirs) {
    if (existsSync(dir)) {
      console.log(`[findVideoFile] Searching: ${dir}`);
      const found = findFileRecursive(dir, filename, size, 5);
      if (found) {
        console.log(`[findVideoFile] Found: ${found}`);
        return found;
      }
    }
  }

  console.log(`[findVideoFile] Not found in any directory`);
  return null;
}

export const videoRoutes = new Elysia({ prefix: "/api/video" })
  /**
   * Find a video file by name and size.
   * Searches common directories (Movies, Downloads, Desktop, Documents).
   */
  .post(
    "/find",
    async ({ body }) => {
      console.log("[API] POST /api/video/find", body.filename, body.size);
      const path = findVideoFile(body.filename, body.size);
      console.log("[API] Found path:", path);
      return { path };
    },
    {
      body: t.Object({
        filename: t.String({ minLength: 1 }),
        size: t.Number(),
      }),
    }
  )

  /**
   * Load a video file and return its metadata.
   * This must be called before any other video operations.
   */
  .post(
    "/load",
    async ({ body }) => {
      const decodedPath = decodeURIComponent(body.path);
      console.log("[API] POST /api/video/load", decodedPath);
      const info = await clipper.load_video(decodedPath);
      loadedVideoPath = info.path;
      console.log("[API] Video loaded:", info.path);
      return info;
    },
    {
      body: t.Object({
        path: t.String({ minLength: 1 }),
      }),
    }
  )

  /**
   * Get a single thumbnail at a specific timestamp.
   * Returns base64-encoded JPEG image data.
   */
  .get(
    "/thumbnail",
    async ({ query }) => {
      const time = parseFloat(query.time);
      const width = query.width ? parseInt(query.width) : 160;

      if (isNaN(time) || time < 0) {
        throw new Error("Invalid time parameter");
      }

      const base64 = await clipper.get_thumbnail(time, width);
      return { thumbnail: base64 };
    },
    {
      query: t.Object({
        time: t.String(),
        width: t.Optional(t.String()),
      }),
    }
  )

  /**
   * Get multiple thumbnails in a single request.
   * More efficient than making individual thumbnail requests.
   */
  .post(
    "/thumbnails",
    async ({ body }) => {
      const thumbnails = await clipper.get_thumbnails_batch(
        body.times,
        body.width ?? 160
      );
      return { thumbnails };
    },
    {
      body: t.Object({
        times: t.Array(t.Number()),
        width: t.Optional(t.Number()),
      }),
    }
  )

  /**
   * Stream video with support for range requests.
   * Enables video seeking in the browser player.
   */
  .get("/stream", async ({ set, headers }) => {
    if (!loadedVideoPath) {
      set.status = 400;
      return { error: "No video loaded" };
    }

    if (!existsSync(loadedVideoPath)) {
      set.status = 404;
      return { error: "Video file not found" };
    }

    const stat = statSync(loadedVideoPath);
    const fileSize = stat.size;
    const range = headers.range;

    // Determine content type from extension
    const ext = loadedVideoPath.split(".").pop()?.toLowerCase();
    const mimeTypes: Record<string, string> = {
      mp4: "video/mp4",
      webm: "video/webm",
      mov: "video/quicktime",
      avi: "video/x-msvideo",
      mkv: "video/x-matroska",
    };
    const contentType = mimeTypes[ext ?? ""] ?? "video/mp4";

    if (range) {
      // Handle range request for seeking support
      const parts = range.replace(/bytes=/, "").split("-");
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
      const chunkSize = end - start + 1;

      set.status = 206;
      set.headers["Content-Range"] = `bytes ${start}-${end}/${fileSize}`;
      set.headers["Accept-Ranges"] = "bytes";
      set.headers["Content-Length"] = chunkSize.toString();
      set.headers["Content-Type"] = contentType;

      return new Response(
        Bun.file(loadedVideoPath).slice(start, end + 1) as unknown as BodyInit,
        {
          status: 206,
          headers: {
            "Content-Range": `bytes ${start}-${end}/${fileSize}`,
            "Accept-Ranges": "bytes",
            "Content-Length": chunkSize.toString(),
            "Content-Type": contentType,
          },
        }
      );
    }

    // Full file response
    set.headers["Content-Length"] = fileSize.toString();
    set.headers["Content-Type"] = contentType;
    set.headers["Accept-Ranges"] = "bytes";

    return Bun.file(loadedVideoPath);
  });
