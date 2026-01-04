/**
 * Video-related API routes.
 *
 * Handles video loading, metadata retrieval, thumbnail generation, and streaming.
 */
import { Elysia, t } from "elysia";
import { clipper } from "../python";
import { createReadStream, statSync, existsSync } from "fs";

// Track loaded video path for streaming
let loadedVideoPath: string | null = null;

export const videoRoutes = new Elysia({ prefix: "/api/video" })
  /**
   * Load a video file and return its metadata.
   * This must be called before any other video operations.
   */
  .post(
    "/load",
    async ({ body }) => {
      const info = await clipper.load_video(body.path);
      loadedVideoPath = info.path;
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
