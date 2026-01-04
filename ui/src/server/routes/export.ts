/**
 * Export API routes.
 *
 * Handles clip export operations including individual clips and stitched output.
 */
import { Elysia, t } from "elysia";
import { existsSync } from "fs";
import { basename } from "path";
import { clipper } from "../python";

// Temp directory for exports
const EXPORT_DIR = "/tmp/video-clipper-exports";

export const exportRoutes = new Elysia({ prefix: "/api/export" })
  /**
   * Export selected clips to the specified output directory.
   *
   * When stitch is true, combines all clips into a single video file.
   * When stitch is false (default), exports each clip as a separate file.
   */
  .post(
    "/clips",
    async ({ body }) => {
      const result = await clipper.export_clips(
        body.clips,
        EXPORT_DIR,
        body.stitch ?? false
      );
      // Return download URLs instead of file paths
      const downloads = result.outputs.map((path: string) => ({
        path,
        filename: basename(path),
        url: `/api/export/download?file=${encodeURIComponent(path)}`,
      }));
      return { outputs: downloads };
    },
    {
      body: t.Object({
        clips: t.Array(
          t.Object({
            start: t.Number({ minimum: 0 }),
            end: t.Number({ minimum: 0 }),
          })
        ),
        stitch: t.Optional(t.Boolean()),
      }),
    }
  )

  /**
   * Download an exported file.
   */
  .get(
    "/download",
    async ({ query, set }) => {
      const filePath = decodeURIComponent(query.file);

      // Security: only allow files from export dir
      if (!filePath.startsWith(EXPORT_DIR)) {
        set.status = 403;
        return { error: "Access denied" };
      }

      if (!existsSync(filePath)) {
        set.status = 404;
        return { error: "File not found" };
      }

      const filename = basename(filePath);
      set.headers["Content-Disposition"] = `attachment; filename="${filename}"`;
      set.headers["Content-Type"] = "video/mp4";

      return Bun.file(filePath);
    },
    {
      query: t.Object({
        file: t.String(),
      }),
    }
  );
