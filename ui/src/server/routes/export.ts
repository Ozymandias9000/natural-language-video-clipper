/**
 * Export API routes.
 *
 * Handles clip export operations including individual clips and stitched output.
 */
import { Elysia, t } from "elysia";
import { clipper } from "../python";

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
        body.output_dir,
        body.stitch ?? false
      );
      return result;
    },
    {
      body: t.Object({
        clips: t.Array(
          t.Object({
            start: t.Number({ minimum: 0 }),
            end: t.Number({ minimum: 0 }),
          })
        ),
        output_dir: t.String({ minLength: 1 }),
        stitch: t.Optional(t.Boolean()),
      }),
    }
  );
