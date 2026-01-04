/**
 * Search and indexing API routes.
 *
 * Handles index building, status checking, and semantic search queries.
 */
import { Elysia, t } from "elysia";
import { clipper } from "../python";

export const searchRoutes = new Elysia({ prefix: "/api/index" })
  /**
   * Build or load the search index for the current video.
   * This processes shot detection and optionally transcription.
   * Returns immediately if index already exists.
   */
  .post(
    "/build",
    async ({ body }) => {
      const result = await clipper.build_index(body.transcribe ?? true);
      return result;
    },
    {
      body: t.Object({
        transcribe: t.Optional(t.Boolean()),
      }),
    }
  )

  /**
   * Get current index status.
   * Use this to check if the index is ready before searching.
   */
  .get("/status", async () => {
    const status = await clipper.get_index_status();
    return status;
  })

  /**
   * Search for clips matching a natural language query.
   * Requires index to be built first.
   */
  .post(
    "/search",
    async ({ body }) => {
      const matches = await clipper.search(body.query, body.top_k ?? 5);
      return { matches };
    },
    {
      body: t.Object({
        query: t.String({ minLength: 1 }),
        top_k: t.Optional(t.Number({ minimum: 1, maximum: 50 })),
      }),
    }
  );
