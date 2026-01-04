/**
 * Video Clipper API Server
 *
 * Elysia server providing REST API endpoints for:
 * - Video loading and streaming
 * - Thumbnail generation
 * - Search index building and querying
 * - Clip export
 *
 * The server uses PyBridge to communicate with the Python video clipper backend,
 * keeping ML models loaded in memory for fast inference.
 */
import { Elysia } from "elysia";
import { cors } from "@elysiajs/cors";
import { staticPlugin } from "@elysiajs/static";
import { join } from "path";

import { videoRoutes } from "./routes/video";
import { searchRoutes } from "./routes/search";
import { exportRoutes } from "./routes/export";

const PORT = 3000;

const app = new Elysia()
  // Enable CORS for development - allows Vite dev server to access API
  .use(
    cors({
      origin: true,
      methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
      allowedHeaders: ["Content-Type", "Authorization", "Range"],
      exposeHeaders: ["Content-Range", "Accept-Ranges", "Content-Length"],
    })
  )

  // Serve static files from the dist directory in production
  .use(
    staticPlugin({
      assets: join(import.meta.dir, "../../dist"),
      prefix: "/",
    })
  )

  // Health check endpoint
  .get("/health", () => ({ status: "ok", timestamp: new Date().toISOString() }))

  // Mount API routes
  .use(videoRoutes)
  .use(searchRoutes)
  .use(exportRoutes)

  // Global error handler
  .onError(({ code, error, set }) => {
    console.error(`[${code}] ${error.message}`);

    if (code === "VALIDATION") {
      set.status = 400;
      return {
        error: "Validation error",
        details: error.message,
      };
    }

    if (code === "NOT_FOUND") {
      set.status = 404;
      return { error: "Not found" };
    }

    set.status = 500;
    return {
      error: "Internal server error",
      message:
        process.env.NODE_ENV === "development" ? error.message : undefined,
    };
  })

  .listen(PORT);

console.log(
  `Video Clipper server running at http://localhost:${app.server?.port}`
);

export type App = typeof app;
