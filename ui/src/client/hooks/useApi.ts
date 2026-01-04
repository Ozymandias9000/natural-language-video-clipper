import { useState, useCallback } from "react";
import type { VideoInfo, ClipMatch, IndexStatus } from "../types";

const API_BASE = "/api";

interface UseApiReturn {
  /** Loading state for async operations */
  loading: boolean;
  /** Last error message, null if no error */
  error: string | null;
  /** Load a video file and retrieve metadata */
  loadVideo: (path: string) => Promise<VideoInfo>;
  /** Build or load the search index */
  buildIndex: (transcribe?: boolean) => Promise<IndexStatus>;
  /** Check current index status */
  getIndexStatus: () => Promise<IndexStatus>;
  /** Search for clips matching a query */
  search: (query: string, topK?: number) => Promise<ClipMatch[]>;
  /** Get a single thumbnail at a specific time */
  getThumbnail: (time: number, width?: number) => Promise<string>;
  /** Get multiple thumbnails in batch */
  getThumbnails: (times: number[], width?: number) => Promise<string[]>;
  /** Export selected clips to disk */
  exportClips: (
    clips: { start: number; end: number }[],
    stitch?: boolean,
    outputDir?: string
  ) => Promise<string[]>;
}

/**
 * Hook for interacting with the video clipper API.
 * Manages loading/error state and provides typed API methods.
 */
export function useApi(): UseApiReturn {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadVideo = useCallback(async (path: string): Promise<VideoInfo> => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/video/load`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path }),
      });
      if (!res.ok) throw new Error(await res.text());
      return await res.json();
    } catch (e) {
      const message = e instanceof Error ? e.message : "Unknown error";
      setError(message);
      throw e;
    } finally {
      setLoading(false);
    }
  }, []);

  const buildIndex = useCallback(
    async (transcribe = true): Promise<IndexStatus> => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API_BASE}/index/build`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ transcribe }),
        });
        if (!res.ok) throw new Error(await res.text());
        return await res.json();
      } catch (e) {
        const message = e instanceof Error ? e.message : "Unknown error";
        setError(message);
        throw e;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const getIndexStatus = useCallback(async (): Promise<IndexStatus> => {
    const res = await fetch(`${API_BASE}/index/status`);
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  }, []);

  const search = useCallback(
    async (query: string, topK = 5): Promise<ClipMatch[]> => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API_BASE}/search`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, top_k: topK }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        return data.matches;
      } catch (e) {
        const message = e instanceof Error ? e.message : "Unknown error";
        setError(message);
        throw e;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const getThumbnail = useCallback(
    async (time: number, width = 160): Promise<string> => {
      const res = await fetch(
        `${API_BASE}/video/thumbnail?t=${time}&w=${width}`
      );
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      return `data:image/jpeg;base64,${data.data}`;
    },
    []
  );

  const getThumbnails = useCallback(
    async (times: number[], width = 160): Promise<string[]> => {
      const res = await fetch(`${API_BASE}/video/thumbnails`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ times, width }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      return data.thumbnails.map(
        (t: string) => `data:image/jpeg;base64,${t}`
      );
    },
    []
  );

  const exportClips = useCallback(
    async (
      clips: { start: number; end: number }[],
      stitch = false,
      outputDir = "./clips"
    ): Promise<string[]> => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API_BASE}/export`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            clips,
            stitch,
            output_dir: outputDir,
          }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        return data.outputs;
      } catch (e) {
        const message = e instanceof Error ? e.message : "Unknown error";
        setError(message);
        throw e;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  return {
    loading,
    error,
    loadVideo,
    buildIndex,
    getIndexStatus,
    search,
    getThumbnail,
    getThumbnails,
    exportClips,
  };
}
