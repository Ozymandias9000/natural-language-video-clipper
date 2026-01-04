import { useState, useCallback } from "react";

interface FileSelectorProps {
  onSelect: (path: string) => void;
  loading?: boolean;
  status?: string;
}

export function FileSelector({ onSelect, loading, status }: FileSelectorProps) {
  const [path, setPath] = useState("");

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (path.trim() && path.startsWith("/")) {
        onSelect(path.trim());
      }
    },
    [path, onSelect]
  );

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-xl">
      {loading ? (
        <div className="border-2 border-dashed border-charcoal-700 rounded-lg p-8 text-center">
          <div className="mb-4">
            <svg
              className="w-12 h-12 mx-auto text-blue-500 animate-spin"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
          </div>
          <p className="text-charcoal-400">{status || "Loading..."}</p>
          <p className="text-charcoal-500 text-sm mt-2 font-mono">{path}</p>
        </div>
      ) : (
        <div className="border-2 border-dashed border-charcoal-700 rounded-lg p-8 text-center hover:border-charcoal-500 transition-colors">
          <div className="mb-4">
            <svg
              className="w-12 h-12 mx-auto text-charcoal-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
              />
            </svg>
          </div>
          <p className="text-charcoal-400 mb-4">
            Paste the full path to a video file
          </p>
          <div className="flex gap-2">
            <input
              type="text"
              value={path}
              onChange={(e) => setPath(e.target.value)}
              placeholder="/Users/you/videos/example.mp4"
              className="flex-1 px-4 py-2 bg-charcoal-800 border border-charcoal-700 rounded-lg text-white placeholder-charcoal-500 focus:outline-none focus:border-charcoal-500 font-mono text-sm"
              autoFocus
            />
            <button
              type="submit"
              disabled={!path.trim() || !path.startsWith("/")}
              className="px-6 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-charcoal-800 disabled:text-charcoal-600 rounded-lg font-medium transition-colors"
            >
              Open
            </button>
          </div>
          <p className="text-charcoal-600 text-xs mt-3">
            Tip: In Finder, right-click a file → Hold Option → "Copy as Pathname"
          </p>
        </div>
      )}
    </form>
  );
}
