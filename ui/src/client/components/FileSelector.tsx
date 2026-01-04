import { useState, useCallback } from "react";

interface FileSelectorProps {
  onSelect: (path: string) => void;
  onFindFile?: (filename: string, size: number) => Promise<string | null>;
  loading?: boolean;
  status?: string;
}

export function FileSelector({ onSelect, onFindFile, loading, status }: FileSelectorProps) {
  const [path, setPath] = useState("");
  const [browsing, setBrowsing] = useState(false);

  const handleBrowse = useCallback(async () => {
    // Check if File System Access API is available
    if (!("showOpenFilePicker" in window)) {
      alert("File picker not supported in this browser. Please use Chrome, Edge, or Arc.");
      return;
    }

    try {
      setBrowsing(true);
      const [fileHandle] = await (window as any).showOpenFilePicker({
        types: [
          {
            description: "Video files",
            accept: {
              "video/*": [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"],
            },
          },
        ],
        multiple: false,
      });

      const file = await fileHandle.getFile();

      // Try to find the file on the backend
      if (onFindFile) {
        const foundPath = await onFindFile(file.name, file.size);
        if (foundPath) {
          onSelect(foundPath);
          return;
        }
      }

      // Fallback: populate the text input with a template path
      // User can edit the directory portion
      setPath(`/Users/you/path/to/${file.name}`);
    } catch (err: any) {
      // User cancelled the picker
      if (err.name !== "AbortError") {
        console.error("File picker error:", err);
      }
    } finally {
      setBrowsing(false);
    }
  }, [onSelect, onFindFile]);

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
            Select a video file to analyze
          </p>
          <button
            type="button"
            onClick={handleBrowse}
            disabled={browsing}
            className="w-full mb-4 px-6 py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-blue-800 rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
          >
            {browsing ? (
              <>
                <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Searching...
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                </svg>
                Browse Files
              </>
            )}
          </button>
          <div className="relative flex items-center gap-4 mb-4">
            <div className="flex-1 border-t border-charcoal-700" />
            <span className="text-charcoal-600 text-xs">or paste path</span>
            <div className="flex-1 border-t border-charcoal-700" />
          </div>
          <div className="flex gap-2">
            <input
              type="text"
              value={path}
              onChange={(e) => setPath(e.target.value)}
              placeholder="/Users/you/videos/example.mp4"
              className="flex-1 px-4 py-2 bg-charcoal-800 border border-charcoal-700 rounded-lg text-white placeholder-charcoal-500 focus:outline-none focus:border-charcoal-500 font-mono text-sm"
            />
            <button
              type="submit"
              disabled={!path.trim() || !path.startsWith("/")}
              className="px-6 py-2 bg-charcoal-700 hover:bg-charcoal-600 disabled:bg-charcoal-800 disabled:text-charcoal-600 rounded-lg font-medium transition-colors"
            >
              Open
            </button>
          </div>
        </div>
      )}
    </form>
  );
}
