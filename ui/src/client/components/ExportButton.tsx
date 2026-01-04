import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "motion/react";

interface ExportButtonProps {
  /** Number of selected clips to export */
  selectedCount: number;
  /** Async handler called when user initiates export. `stitch` is true when combining clips. */
  onExport: (stitch: boolean) => Promise<void>;
  /** Shows loading spinner and disables interaction when true */
  loading: boolean;
}

/**
 * Export button with dropdown menu for export options.
 * Allows exporting clips individually or stitched into a single video.
 */
export function ExportButton({
  selectedCount,
  onExport,
  loading,
}: ExportButtonProps) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const disabled = selectedCount === 0 || loading;

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [isOpen]);

  // Close dropdown on Escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener("keydown", handleEscape);
      return () => document.removeEventListener("keydown", handleEscape);
    }
  }, [isOpen]);

  const handleExport = async (stitch: boolean) => {
    setIsOpen(false);
    await onExport(stitch);
  };

  return (
    <div ref={containerRef} className="relative">
      {/* Main button */}
      <button
        type="button"
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
          disabled
            ? "bg-charcoal-700 text-charcoal-500 cursor-not-allowed"
            : "bg-amber-500 text-charcoal-900 hover:bg-amber-400"
        }`}
      >
        {loading ? (
          <motion.svg
            animate={{ rotate: 360 }}
            transition={{
              duration: 1,
              repeat: Infinity,
              ease: "linear",
            }}
            className="w-5 h-5"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M12 2v4m0 12v4m10-10h-4M6 12H2m15.07-5.07l-2.83 2.83M8.76 15.24l-2.83 2.83m11.14 0l-2.83-2.83M8.76 8.76L5.93 5.93" />
          </motion.svg>
        ) : (
          <svg
            className="w-5 h-5"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
        )}
        <span>Export ({selectedCount})</span>
        {/* Dropdown chevron */}
        <svg
          className={`w-4 h-4 transition-transform ${isOpen ? "rotate-180" : ""}`}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      {/* Dropdown menu */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.15 }}
            className="absolute right-0 mt-2 w-48 bg-charcoal-800 border border-charcoal-600 rounded-lg shadow-lg overflow-hidden z-50"
          >
            <button
              type="button"
              onClick={() => handleExport(false)}
              className="w-full flex items-center gap-3 px-4 py-3 text-left text-white hover:bg-charcoal-700 transition-colors"
            >
              <svg
                className="w-5 h-5 text-charcoal-400"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <rect x="3" y="3" width="7" height="7" />
                <rect x="14" y="3" width="7" height="7" />
                <rect x="3" y="14" width="7" height="7" />
                <rect x="14" y="14" width="7" height="7" />
              </svg>
              <span>Individual clips</span>
            </button>
            <button
              type="button"
              onClick={() => handleExport(true)}
              className="w-full flex items-center gap-3 px-4 py-3 text-left text-white hover:bg-charcoal-700 transition-colors border-t border-charcoal-700"
            >
              <svg
                className="w-5 h-5 text-charcoal-400"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <rect x="2" y="7" width="20" height="10" rx="2" />
                <path d="M10 12h4" />
              </svg>
              <span>Stitch into one</span>
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
