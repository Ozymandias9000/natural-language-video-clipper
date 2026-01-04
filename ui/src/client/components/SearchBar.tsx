import { useState, type FormEvent } from "react";
import { motion } from "motion/react";

interface SearchBarProps {
  /** Async handler called with trimmed query on form submission */
  onSearch: (query: string) => Promise<void>;
  /** Shows loading spinner when true */
  loading: boolean;
  /** Disables input and button when true */
  disabled: boolean;
}

/**
 * Search bar for querying video clips.
 * Submits trimmed query to parent handler on form submission.
 */
export function SearchBar({ onSearch, loading, disabled }: SearchBarProps) {
  const [query, setQuery] = useState("");

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const trimmed = query.trim();
    if (trimmed && !loading && !disabled) {
      await onSearch(trimmed);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="relative w-full">
      <div className="relative">
        {/* Search icon */}
        <div className="absolute left-3 top-1/2 -translate-y-1/2 text-charcoal-500 pointer-events-none">
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
              <circle cx="11" cy="11" r="8" />
              <path d="m21 21-4.35-4.35" />
            </svg>
          )}
        </div>

        {/* Input field */}
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search for clips..."
          disabled={disabled || loading}
          className="w-full pl-10 pr-4 py-2.5 bg-charcoal-800 border border-charcoal-600 rounded-lg text-white placeholder-charcoal-500 focus:outline-none focus:border-amber-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        />
      </div>
    </form>
  );
}
