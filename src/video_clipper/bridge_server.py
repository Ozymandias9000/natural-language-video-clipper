"""Simple JSON-RPC style bridge server.

Reads JSON commands from stdin, executes them, and writes JSON responses to stdout.
Keeps the process running to maintain loaded models between calls.
"""
import json
import sys
import time
import traceback

from . import bridge


def log(msg: str) -> None:
    """Log to stderr (visible in server console)."""
    print(f"[Python] {msg}", file=sys.stderr, flush=True)


def main():
    """Main loop: read commands, execute, respond."""
    # Flush stdout immediately for responsive communication
    sys.stdout.reconfigure(line_buffering=True)

    log("Bridge server started")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            method = request.get("method")
            params = request.get("params", [])
            request_id = request.get("id")

            log(f"→ {method}({params})")
            start = time.time()

            # Get the function from bridge module
            fn = getattr(bridge, method, None)
            if fn is None:
                response = {
                    "id": request_id,
                    "error": {"message": f"Unknown method: {method}"},
                }
                log(f"✗ Unknown method: {method}")
            else:
                # Call with positional args if list, kwargs if dict
                if isinstance(params, list):
                    result = fn(*params)
                else:
                    result = fn(**params)
                response = {"id": request_id, "result": result}
                elapsed = time.time() - start
                log(f"✓ {method} completed in {elapsed:.2f}s")

        except Exception as e:
            response = {
                "id": request.get("id") if "request" in dir() else None,
                "error": {
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                },
            }
            log(f"✗ {method} failed: {e}")

        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
